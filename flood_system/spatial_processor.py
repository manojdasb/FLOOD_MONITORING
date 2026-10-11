"""Module B - Cloud Geospatial Core (Sentinel-1 GRD water extraction).

CLI:  python spatial_processor.py zone1
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import planetary_computer
import pystac
import rasterio
import requests
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds

logger = logging.getLogger("flood.spatial")

try:  # optional NVIDIA GPU backend (pip install cupy-cuda12x)
    import cupy as cp
except ImportError:
    cp = None


def gpu_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # driver missing / no CUDA device
        return False

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-grd"

AOIS: dict[str, tuple[float, float, float, float]] = {
    "zone1": (91.50, 26.10, 91.90, 26.30),  # Brahmaputra, Guwahati
    "zone2": (86.30, 25.50, 86.90, 26.20),  # Kosi, North Bihar
}


class SpatialProcessingError(RuntimeError):
    """Raised for STAC search or raster-read failures."""


@dataclass
class WaterMaskResult:
    item_id: str
    acquired_at: datetime
    bbox: tuple[float, float, float, float]
    transform: Affine
    db: np.ndarray            # float32, NaN where no data
    water_mask: np.ndarray    # uint8, 1 = water
    threshold_db: float
    water_fraction: float
    water_area_km2: float


class SentinelFloodMapper:
    def __init__(self, bbox, lookback_days: int = 14, threshold_db: float = -16.0,
                 polarization: str = "vv", max_dim: int = 2048,
                 db_scale: float = 10.0, retries: int = 3, use_gpu: bool = True):
        self.bbox = tuple(bbox)
        self.lookback_days = lookback_days
        self.threshold_db = threshold_db
        self.polarization = polarization.lower()
        self.max_dim = max_dim
        self.db_scale = db_scale  # 10 per spec; use 20 for strict amplitude->dB
        self.retries = retries
        self.use_gpu = use_gpu and gpu_available()
        logger.info("compute backend: %s", "CuPy/CUDA" if self.use_gpu else "NumPy/CPU")
        self._catalog: Optional[Client] = None

    # ------------------------------------------------------------- STAC search
    def _get_catalog(self) -> Client:
        if self._catalog is None:
            self._catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        return self._catalog

    def _covers_aoi(self, item: pystac.Item) -> bool:
        if not item.bbox:
            return False
        x0, y0, x1, y1 = item.bbox
        a0, b0, a1, b1 = self.bbox
        return x0 <= a0 and y0 <= b0 and x1 >= a1 and y1 >= b1

    def find_latest_item(self) -> pystac.Item:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.lookback_days)
        window = f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                search = self._get_catalog().search(
                    collections=[COLLECTION], bbox=self.bbox, datetime=window)
                items = [i for i in search.items() if self.polarization in i.assets]
                break
            except (APIError, requests.RequestException) as exc:
                last_exc = exc
                logger.warning("STAC search attempt %d/%d failed: %s", attempt, self.retries, exc)
                time.sleep(2 ** attempt)
        else:
            raise SpatialProcessingError(f"STAC search failed: {last_exc}") from last_exc

        if not items:
            raise SpatialProcessingError(
                f"No {COLLECTION} scenes with '{self.polarization}' over {self.bbox} in last "
                f"{self.lookback_days} days")
        items.sort(key=lambda i: i.datetime, reverse=True)
        full = [i for i in items if self._covers_aoi(i)]
        return (full or items)[0]  # prefer newest scene fully covering the AOI

    # ------------------------------------------------------------ COG windowing
    def _read_window(self, href: str) -> tuple[np.ndarray, Affine]:
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                   CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                   GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1")
        try:
            with rasterio.Env(**env), rasterio.open(href) as src:
                gcps, gcp_crs = src.gcps
                # PC Sentinel-1 GRD is in radar geometry, georeferenced by GCPs only:
                # warp lazily to EPSG:4326 so a lon/lat window is meaningful.
                ds = (WarpedVRT(src, src_crs=gcp_crs, crs="EPSG:4326", nodata=0,
                                resampling=Resampling.bilinear) if gcps else src)
                try:
                    win = from_bounds(*self.bbox, transform=ds.transform)
                    win = win.intersection(Window(0, 0, ds.width, ds.height))
                    h, w = int(win.height), int(win.width)
                    if h < 1 or w < 1:
                        raise SpatialProcessingError("AOI does not intersect the scene raster")
                    scale = min(1.0, self.max_dim / max(h, w))  # decimate via COG overviews
                    out = (max(1, round(h * scale)), max(1, round(w * scale)))
                    arr = ds.read(1, window=win, out_shape=out,
                                  resampling=Resampling.average).astype("float32")
                    transform = ds.window_transform(win) * Affine.scale(win.width / out[1],
                                                                       win.height / out[0])
                    return arr, transform
                finally:
                    if ds is not src:
                        ds.close()
        except (RasterioError, ValueError, OSError) as exc:
            raise SpatialProcessingError(f"Raster read failed: {exc}") from exc

    # ---------------------------------------------------------------- vector math
    def _to_water_mask(self, amp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.use_gpu:
            try:
                return self._to_water_mask_xp(amp, cp)
            except Exception as exc:  # CUDA OOM / driver error -> CPU fallback
                logger.warning("GPU path failed (%s); falling back to CPU", exc)
        return self._to_water_mask_xp(amp, np)

    def _to_water_mask_xp(self, amp, xp) -> tuple[np.ndarray, np.ndarray]:
        """Array-API style: identical code runs on NumPy (CPU) or CuPy (GPU)."""
        a = xp.asarray(amp)
        valid = xp.isfinite(a) & (a > 0)
        safe = xp.where(valid, a, 1.0)  # avoid log10(0) warnings
        db = (self.db_scale * xp.log10(safe)).astype("float32")
        db = xp.where(valid, db, xp.nan).astype("float32")
        mask = ((db < self.threshold_db) & valid).astype("uint8")
        if xp is not np:
            db, mask = cp.asnumpy(db), cp.asnumpy(mask)
        return db, mask

    def _pixel_area_km2(self, transform: Affine) -> float:
        lat = (self.bbox[1] + self.bbox[3]) / 2
        dx = abs(transform.a) * 111.320 * math.cos(math.radians(lat))
        dy = abs(transform.e) * 110.574
        return dx * dy

    # ------------------------------------------------------------------ public API
    def analyze(self) -> WaterMaskResult:
        item = self.find_latest_item()
        try:
            href = item.assets[self.polarization].href  # already SAS-signed
            amp, transform = self._read_window(href)
            db, mask = self._to_water_mask(amp)
            n_valid = int(np.isfinite(db).sum())
            if n_valid == 0:
                raise SpatialProcessingError("Window contains no valid pixels")
            n_water = int(mask.sum())
        except SpatialProcessingError:
            raise
        except (KeyError, ValueError, FloatingPointError, MemoryError) as exc:
            raise SpatialProcessingError(f"Matrix processing failed: {exc}") from exc
        return WaterMaskResult(
            item_id=item.id, acquired_at=item.datetime, bbox=self.bbox, transform=transform,
            db=db, water_mask=mask, threshold_db=self.threshold_db,
            water_fraction=n_water / n_valid,
            water_area_km2=n_water * self._pixel_area_km2(transform))

    async def analyze_async(self) -> WaterMaskResult:
        """Non-blocking wrapper: rasterio/pystac are sync, so run in a worker thread."""
        return await asyncio.to_thread(self.analyze)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("zone", choices=AOIS)
    p.add_argument("--cpu", action="store_true", help="force NumPy even if a GPU exists")
    args = p.parse_args()
    res = asyncio.run(SentinelFloodMapper(AOIS[args.zone], use_gpu=not args.cpu).analyze_async())
    print(f"scene={res.item_id} acquired={res.acquired_at:%Y-%m-%d %H:%MZ} "
          f"shape={res.water_mask.shape} water={res.water_fraction:.1%} "
          f"(~{res.water_area_km2:.1f} km2)")
