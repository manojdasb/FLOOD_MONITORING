"""Module A - Event Broker & Data Engine.

Run:  uvicorn stream_server:app --host 0.0.0.0 --port 8000

Serves Server-Sent Events at /api/v1/flood-stream/{zone_id}, 1 event/second.
Hydrology is simulated as a deterministic function of wall-clock time, so every
connected client sees the same series and the server holds no per-client state.
Replace `simulate()` with a real CWC/India-WRIS feed adapter for production.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logger = logging.getLogger("flood.stream")
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class Zone:
    zone_id: str
    name: str
    river_name: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    station_id: str
    station_lat: float  # approximate gauge position (inside the AOI)
    station_lon: float
    danger_level: float
    warning_level: Optional[float]
    # --- simulation parameters ---
    base_level: float    # mean seasonal level (m)
    amplitude: float     # surge amplitude (m)
    period_s: float      # seconds per full surge cycle
    flood_k: float       # area coefficient (km2 / m^1.7)


ZONES: dict[str, Zone] = {
    "zone1": Zone(
        "zone1", "Brahmaputra River Basin (Guwahati, Assam)", "Brahmaputra",
        (91.50, 26.10, 91.90, 26.30), "Guwahati_CWC", 26.185, 91.745,
        danger_level=49.68, warning_level=48.68,
        base_level=48.3, amplitude=1.6, period_s=240.0, flood_k=12.0,
    ),
    "zone2": Zone(
        "zone2", "Kosi River Catchment (North Bihar)", "Kosi",
        (86.30, 25.50, 86.90, 26.20), "Baltara_CWC", 25.58, 86.62,
        danger_level=33.85, warning_level=None,
        base_level=33.0, amplitude=1.1, period_s=200.0, flood_k=40.0,
    ),
}


def simulate(zone: Zone, now: float) -> dict:
    """Deterministic surge model: seasonal sinusoid + two incommensurate ripples."""
    phase = 2.0 * math.pi * now / zone.period_s
    ripple = 0.02 * math.sin(now * 1.7) + 0.015 * math.sin(now * 0.63)
    level = zone.base_level + zone.amplitude * math.sin(phase) + ripple

    # Inundation begins ~1.5 m below the danger mark; area grows super-linearly.
    depth = max(0.0, level - (zone.danger_level - 1.5))
    area_km2 = 0.5 + zone.flood_k * depth ** 1.7

    # Footprint centroid drifts off the gauge as the surge expands.
    lat = zone.station_lat + 0.012 * depth
    lon = zone.station_lon - 0.020 * depth
    return {"level": level, "area": area_km2, "lat": lat, "lon": lon}


def build_event(zone: Zone, seq: int) -> dict:
    now = time.time()
    s = simulate(zone, now)
    return {
        "seq": seq,
        "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "station_id": zone.station_id,
        "river_name": zone.river_name,
        "telemetry": {
            "water_level_meters": round(s["level"], 3),
            "danger_level_meters": zone.danger_level,
            "warning_level_meters": zone.warning_level,
        },
        "status": "CRITICAL" if s["level"] >= zone.danger_level else "NOMINAL",
        "geospatial": {
            "lat": round(s["lat"], 5),
            "lon": round(s["lon"], 5),
            "flooded_area_km2": round(s["area"], 2),
        },
        "zone": {"zone_id": zone.zone_id, "name": zone.name, "bbox": list(zone.bbox),
                 "station_lat": zone.station_lat, "station_lon": zone.station_lon},
    }


async def event_stream(zone: Zone, request: Request) -> AsyncIterator[str]:
    seq = 0
    logger.info("client connected: %s", zone.zone_id)
    try:
        while True:
            if await request.is_disconnected():
                break
            seq += 1
            payload = json.dumps(build_event(zone, seq), separators=(",", ":"))
            yield f"id: {seq}\nevent: telemetry\ndata: {payload}\n\n"
            # Align to the next whole second to avoid cumulative drift.
            await asyncio.sleep(1.0 - (time.time() % 1.0))
    except asyncio.CancelledError:  # server shutdown / client abort
        raise
    except Exception:
        logger.exception("stream error for %s", zone.zone_id)
    finally:
        logger.info("client disconnected: %s", zone.zone_id)


app = FastAPI(title="India Flood Event Broker", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "zones": list(ZONES)}


@app.get("/api/v1/zones")
async def list_zones() -> list[dict]:
    return [{"zone_id": z.zone_id, "name": z.name, "bbox": list(z.bbox),
             "station_id": z.station_id} for z in ZONES.values()]


@app.get("/api/v1/flood-stream/{zone_id}")
async def flood_stream(zone_id: str, request: Request) -> StreamingResponse:
    zone = ZONES.get(zone_id)
    if zone is None:
        raise HTTPException(404, f"Unknown zone '{zone_id}'. Valid: {list(ZONES)}")
    return StreamingResponse(
        event_stream(zone, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )
