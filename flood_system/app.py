"""Module C - Streamlit dashboard.

Run:  streamlit run app.py
A daemon thread consumes the SSE feed (with exponential-backoff auto-reconnect);
a 1-second st.fragment repaints the UI from the thread's shared buffer.
"""
from __future__ import annotations

import json
import math
import threading
from collections import deque
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd
import pydeck as pdk
import streamlit as st

ZONE_LABELS = {"zone1": "Zone 1 - Brahmaputra (Guwahati)", "zone2": "Zone 2 - Kosi (North Bihar)"}


class StreamConsumer:
    """Background SSE client with auto-reconnect; thread-safe snapshot access."""

    def __init__(self, url: str, history: int = 30):
        self.url = url
        self._buf: deque = deque(maxlen=history)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.state, self.error, self.retries = "CONNECTING", None, 0
        threading.Thread(target=self._run, daemon=True, name=f"sse-{url}").start()

    def _set(self, state: str, error: Optional[str] = None) -> None:
        with self._lock:
            self.state, self.error = state, error

    def _ingest(self, raw: str) -> None:
        try:
            evt = json.loads(raw)
            _ = evt["telemetry"]["water_level_meters"], evt["geospatial"]["flooded_area_km2"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return  # drop malformed frame, keep stream alive
        with self._lock:
            self._buf.append(evt)

    def _run(self) -> None:
        backoff = 1.0
        timeout = httpx.Timeout(5.0, read=10.0)  # server emits every 1s -> 10s = dead
        while not self._stop.is_set():
            try:
                with httpx.Client(timeout=timeout) as c, c.stream(
                        "GET", self.url, headers={"Accept": "text/event-stream"}) as r:
                    r.raise_for_status()
                    self._set("LIVE")
                    backoff = 1.0
                    for line in r.iter_lines():
                        if self._stop.is_set():
                            return
                        if line.startswith("data:"):
                            self._ingest(line[5:].strip())
                self._set("RECONNECTING", "stream closed by server")
            except (httpx.HTTPError, OSError) as exc:
                self._set("RECONNECTING", f"{type(exc).__name__}: {exc}")
            self.retries += 1
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 15.0)

    def snapshot(self) -> tuple[str, Optional[str], list[dict]]:
        with self._lock:
            return self.state, self.error, list(self._buf)


@st.cache_resource
def get_consumer(url: str) -> StreamConsumer:
    return StreamConsumer(url)


def backend_healthy(base_url: str) -> bool:
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        response = httpx.get(health_url, timeout=2.0)
        if response.status_code != 200:
            return False
        payload = response.json()
        return bool(payload.get("status") == "ok")
    except (httpx.HTTPError, OSError, ValueError, TypeError):
        return False


CSS = """
<style>
@keyframes flash {0%,100%{background:#b00020;} 50%{background:#ff5252;}}
.alert-critical{animation:flash 1s infinite;color:#fff;padding:14px 18px;border-radius:8px;
  font-weight:700;font-size:1.1rem;margin-bottom:12px;}
.alert-offline{background:#8a6d00;color:#fff;padding:10px 18px;border-radius:8px;margin-bottom:12px;}
</style>
"""


def zoom_for(bbox: list[float]) -> float:
    span = max(bbox[2] - bbox[0], (bbox[3] - bbox[1]))
    return max(5.0, min(12.0, math.log2(360 / span) - 0.5))


def render_map(evt: dict) -> pdk.Deck:
    z, g = evt["zone"], evt["geospatial"]
    x0, y0, x1, y1 = z["bbox"]
    critical = evt["status"] == "CRITICAL"
    radius_m = math.sqrt(max(g["flooded_area_km2"], 0.01) / math.pi) * 1000.0
    layers = [
        pdk.Layer("PolygonLayer", [{"polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}],
                  get_polygon="polygon", filled=False, stroked=True,
                  get_line_color=[120, 120, 120], line_width_min_pixels=1),
        pdk.Layer("ScatterplotLayer",
                  [{"position": [g["lon"], g["lat"]], "area": g["flooded_area_km2"]}],
                  get_position="position", get_radius=radius_m,
                  get_fill_color=[220, 20, 20, 90], get_line_color=[220, 20, 20, 220],
                  stroked=True, line_width_min_pixels=2, pickable=True),
        pdk.Layer("ScatterplotLayer",
                  [{"position": [z["station_lon"], z["station_lat"]], "name": evt["station_id"],
                    "level": evt["telemetry"]["water_level_meters"]}],
                  get_position="position", get_radius=700, radius_min_pixels=7,
                  get_fill_color=[255, 40, 40] if critical else [30, 144, 255],
                  get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=2,
                  pickable=True),
    ]
    view = pdk.ViewState(longitude=(x0 + x1) / 2, latitude=(y0 + y1) / 2,
                         zoom=zoom_for(z["bbox"]), pitch=0)
    return pdk.Deck(layers=layers, initial_view_state=view, map_style="light",
                    tooltip={"text": "{name}\nLevel: {level} m\nFlooded: {area} km2"})


@st.fragment(run_every=1)
def live_panel(url: str) -> None:
    state, error, events = get_consumer(url).snapshot()
    if state != "LIVE":
        st.markdown(f'<div class="alert-offline">Stream {state}: {error or "waiting for data"} '
                    f"- retrying automatically.</div>", unsafe_allow_html=True)
    if not events:
        st.info("Waiting for first telemetry event...")
        return

    evt = events[-1]
    prev = events[-2] if len(events) > 1 else evt
    t, g = evt["telemetry"], evt["geospatial"]
    level, danger, warn = t["water_level_meters"], t["danger_level_meters"], t["warning_level_meters"]
    critical = evt["status"] == "CRITICAL"

    if critical:
        st.markdown(f'<div class="alert-critical">CRITICAL FLOOD ALERT - {evt["station_id"]} '
                    f'({evt["river_name"]}) is {level - danger:+.2f} m above the danger level.</div>',
                    unsafe_allow_html=True)

    cols = st.columns(5 if warn is not None else 4)
    cols[0].metric("Water level", f"{level:.2f} m", f"{level - prev['telemetry']['water_level_meters']:+.3f} m/s",
                   delta_color="inverse")
    cols[1].metric("Vs danger level", f"{level - danger:+.2f} m", f"limit {danger:.2f} m", delta_color="off")
    i = 2
    if warn is not None:
        cols[i].metric("Vs warning level", f"{level - warn:+.2f} m", f"limit {warn:.2f} m", delta_color="off")
        i += 1
    cols[i].metric("Flooded area", f"{g['flooded_area_km2']:.1f} km2",
                   f"{g['flooded_area_km2'] - prev['geospatial']['flooded_area_km2']:+.2f}", delta_color="inverse")
    cols[i + 1].metric("Status", evt["status"], datetime.fromisoformat(evt["timestamp"]).strftime("%H:%M:%S UTC"),
                       delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        st.pydeck_chart(render_map(evt), use_container_width=True)
    with right:
        df = pd.DataFrame({
            "time": [datetime.fromisoformat(e["timestamp"]) for e in events],
            "Water level (m)": [e["telemetry"]["water_level_meters"] for e in events],
            "Danger level (m)": [e["telemetry"]["danger_level_meters"] for e in events],
        }).set_index("time")
        st.caption("Last 30 ticks")
        st.line_chart(df)


def main() -> None:
    st.set_page_config(page_title="India Flood Monitor", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Real-Time Flood Monitor - India")
    with st.sidebar:
        server = st.text_input("Event broker URL", "http://localhost:8000").rstrip("/")
        zone = st.selectbox("Zone", list(ZONE_LABELS), format_func=ZONE_LABELS.get)

    if not backend_healthy(server):
        st.warning("Flood data broker is not reachable. Start the backend service before using the dashboard.")
        st.code(
            "cd C:\\Users\\Admin\\Downloads\\flood_system\\flood_system\n"
            "..\\floodenv\\Scripts\\Activate.ps1\n"
            "uvicorn stream_server:app --host 0.0.0.0 --port 8000",
            language="powershell",
        )
        st.stop()

    live_panel(f"{server}/api/v1/flood-stream/{zone}")


if __name__ == "__main__":
    main()
