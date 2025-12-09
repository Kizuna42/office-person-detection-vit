"""Streamlit ダッシュボード用のデータローダと UI ユーティリティ."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


PHASE_DIR_CANDIDATES = {
    "extraction": ["01_extraction", "phase1_extraction"],
    "detection": ["02_detection", "phase2_detection"],
    "tracking": ["03_tracking", "phase2.5_tracking"],
    "transform": ["04_transform", "phase3_transform"],
    "aggregation": ["05_aggregation", "phase4_aggregation"],
    "visualization": ["06_visualization", "phase5_visualization"],
}


class SessionDataLoader:
    """セッション成果物を扱うローダ."""

    def __init__(self, sessions_root: str | Path, latest_symlink: str | Path | None = None):
        self.sessions_root = Path(sessions_root)
        self.latest_symlink = Path(latest_symlink) if latest_symlink else None

    def _resolve_session_dir(self, session_id: str) -> Path:
        if session_id == "latest" and self.latest_symlink and self.latest_symlink.exists():
            return self.latest_symlink.resolve()
        return self.sessions_root / session_id

    def get_available_sessions(self) -> list[str]:
        if not self.sessions_root.exists():
            return []
        sessions = [d.name for d in self.sessions_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        sessions = sorted(sessions, reverse=True)
        if self.latest_symlink and self.latest_symlink.exists():
            return ["latest", *sessions]
        return sessions

    def get_session_path(self, session_id: str) -> Path:
        return self._resolve_session_dir(session_id)

    def _find_phase_dir(self, session_dir: Path, candidates: Iterable[str]) -> Path | None:
        for candidate in candidates:
            phase_dir = session_dir / candidate
            if phase_dir.exists():
                return phase_dir
        return None

    def _load_json(self, path: Path) -> dict[str, Any] | list[Any] | None:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict | list):
                    return data
                return None
        except Exception as exc:
            logger.warning("Failed to load json %s: %s", path, exc)
            return None

    def _load_csv(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to load csv %s: %s", path, exc)
            return None

    def _list_files(self, path: Path, pattern: str) -> list[str]:
        if not path.exists():
            return []
        return sorted([p.name for p in path.glob(pattern)])

    @st.cache_data(show_spinner=False)
    def load_metadata(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        data = _self._load_json(session_dir / "metadata.json")
        return data if isinstance(data, dict) else {}

    @st.cache_data(show_spinner=False)
    def load_summary(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        data = _self._load_json(session_dir / "summary.json")
        return data if isinstance(data, dict) else {}

    @st.cache_data(show_spinner=False)
    def load_pipeline_checkpoint(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        data = _self._load_json(session_dir / "pipeline_checkpoint.json")
        return data if isinstance(data, dict) else {}

    @st.cache_data(show_spinner=False)
    def load_config(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        config_path = session_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                    return cfg
            except Exception as exc:
                logger.warning("Failed to load config from %s: %s", config_path, exc)
        metadata = _self.load_metadata(session_id)
        if metadata:
            config_data = metadata.get("config", {})
            if isinstance(config_data, dict):
                return config_data
        return {}

    @st.cache_data(show_spinner=False)
    def load_extraction(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        phase_dir = _self._find_phase_dir(session_dir, PHASE_DIR_CANDIDATES["extraction"])
        frames: list[str] = []
        results: pd.DataFrame | None = None
        if phase_dir:
            results = _self._load_csv(phase_dir / "extraction_results.csv")
            frames_dir = phase_dir / "frames"
            frames = _self._list_files(frames_dir, "*.jpg")
        return {"frames": frames, "results": results, "phase_dir": phase_dir}

    @st.cache_data(show_spinner=False)
    def load_detection(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        phase_dir = _self._find_phase_dir(session_dir, PHASE_DIR_CANDIDATES["detection"])
        statistics: dict[str, Any] = {}
        results: dict[str, Any] | list[Any] | None = None
        images: list[str] = []
        if phase_dir:
            stats_raw = _self._load_json(phase_dir / "detection_statistics.json")
            if isinstance(stats_raw, dict):
                statistics = stats_raw
            results_raw = _self._load_json(phase_dir / "detection_results.json")
            if isinstance(results_raw, dict | list):
                results = results_raw
            images_dir = phase_dir / "images"
            images = _self._list_files(images_dir, "*.jpg")
        return {"statistics": statistics, "results": results, "images": images, "phase_dir": phase_dir}

    @st.cache_data(show_spinner=False)
    def load_tracking(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        phase_dir = _self._find_phase_dir(session_dir, PHASE_DIR_CANDIDATES["tracking"])
        statistics: dict[str, Any] = {}
        tracks_json: dict[str, Any] | list[Any] | None = None
        tracks_csv: pd.DataFrame | None = None
        images: list[str] = []
        if phase_dir:
            stats_raw = _self._load_json(phase_dir / "tracking_statistics.json")
            if isinstance(stats_raw, dict):
                statistics = stats_raw
            tracks_json_raw = _self._load_json(phase_dir / "tracks.json")
            if isinstance(tracks_json_raw, dict | list):
                tracks_json = tracks_json_raw
            tracks_csv = _self._load_csv(phase_dir / "tracks.csv")
            images_dir = phase_dir / "images"
            images = _self._list_files(images_dir, "*.jpg")
        else:
            # fallback to root tracks.json
            tracks_json_raw = _self._load_json(session_dir / "tracks.json")
            if isinstance(tracks_json_raw, dict | list):
                tracks_json = tracks_json_raw
        return {
            "statistics": statistics,
            "tracks_json": tracks_json,
            "tracks_csv": tracks_csv,
            "images": images,
            "phase_dir": phase_dir,
        }

    @st.cache_data(show_spinner=False)
    def load_transform(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        phase_dir = _self._find_phase_dir(session_dir, PHASE_DIR_CANDIDATES["transform"])
        transformations: dict[str, Any] | list[Any] | None = None
        if phase_dir:
            transformations = _self._load_json(phase_dir / "coordinate_transformations.json")
        else:
            transformations = _self._load_json(session_dir / "coordinate_transformations.json")
        return {"transformations": transformations, "phase_dir": phase_dir}

    @st.cache_data(show_spinner=False)
    def load_aggregation(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        phase_dir = _self._find_phase_dir(session_dir, PHASE_DIR_CANDIDATES["aggregation"])
        zone_counts: pd.DataFrame | None = None
        if phase_dir:
            zone_counts = _self._load_csv(phase_dir / "zone_counts.csv")
        else:
            zone_counts = _self._load_csv(session_dir / "zone_counts.csv")
        return {"zone_counts": zone_counts, "phase_dir": phase_dir}

    @st.cache_data(show_spinner=False)
    def load_visualization(_self, session_id: str) -> dict[str, Any]:
        session_dir = _self._resolve_session_dir(session_id)
        phase_dir = _self._find_phase_dir(session_dir, PHASE_DIR_CANDIDATES["visualization"])
        graphs: list[str] = []
        floormaps: list[str] = []
        videos: list[str] = []
        if phase_dir:
            graphs_dir = phase_dir / "graphs"
            graphs = _self._list_files(graphs_dir, "*.png")
            floormap_dir = phase_dir / "floormaps"
            if floormap_dir.exists():
                floormaps = sorted([str(p.relative_to(floormap_dir)) for p in floormap_dir.rglob("*.png")])
            videos = _self._list_files(phase_dir, "*.mp4")
        return {"graphs": graphs, "floormaps": floormaps, "videos": videos, "phase_dir": phase_dir}


def render_kpi_card(
    title: str, value: Any, delta: str | None = None, help_text: str | None = None, icon: str | None = None
):
    st.metric(label=f"{icon + ' ' if icon else ''}{title}", value=value, delta=delta, help=help_text)


def get_phase_status_icon(status: str) -> str:
    if status == "completed":
        return "✅"
    if status == "failed":
        return "❌"
    if status == "skipped":
        return "⏭️"
    if status == "in_progress":
        return "⏳"
    return "❓"
