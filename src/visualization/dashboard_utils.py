"""Dashboard utilities for the interactive visualizer.

This module provides data loading and UI utility functions for the Streamlit dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

logger = logging.getLogger(__name__)


class SessionDataLoader:
    """Data loader for session data with caching."""

    def __init__(self, sessions_root: str | Path):
        """Initialize the data loader.

        Args:
            sessions_root: Root directory containing session outputs
        """
        self.sessions_root = Path(sessions_root)

    def get_available_sessions(self) -> list[str]:
        """Get list of available session IDs sorted by date (newest first)."""
        if not self.sessions_root.exists():
            return []

        sessions = [d.name for d in self.sessions_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        return sorted(sessions, reverse=True)

    def get_session_path(self, session_id: str) -> Path:
        """Get full path for a session ID."""
        return self.sessions_root / session_id

    @st.cache_data
    def load_config(_self, session_id: str) -> dict[str, Any]:
        """Load config.yaml for a session."""
        # Note: config.yaml might not be copied to session dir in current implementation.
        # If not found in session, try to load from metadata or root config as fallback.
        session_dir = _self.get_session_path(session_id)
        config_path = session_dir / "config.yaml"

        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Fallback: Try to extract from metadata
        metadata = _self.load_metadata(session_id)
        if metadata and "config" in metadata:
            return metadata["config"]

        return {}

    @st.cache_data
    def load_metadata(_self, session_id: str) -> dict[str, Any]:
        """Load metadata.json for a session."""
        session_dir = _self.get_session_path(session_id)
        metadata_path = session_dir / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        return {}

    @st.cache_data
    def load_summary(_self, session_id: str) -> dict[str, Any]:
        """Load summary.json for a session."""
        session_dir = _self.get_session_path(session_id)
        summary_path = session_dir / "summary.json"

        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load summary from {summary_path}: {e}")
        return {}

    @st.cache_data
    def load_baseline_metrics(_self, session_id: str) -> dict[str, Any]:
        """Load baseline_metrics.json if available."""
        session_dir = _self.get_session_path(session_id)
        metrics_path = session_dir / "baseline_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load baseline metrics from {metrics_path}: {e}")
        return {}

    @st.cache_data
    def load_mot_metrics(_self, session_id: str) -> dict[str, Any]:
        """Load mot_metrics.json if available."""
        session_dir = _self.get_session_path(session_id)
        metrics_path = session_dir / "mot_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load mot metrics from {metrics_path}: {e}")
        return {}

    @st.cache_data
    def load_reprojection_error(_self, session_id: str) -> dict[str, Any]:
        """Load reprojection_error.json if available."""
        session_dir = _self.get_session_path(session_id)
        metrics_path = session_dir / "reprojection_error.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load reprojection error from {metrics_path}: {e}")
        return {}

    @st.cache_data
    def load_performance_metrics(_self, session_id: str) -> dict[str, Any]:
        """Load performance_metrics.json if available."""
        session_dir = _self.get_session_path(session_id)
        metrics_path = session_dir / "performance_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load performance metrics from {metrics_path}: {e}")
        return {}

    @st.cache_data
    def load_phase1_data(_self, session_id: str) -> dict[str, Any]:
        """Load Phase 1 (Extraction) data."""
        session_dir = _self.get_session_path(session_id)
        phase_dir = session_dir / "phase1_extraction"

        data = {"results": None, "frames": []}

        if not phase_dir.exists():
            return data

        # Load CSV results
        csv_path = phase_dir / "extraction_results.csv"
        if csv_path.exists():
            try:
                data["results"] = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning(f"Failed to load phase1 csv: {e}")

        # List frames
        frames_dir = phase_dir / "frames"
        if frames_dir.exists():
            data["frames"] = sorted([f.name for f in frames_dir.glob("*.jpg")])

        return data

    @st.cache_data
    def load_phase2_data(_self, session_id: str) -> dict[str, Any]:
        """Load Phase 2 (Detection) data."""
        session_dir = _self.get_session_path(session_id)
        phase_dir = session_dir / "phase2_detection"

        data = {"statistics": {}, "results": None, "images": []}

        if not phase_dir.exists():
            return data

        # Load statistics
        stats_path = phase_dir / "detection_statistics.json"
        if stats_path.exists():
            try:
                with open(stats_path, encoding="utf-8") as f:
                    data["statistics"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load phase2 stats: {e}")

        # Load results (if json exists)
        results_path = phase_dir / "detection_results.json"
        if results_path.exists():
            try:
                with open(results_path, encoding="utf-8") as f:
                    data["results"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load phase2 results: {e}")

        # List images
        images_dir = phase_dir / "images"
        if images_dir.exists():
            data["images"] = sorted([f.name for f in images_dir.glob("*.jpg")])

        return data

    @st.cache_data
    def load_phase2_5_data(_self, session_id: str) -> dict[str, Any]:
        """Load Phase 2.5 (Tracking) data."""
        session_dir = _self.get_session_path(session_id)
        phase_dir = session_dir / "phase2.5_tracking"

        data = {"tracks": [], "statistics": {}, "images": []}

        if not phase_dir.exists():
            # Fallback for older structure or if phase dir is missing but tracks.json exists in root
            tracks_path = session_dir / "tracks.json"
            if tracks_path.exists():
                try:
                    with open(tracks_path, encoding="utf-8") as f:
                        json_data = json.load(f)
                        data["tracks"] = json_data.get("tracks", [])
                except Exception as e:
                    logger.warning(f"Failed to load tracks.json: {e}")
            return data

        # Load tracks
        tracks_path = phase_dir / "tracks.json"
        if tracks_path.exists():
            try:
                with open(tracks_path, encoding="utf-8") as f:
                    json_data = json.load(f)
                    data["tracks"] = json_data.get("tracks", [])
            except Exception as e:
                logger.warning(f"Failed to load phase2.5 tracks: {e}")

        # Load statistics
        stats_path = phase_dir / "tracking_statistics.json"
        if stats_path.exists():
            try:
                with open(stats_path, encoding="utf-8") as f:
                    data["statistics"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load phase2.5 stats: {e}")

        # List images
        images_dir = phase_dir / "images"
        if images_dir.exists():
            data["images"] = sorted([f.name for f in images_dir.glob("*.jpg")])

        return data

    @st.cache_data
    def load_phase3_data(_self, session_id: str) -> dict[str, Any]:
        """Load Phase 3 (Transform) data."""
        session_dir = _self.get_session_path(session_id)
        phase_dir = session_dir / "phase3_transform"

        data = {"transformations": []}

        if not phase_dir.exists():
            return data

        trans_path = phase_dir / "coordinate_transformations.json"
        if trans_path.exists():
            try:
                with open(trans_path, encoding="utf-8") as f:
                    json_data = json.load(f)
                    data["transformations"] = json_data.get("transformations", [])
            except Exception as e:
                logger.warning(f"Failed to load phase3 transformations: {e}")

        return data

    @st.cache_data
    def load_phase4_data(_self, session_id: str) -> dict[str, Any]:
        """Load Phase 4 (Aggregation) data."""
        session_dir = _self.get_session_path(session_id)
        phase_dir = session_dir / "phase4_aggregation"

        data = {"zone_counts": None}

        if not phase_dir.exists():
            # Fallback
            csv_path = session_dir / "zone_counts.csv"
            if csv_path.exists():
                try:
                    data["zone_counts"] = pd.read_csv(csv_path)
                except Exception as e:
                    logger.warning(f"Failed to load zone_counts.csv: {e}")
            return data

        csv_path = phase_dir / "zone_counts.csv"
        if csv_path.exists():
            try:
                data["zone_counts"] = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning(f"Failed to load phase4 csv: {e}")

        return data

    @st.cache_data
    def load_phase5_data(_self, session_id: str) -> dict[str, Any]:
        """Load Phase 5 (Visualization) data."""
        session_dir = _self.get_session_path(session_id)
        phase_dir = session_dir / "phase5_visualization"

        data = {"graphs": [], "floormaps": [], "videos": []}

        if not phase_dir.exists():
            return data

        # Graphs
        graphs_dir = phase_dir / "graphs"
        if graphs_dir.exists():
            data["graphs"] = sorted([f.name for f in graphs_dir.glob("*.png")])

        # Floormaps
        floormaps_dir = phase_dir / "floormaps"
        if floormaps_dir.exists():
            # Recursive search for floormaps
            data["floormaps"] = sorted([str(f.relative_to(floormaps_dir)) for f in floormaps_dir.rglob("*.png")])

        # Videos
        videos = list(phase_dir.glob("*.mp4"))
        data["videos"] = sorted([f.name for f in videos])

        return data


def render_kpi_card(
    title: str, value: Any, delta: str | None = None, help_text: str | None = None, icon: str | None = None
):
    """Render a KPI card with consistent styling."""
    st.metric(label=f"{icon + ' ' if icon else ''}{title}", value=value, delta=delta, help=help_text)


def get_phase_status_icon(status: str) -> str:
    """Get icon for phase status."""
    if status == "completed":
        return "✅"
    if status == "failed":
        return "❌"
    if status == "skipped":
        return "⏭️"
    return "❓"
