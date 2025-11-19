"""Streamlit-based interactive visualization dashboard for the project.

This dashboard provides a comprehensive view of the entire pipeline, from frame extraction
to aggregation and visualization.
"""

import contextlib
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import setup_logging
from src.visualization.dashboard_utils import SessionDataLoader, get_phase_status_icon, render_kpi_card

logger = logging.getLogger(__name__)


class DashboardApp:
    """Main dashboard application class."""

    def __init__(self):
        """Initialize the dashboard application."""
        self.sessions_root = Path("output/sessions")
        self.data_loader = SessionDataLoader(self.sessions_root)

        # Initialize session state
        if "current_session_id" not in st.session_state:
            sessions = self.data_loader.get_available_sessions()
            if sessions:
                st.session_state.current_session_id = sessions[0]
            else:
                st.session_state.current_session_id = None

        if "page" not in st.session_state:
            st.session_state.page = "Overview"

    def run(self):
        """Run the dashboard application."""
        self._setup_page()
        self._render_sidebar()
        self._render_main_content()

    def _setup_page(self):
        """Setup page configuration and styling."""
        st.set_page_config(
            page_title="Office Person Detection Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS (dark-mode optimized)
        st.markdown(
            """
        <style>
        .main > div { padding-top: 1rem; }

        /* KPI Card Styling */
        div[data-testid="stMetric"] {
            background-color: #1e1e1e;
            border: 1px solid #333333;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.6);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"] label {
            color: #cccccc !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #ffffff !important;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.8);
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #2a2a2a;
            color: #e0e0e0;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            border: 1px solid #3a3a3a;
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e1e1e;
            color: #66b2ff;
            border-top: 3px solid #66b2ff;
            font-weight: bold;
        }

        /* Headers */
        h1 {
            color: #66b2ff !important;
            font-weight: 700;
        }
        h2 {
            color: #e0e0e0 !important;
            border-bottom: 2px solid #333333;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        h3 {
            color: #cccccc !important;
            margin-top: 20px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            color: #e0e0e0 !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #66b2ff !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def _render_sidebar(self):
        """Render the sidebar navigation."""
        with st.sidebar:
            st.title("📊 Dashboard")

            # Session Selector
            st.header("Session")
            sessions = self.data_loader.get_available_sessions()
            if sessions:
                selected_session = st.selectbox(
                    "Select Session",
                    sessions,
                    index=sessions.index(st.session_state.current_session_id)
                    if st.session_state.current_session_id in sessions
                    else 0,
                    key="session_selector",
                )
                if selected_session != st.session_state.current_session_id:
                    st.session_state.current_session_id = selected_session
                    st.rerun()
            else:
                st.warning("No sessions found.")

            st.divider()

            # Navigation
            st.header("Navigation")
            page = st.radio(
                "Go to",
                ["Overview", "Pipeline Inspector", "Comparison & Evaluation"],
                index=["Overview", "Pipeline Inspector", "Comparison & Evaluation"].index(st.session_state.page),
                key="nav_radio",
            )
            st.session_state.page = page

            st.divider()

            # About
            st.info(
                "**Office Person Detection System**\n\n"
                "Integrated dashboard for monitoring and analyzing the person detection pipeline."
            )

    def _render_main_content(self):
        """Render the main content based on selected page."""
        session_id = st.session_state.current_session_id
        if not session_id:
            st.error("Please select a session to view.")
            return

        if st.session_state.page == "Overview":
            self._render_overview_page(session_id)
        elif st.session_state.page == "Pipeline Inspector":
            self._render_pipeline_page(session_id)
        elif st.session_state.page == "Comparison & Evaluation":
            self._render_comparison_page(session_id)

    def _render_overview_page(self, session_id: str):
        """Render the project overview page."""
        st.title(f"🏠 Project Overview: {session_id}")

        # Load data
        metadata = self.data_loader.load_metadata(session_id)
        summary = self.data_loader.load_summary(session_id)
        config = self.data_loader.load_config(session_id)

        # 1. System Status & KPI
        st.subheader("📈 System Status & KPIs")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = metadata.get("status", "unknown") if isinstance(metadata, dict) else "unknown"
            icon = get_phase_status_icon(status)
            render_kpi_card("Status", status.upper(), icon=icon)

        with col2:
            # Total processing time
            perf = summary.get("performance", {}) if isinstance(summary, dict) else {}
            total_time = 0.0
            if isinstance(perf, dict):
                total_time = sum(
                    v.get("avg_time", 0.0) for v in perf.values() if isinstance(v, dict) and "avg_time" in v
                )
            render_kpi_card("Total Time", f"{total_time:.1f}s", icon="⏱️")

        with col3:
            # Total visitors (from tracking results)
            total_visitors = "N/A"
            phase4_data = self.data_loader.load_phase4_data(session_id)
            zone_counts = phase4_data.get("zone_counts") if isinstance(phase4_data, dict) else None
            if zone_counts is not None:
                phase2_5_data = self.data_loader.load_phase2_5_data(session_id)
                tracks = phase2_5_data.get("tracks") if isinstance(phase2_5_data, dict) else None
                if tracks:
                    total_visitors = len(tracks)

            render_kpi_card("Total Visitors", total_visitors, icon="👥")

        with col4:
            # Frame count
            phase1_data = self.data_loader.load_phase1_data(session_id)
            frames = phase1_data.get("frames") if isinstance(phase1_data, dict) else None
            frame_count = len(frames) if frames is not None else 0
            render_kpi_card("Processed Frames", frame_count, icon="🖼️")

        st.divider()

        # 2. Pipeline Configuration
        st.subheader("⚙️ Pipeline Configuration")

        with st.expander("View Full Configuration", expanded=False):
            st.json(config)

        # Visual Pipeline Flow (Simple representation)
        st.write("#### Pipeline Flow")

        phases = [
            ("Phase 1: Extraction", "phase1_extraction"),
            ("Phase 2: Detection", "phase2_detection"),
            ("Phase 2.5: Tracking", "phase2.5_tracking"),
            ("Phase 3: Transform", "phase3_transform"),
            ("Phase 4: Aggregation", "phase4_aggregation"),
            ("Phase 5: Visualization", "phase5_visualization"),
        ]

        cols = st.columns(len(phases))
        for i, (name, key) in enumerate(phases):
            with cols[i]:
                phase_dir = self.data_loader.get_session_path(session_id) / key
                phase_status = "✅" if phase_dir.exists() else "⬜"
                st.info(f"{phase_status} **{name}**")

        st.divider()
        self._render_evaluation_section(session_id)

    def _render_evaluation_section(self, session_id: str):
        """Render evaluation/baseline metrics."""
        st.subheader("🧪 Evaluation & Baseline Metrics")

        eval_data = self._collect_evaluation_data(session_id)
        baseline = eval_data["baseline"]
        mot = eval_data["mot"]
        reproj = eval_data["reprojection"]
        performance = eval_data["performance"]

        if not baseline and not mot and not reproj and not performance:
            st.info(
                "No `baseline_metrics.json` found. Run `scripts/evaluate_baseline.py` "
                "to generate MOT / 再投影誤差 / パフォーマンスレポート."
            )
            self._render_evaluation_actions(session_id)
            return

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_kpi_card(
                "MOTA",
                f"{mot.get('MOTA', 0.0):.3f}" if isinstance(mot.get("MOTA"), int | float) else "N/A",
                icon="🎯",
                help_text="目標 0.7 以上",
            )
        with col2:
            render_kpi_card(
                "IDF1",
                f"{mot.get('IDF1', 0.0):.3f}" if isinstance(mot.get("IDF1"), int | float) else "N/A",
                icon="🆔",
                help_text="目標 0.8 以上",
            )
        with col3:
            render_kpi_card(
                "Mean Reproj (px)",
                f"{reproj.get('mean_error', 0.0):.2f}" if isinstance(reproj.get("mean_error"), int | float) else "N/A",
                icon="🧭",
                help_text="目標 ≤ 2px",
            )
        with col4:
            render_kpi_card(
                "Max Reproj (px)",
                f"{reproj.get('max_error', 0.0):.2f}" if isinstance(reproj.get("max_error"), int | float) else "N/A",
                icon="📐",
                help_text="目標 ≤ 4px",
            )
        with col5:
            time_per_frame = eval_data["time_per_frame"]
            render_kpi_card(
                "Time / Frame",
                f"{time_per_frame:.2f}s" if time_per_frame is not None else "N/A",
                icon="⏱️",
                help_text="目標 ≤ 2.0s",
            )

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Baseline Metrics JSON**")
            if baseline:
                st.json(baseline)
            else:
                st.info("baseline_metrics.json is not available.")
        with col2:
            st.write("**Performance Breakdown**")
            perf_df = None
            phase_times = (performance or {}).get("phase_times", {})
            if isinstance(phase_times, dict) and phase_times:
                perf_df = pd.DataFrame(
                    [
                        {"Phase": phase, "Seconds": value}
                        for phase, value in phase_times.items()
                        if isinstance(value, int | float)
                    ]
                )
            if perf_df is not None and not perf_df.empty:
                perf_df = perf_df.sort_values("Seconds", ascending=False)
                st.dataframe(perf_df, use_container_width=True)
            else:
                st.info("No performance metrics recorded.")

        self._render_evaluation_actions(session_id)

    def _render_evaluation_actions(self, session_id: str):
        """Render evaluation-related actions (manual/auto triggers)."""
        with st.expander("Evaluation Actions", expanded=False):
            st.markdown(
                "- `scripts/evaluate_baseline.py` を実行すると、MOT / 再投影誤差 / パフォーマンスが再計算されます。\n"
                "- 設定やセッションを変更した場合は再実行してください。"
            )
            if st.button("Run evaluate_baseline.py", key=f"run_eval_{session_id}"):
                self._run_evaluate_baseline(session_id)

    def _collect_evaluation_data(self, session_id: str) -> dict[str, Any]:
        """Collect baseline/mot/reprojection/performance metrics for a session."""
        baseline = self.data_loader.load_baseline_metrics(session_id)
        mot_metrics = baseline.get("mot_metrics") if isinstance(baseline, dict) else None
        if not mot_metrics:
            mot_json = self.data_loader.load_mot_metrics(session_id)
            mot_metrics = mot_json.get("metrics") if isinstance(mot_json, dict) else mot_json

        reprojection = baseline.get("reprojection_error") if isinstance(baseline, dict) else None
        if not reprojection:
            reprojection = self.data_loader.load_reprojection_error(session_id).get("error_metrics", {})

        performance = baseline.get("performance") if isinstance(baseline, dict) else None
        if not performance:
            performance = self.data_loader.load_performance_metrics(session_id)

        time_per_frame = None
        if isinstance(performance, dict):
            time_per_frame = performance.get("time_per_frame_seconds")
            if time_per_frame is None:
                total = performance.get("phase_times", {}).get("total")
                num_frames = performance.get("num_frames")
                if isinstance(total, int | float) and isinstance(num_frames, int | float) and num_frames > 0:
                    time_per_frame = total / num_frames

        return {
            "baseline": baseline,
            "mot": mot_metrics or {},
            "reprojection": reprojection or {},
            "performance": performance or {},
            "time_per_frame": time_per_frame,
        }

    def _run_evaluate_baseline(self, session_id: str):
        """Execute evaluate_baseline.py for the selected session."""
        st.info("Running `scripts/evaluate_baseline.py`... This may take a minute.")

        cmd = [
            sys.executable,
            "scripts/evaluate_baseline.py",
            "--session",
            session_id,
            "--config",
            "config.yaml",
        ]

        gt_path = Path("data/gt_tracks_auto.json")
        if gt_path.exists():
            cmd += ["--gt", str(gt_path)]

        points_path = Path("output/calibration/correspondence_points_cam01.json")
        if points_path.exists():
            cmd += ["--points", str(points_path)]

        try:
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                st.success("evaluate_baseline.py completed successfully.")
            else:
                st.error(f"evaluate_baseline.py failed with exit code {result.returncode}")
            if result.stdout:
                st.code(result.stdout, language="text")
            if result.stderr:
                st.code(result.stderr, language="text")
        except Exception as exc:
            st.error(f"Failed to run evaluate_baseline.py: {exc}")
        finally:
            self._clear_evaluation_cache()
            st.rerun()

    def _clear_evaluation_cache(self):
        """Clear cached evaluation files so that Streamlit reloads them after re-run."""
        for loader in [
            self.data_loader.load_baseline_metrics,
            self.data_loader.load_mot_metrics,
            self.data_loader.load_reprojection_error,
            self.data_loader.load_performance_metrics,
        ]:
            with contextlib.suppress(AttributeError):
                # Older Streamlit versions may not expose .clear()
                loader.clear()

    def _render_pipeline_page(self, session_id: str):
        """Render the pipeline inspector page."""
        st.title(f"🔍 Pipeline Inspector: {session_id}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "1. Input & Extraction",
                "2. Detection Analysis",
                "3. Tracking & Trajectory",
                "4. Space & Zones",
                "5. Aggregation & Report",
            ]
        )

        with tab1:
            self._render_phase1_tab(session_id)

        with tab2:
            self._render_phase2_tab(session_id)

        with tab3:
            self._render_phase2_5_tab(session_id)

        with tab4:
            self._render_phase3_4_tab(session_id)

        with tab5:
            self._render_phase5_tab(session_id)

    def _render_phase1_tab(self, session_id: str):
        """Render Phase 1: Input & Extraction tab."""
        st.header("Phase 1: Input & Extraction")

        # Load data
        config = self.data_loader.load_config(session_id)
        phase1_data = self.data_loader.load_phase1_data(session_id)

        # 1. Input Configuration
        st.subheader("🎥 Input Configuration")
        input_config = config.get("input", {})
        sampling_config = config.get("sampling", {})

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Video Source**")
            st.code(input_config.get("video_path", "N/A"))

        with col2:
            st.write("**Sampling Strategy**")
            st.write(f"- Interval: {sampling_config.get('interval_minutes', 5)} min")
            st.write(f"- Tolerance: ±{sampling_config.get('tolerance_seconds', 10)} sec")

        st.divider()

        # 2. Extraction Results
        st.subheader("📸 Extraction Results")

        results_df = phase1_data.get("results") if isinstance(phase1_data, dict) else None
        if results_df is not None:
            df = results_df

            # Stats
            total_frames = len(df)
            success_count = df["success"].sum() if "success" in df.columns else total_frames

            col1, col2, col3 = st.columns(3)
            with col1:
                render_kpi_card("Extracted Frames", total_frames, icon="🎞️")
            with col2:
                success_rate = (success_count / total_frames * 100) if total_frames else 0.0
                render_kpi_card("Success Rate", f"{success_rate:.1f}%", icon="✅")
            with col3:
                if "confidence" in df.columns:
                    avg_conf = df["confidence"].mean()
                    render_kpi_card("Avg OCR Confidence", f"{avg_conf:.2f}", icon="🔍")

            with st.expander("📊 Extraction Details Table", expanded=False):
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("No extraction results (CSV) found.")

        # 3. Frame Gallery
        st.subheader("🖼️ Extracted Frames Gallery")

        frames = phase1_data.get("frames") if isinstance(phase1_data, dict) else None
        if frames:
            frame_idx = st.slider(
                "Select Frame",
                0,
                len(frames) - 1,
                0,
                key="p1_frame_slider",
            )
            selected_frame = frames[frame_idx]

            session_dir = self.data_loader.get_session_path(session_id)
            img_path = session_dir / "phase1_extraction" / "frames" / selected_frame

            if img_path.exists():
                st.image(
                    str(img_path),
                    caption=selected_frame,
                    use_container_width=True,
                )

                if results_df is not None:
                    df = results_df
                    if frame_idx < len(df):
                        st.write("**Frame Metadata**")
                        st.dataframe(df.iloc[[frame_idx]], use_container_width=True)
        else:
            st.info("No extracted frames found.")

    def _render_phase2_tab(self, session_id: str):
        """Render Phase 2: Detection Analysis tab."""
        st.header("Phase 2: Detection Analysis")

        # Load data
        config = self.data_loader.load_config(session_id)
        phase2_data = self.data_loader.load_phase2_data(session_id)

        # 1. Model Configuration
        st.subheader("🤖 Model Configuration")
        det_config = config.get("detection", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Model**")
            st.code(det_config.get("model_name", "N/A"))
        with col2:
            st.write("**Device**")
            st.code(det_config.get("device", "auto"))
        with col3:
            st.write("**Thresholds**")
            st.write(f"- Confidence: {det_config.get('confidence_threshold', 0.5)}")

        st.divider()

        # 2. Detection Statistics
        st.subheader("📊 Detection Statistics")

        stats = phase2_data.get("statistics") if isinstance(phase2_data, dict) else {}
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                render_kpi_card(
                    "Total Detections",
                    stats.get("total_detections", 0),
                    icon="🎯",
                )
            with col2:
                render_kpi_card(
                    "Avg Detections/Frame",
                    f"{stats.get('avg_detections_per_frame', 0.0):.1f}",
                    icon="🔢",
                )
            with col3:
                render_kpi_card(
                    "Avg Confidence",
                    f"{stats.get('avg_confidence', 0.0):.2f}",
                    icon="💪",
                )

            dist = stats.get("confidence_distribution")
            if dist is not None:
                st.write("**Confidence Distribution**")
                if isinstance(dist, dict):
                    chart_data = pd.DataFrame(list(dist.items()), columns=["Range", "Count"])
                    st.bar_chart(chart_data.set_index("Range"))
                elif isinstance(dist, list):
                    # If raw values are stored (unlikely for large datasets)
                    pass
        else:
            st.info("No detection statistics available.")

        # 3. Detection Gallery
        st.subheader("🖼️ Detection Results Gallery")

        images = phase2_data.get("images") if isinstance(phase2_data, dict) else None
        if images:
            img_idx = st.slider(
                "Select Image",
                0,
                len(images) - 1,
                0,
                key="p2_img_slider",
            )
            selected_img = images[img_idx]

            session_dir = self.data_loader.get_session_path(session_id)
            img_path = session_dir / "phase2_detection" / "images" / selected_img

            if img_path.exists():
                st.image(
                    str(img_path),
                    caption=selected_img,
                    use_container_width=True,
                )
            else:
                st.info("No detection images found.")
        else:
            st.info("No detection images found.")

    def _render_phase2_5_tab(self, session_id: str):
        """Render Phase 2.5: Tracking & Trajectory tab."""
        st.header("Phase 2.5: Tracking & Trajectory")

        # Load data
        config = self.data_loader.load_config(session_id)
        phase2_5_data = self.data_loader.load_phase2_5_data(session_id)
        tracks_data = phase2_5_data.get("tracks") if isinstance(phase2_5_data, dict) else []

        # 1. Tracking Configuration
        st.subheader("👣 Tracking Configuration")
        track_config = config.get("tracking", {})

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Algorithm**")
            st.code("DeepSORT (Kalman Filter + Hungarian)")
        with col2:
            st.write("**Parameters**")
            st.write(f"- Max Age: {track_config.get('max_age', 30)}")
            st.write(f"- Min Hits: {track_config.get('min_hits', 3)}")
            st.write(f"- IOU Threshold: {track_config.get('iou_threshold', 0.3)}")

        st.divider()

        # 2. Tracking Statistics
        st.subheader("📊 Tracking Statistics")

        if tracks_data:
            total_tracks = len(tracks_data)
            total_points = sum(len(t.get("trajectory", [])) for t in tracks_data)
            avg_len = total_points / total_tracks if total_tracks > 0 else 0.0

            short_tracks = sum(1 for t in tracks_data if len(t.get("trajectory", [])) < 5)
            est_switches = int(short_tracks * 0.5)

            stats = phase2_5_data.get("statistics") if isinstance(phase2_5_data, dict) else {}

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_kpi_card("Total Tracks", total_tracks, icon="👣")
            with col2:
                render_kpi_card("Avg Trajectory Length", f"{avg_len:.1f}", icon="📏")
            with col3:
                render_kpi_card(
                    "Est. ID Switches",
                    est_switches,
                    icon="🔄",
                    help_text="Estimated based on short tracks",
                )
            with col4:
                max_concurrent = stats.get("max_concurrent", "N/A")
                render_kpi_card(
                    "Max Concurrent",
                    max_concurrent,
                    icon="👨‍👩‍👧‍👦",
                )
        else:
            st.warning("No tracking data found.")
            return

        st.divider()

        # 3. Interactive Visualization
        st.subheader("🗺️ Interactive Trajectory Visualization")

        col1, col2 = st.columns([3, 1])

        with col2:
            st.write("**Filters**")
            show_trajectories = st.checkbox("Show Trajectories", value=True)
            show_ids = st.checkbox("Show IDs", value=True)

            all_ids = sorted({t.get("track_id", 0) for t in tracks_data})
            selected_ids = st.multiselect("Filter by ID", all_ids, default=[])

            max_traj_len = st.slider("Max Trajectory Tail", 10, 100, 50)

        with col1:
            max_frames = max(
                (len(t.get("trajectory", [])) for t in tracks_data),
                default=0,
            )
            if max_frames > 0:
                frame_idx = st.slider(
                    "Frame Index",
                    0,
                    max_frames - 1,
                    0,
                    key="p2_5_frame_slider",
                )

                floormap = self._load_floormap(config)
                if floormap is not None:
                    vis_image = self._draw_tracks(
                        floormap.copy(),
                        tracks_data,
                        frame_idx,
                        show_trajectories,
                        show_ids,
                        selected_ids,
                        max_traj_len,
                    )
                    st.image(
                        vis_image,
                        use_container_width=True,
                        channels="BGR",
                    )

                    current_count = sum(1 for t in tracks_data if frame_idx < len(t.get("trajectory", [])))
                    st.caption(f"Frame: {frame_idx} | Active Tracks: {current_count}")
                else:
                    st.error("Failed to load floormap image.")
            else:
                st.info("No trajectory data to visualize.")

    def _load_floormap(self, config: dict[str, Any]) -> np.ndarray | None:
        """Load floormap image."""
        floormap_path = config.get("floormap", {}).get("image_path", "data/floormap.png")
        path = Path(floormap_path)
        if not path.exists():
            path = project_root / floormap_path

        if path.exists():
            return cv2.imread(str(path))

        return None

    def _draw_tracks(
        self,
        image: np.ndarray,
        tracks_data: list[dict[str, Any]],
        frame_idx: int,
        show_trajectories: bool,
        show_ids: bool,
        selected_ids: list[int],
        max_length: int,
    ) -> np.ndarray:
        """Draw tracks on the image."""
        for track in tracks_data:
            track_id = track.get("track_id", 0)
            trajectory = track.get("trajectory", [])

            # Filter
            if selected_ids and track_id not in selected_ids:
                continue

            if not trajectory:
                continue

            # Color
            hue = (track_id * 137) % 180
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(int(c) for c in color_bgr)

            # Draw trajectory
            if show_trajectories:
                traj_to_draw = trajectory[: min(frame_idx + 1, len(trajectory))]
                if max_length > 0:
                    traj_to_draw = traj_to_draw[-max_length:]

                for i in range(len(traj_to_draw) - 1):
                    pt1 = traj_to_draw[i]
                    pt2 = traj_to_draw[i + 1]
                    x1, y1 = int(pt1.get("x", 0)), int(pt1.get("y", 0))
                    x2, y2 = int(pt2.get("x", 0)), int(pt2.get("y", 0))
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)

            # Draw current position
            if frame_idx < len(trajectory):
                pt = trajectory[frame_idx]
                x, y = int(pt.get("x", 0)), int(pt.get("y", 0))
                cv2.circle(image, (x, y), 5, color, -1)

                if show_ids:
                    cv2.putText(
                        image, f"ID:{track_id}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                    )
        return image

    def _render_phase3_4_tab(self, session_id: str):
        """Render Phase 3/4: Space & Zones tab."""
        st.header("Phase 3/4: Space & Zones")

        # Load data
        config = self.data_loader.load_config(session_id)

        # 1. Homography Configuration
        st.subheader("📐 Homography Configuration")
        homo_config = config.get("homography", {})

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matrix Source**")
            st.code(homo_config.get("matrix_path", "N/A"))
        with col2:
            st.write("**Origin Offset**")
            offset = homo_config.get("origin_offset", {})
            st.write(f"X: {offset.get('x', 0)}, Y: {offset.get('y', 0)}")

        st.divider()

        # 2. Zone Configuration & Visualization
        st.subheader("📍 Zone Configuration")
        zones = config.get("zones", [])

        if zones:
            st.write(f"**Defined Zones**: {len(zones)}")

            # Zone Table
            zone_data = []
            for z in zones:
                zone_data.append({"ID": z.get("id"), "Name": z.get("name"), "Points": len(z.get("polygon", []))})
            st.dataframe(pd.DataFrame(zone_data), use_container_width=True)

            # Visualization
            st.write("**Zone Map**")
            floormap = self._load_floormap(config)
            if floormap is not None:
                vis_image = self._draw_zones(floormap.copy(), zones)
                st.image(vis_image, use_container_width=True, channels="BGR")
            else:
                st.error("Failed to load floormap image.")
        else:
            st.warning("No zones defined in configuration.")

        # 3. Transformation Accuracy (if available)
        # If we have reprojection error data, show it here

    def _draw_zones(self, image: np.ndarray, zones: list[dict[str, Any]]) -> np.ndarray:
        """Draw zones on the image."""
        overlay = image.copy()

        for zone in zones:
            polygon = zone.get("polygon", [])
            if not polygon:
                continue

            pts = np.array([[int(p[0]), int(p[1])] for p in polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Draw filled polygon with transparency
            color = (0, 255, 0)  # Green
            cv2.fillPoly(overlay, [pts], color)

            # Draw border
            cv2.polylines(image, [pts], True, color, 2)

            # Draw label
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(
                    image, zone.get("name", zone.get("id")), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
                cv2.putText(
                    image,
                    zone.get("name", zone.get("id")),
                    (cX - 20, cY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # Blend overlay
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        return image

    def _render_phase5_tab(self, session_id: str):
        """Render Phase 5: Aggregation & Report tab."""
        st.header("Phase 5: Aggregation & Report")

        # Load data
        phase4_data = self.data_loader.load_phase4_data(session_id)
        df = phase4_data["zone_counts"]

        if df is not None:
            # 1. Time Series Analysis
            st.subheader("📈 Time Series Analysis")

            # Convert timestamp to datetime if possible
            if "timestamp" in df.columns:
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                except Exception:
                    pass

            # Select zones to plot
            zone_cols = [c for c in df.columns if c not in ["timestamp", "frame_index", "total_count"]]
            selected_zones = st.multiselect("Select Zones", zone_cols, default=zone_cols)

            if selected_zones:
                st.line_chart(df[selected_zones])
            else:
                st.info("Select zones to view time series.")

            st.divider()

            # 2. Zone Statistics
            st.subheader("📊 Zone Statistics")

            # Calculate stats
            stats_data = []
            for zone in zone_cols:
                stats_data.append(
                    {
                        "Zone": zone,
                        "Average": df[zone].mean(),
                        "Max": df[zone].max(),
                        "Total Visits": df[zone].sum(),  # This is sum of counts per frame, not unique visits
                    }
                )

            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

            # 3. Download Data
            st.divider()
            st.subheader("💾 Download Data")

            csv = df.to_csv().encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"zone_counts_{session_id}.csv",
                mime="text/csv",
            )

        else:
            st.warning("No aggregation data (zone_counts.csv) found.")

    def _render_comparison_page(self, session_id: str):
        """Render the comparison and evaluation page."""
        st.title("📊 Comparison & Evaluation")

        # 1. Select Sessions
        st.subheader("🔍 Select Sessions to Compare")
        all_sessions = self.data_loader.get_available_sessions()

        # Default to current session + previous one if available
        default_sessions = [session_id]
        if len(all_sessions) > 1:
            prev_idx = all_sessions.index(session_id) + 1
            if prev_idx < len(all_sessions):
                default_sessions.append(all_sessions[prev_idx])

        selected_sessions = st.multiselect("Choose sessions", all_sessions, default=default_sessions)

        if not selected_sessions:
            st.warning("Please select at least one session.")
            return

        # 2. KPI Comparison
        st.subheader("📈 KPI Comparison")

        comp_data = []
        for sess in selected_sessions:
            summary = self.data_loader.load_summary(sess)
            phase2_data = self.data_loader.load_phase2_data(sess)
            phase2_5_data = self.data_loader.load_phase2_5_data(sess)
            eval_data = self._collect_evaluation_data(sess)

            perf = summary.get("performance", {})
            total_time = 0
            if isinstance(perf, dict):
                total_time = sum(v.get("avg_time", 0) for v in perf.values() if isinstance(v, dict) and "avg_time" in v)

            p2_stats = phase2_data.get("statistics", {})
            mot = eval_data["mot"]
            reproj = eval_data["reprojection"]
            perf_metrics = eval_data["performance"] or {}

            comp_data.append(
                {
                    "Session": sess,
                    "Total Time (s)": total_time,
                    "Total Detections": p2_stats.get("total_detections", 0),
                    "Total Tracks": len(phase2_5_data.get("tracks", [])),
                    "Avg Confidence": p2_stats.get("avg_confidence", 0),
                    "MOTA": mot.get("MOTA"),
                    "IDF1": mot.get("IDF1"),
                    "Mean Reproj (px)": reproj.get("mean_error"),
                    "Max Reproj (px)": reproj.get("max_error"),
                    "Time/Frame (s)": eval_data["time_per_frame"],
                    "Memory Δ (MB)": perf_metrics.get("memory_increase_mb"),
                }
            )

        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True)

        # Charts
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Processing Time Comparison**")
            st.bar_chart(df_comp.set_index("Session")["Total Time (s)"])
        with col2:
            st.write("**Detection/Track Count Comparison**")
            st.bar_chart(df_comp.set_index("Session")[["Total Detections", "Total Tracks"]])
        with col3:
            st.write("**Evaluation Metrics**")
            eval_cols = [c for c in ["MOTA", "IDF1", "Time/Frame (s)"] if c in df_comp.columns]
            if eval_cols:
                st.bar_chart(df_comp.set_index("Session")[eval_cols])
            else:
                st.info("No evaluation metrics available.")

        st.divider()

        # 3. Config Diff (if exactly 2 sessions selected)
        if len(selected_sessions) == 2:
            st.subheader("⚙️ Configuration Difference")

            sess1, sess2 = selected_sessions
            conf1 = self.data_loader.load_config(sess1)
            conf2 = self.data_loader.load_config(sess2)

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{sess1}**")
                st.json(conf1)
            with col2:
                st.write(f"**{sess2}**")
                st.json(conf2)

            # Ideally we would show a diff view here, but simple side-by-side is a good start.
            # We could flatten dicts and find diffs.

            diffs = self._find_config_diffs(conf1, conf2)
            if diffs:
                st.write("**Differences Detected**")
                st.dataframe(pd.DataFrame(diffs), use_container_width=True)
            else:
                st.success("No configuration differences found.")

    def _find_config_diffs(self, conf1: dict, conf2: dict, path: str = "") -> list[dict]:
        """Recursively find differences between two configs."""
        diffs = []
        keys = set(conf1.keys()) | set(conf2.keys())

        for k in keys:
            curr_path = f"{path}.{k}" if path else k
            val1 = conf1.get(k)
            val2 = conf2.get(k)

            if isinstance(val1, dict) and isinstance(val2, dict):
                diffs.extend(self._find_config_diffs(val1, val2, curr_path))
            elif val1 != val2:
                diffs.append({"Parameter": curr_path, "Session 1": str(val1), "Session 2": str(val2)})
        return diffs


def main():
    """Main entry point."""
    setup_logging(debug_mode=False)
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
