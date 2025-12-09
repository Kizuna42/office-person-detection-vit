"""Aeterlink パイプライン成果物用 Streamlit ダッシュボード."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

# パッケージ解決のため、プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.visualization.dashboard_utils import (  # noqa: E402
    SessionDataLoader,
    get_phase_status_icon,
    render_kpi_card,
)

SESSIONS_ROOT_DEFAULT = Path("output/sessions")
LATEST_SYMLINK = Path("output/latest")


def _render_phase_status(checkpoint: dict[str, Any]) -> None:
    phases: dict[str, Any] = checkpoint.get("phases", {}) if checkpoint else {}
    if not phases:
        st.info("フェーズステータスがありません。")
        return
    items = sorted(phases.items(), key=lambda kv: kv[0])
    cols = st.columns(min(4, len(items)))
    for idx, (name, info) in enumerate(items):
        with cols[idx % len(cols)]:
            status = info.get("status", "unknown")
            icon = get_phase_status_icon(status)
            ts = info.get("timestamp")
            st.markdown(f"**{icon} {name}**")
            if ts:
                st.caption(ts)


def _render_image_grid(base_dir: Path | None, filenames: list[str], title: str, max_items: int = 12) -> None:
    st.subheader(title)
    if not base_dir or not filenames:
        st.info("表示可能な画像がありません。")
        return
    display = filenames[:max_items]
    cols = st.columns(min(4, len(display)))
    for idx, name in enumerate(display):
        with cols[idx % len(cols)]:
            st.image(base_dir / name, caption=name, use_container_width=True)
    if len(filenames) > max_items:
        st.caption(f"...他 {len(filenames) - max_items} 件")


def _render_performance(perf: dict[str, Any]) -> None:
    if not perf:
        st.info("パフォーマンス情報がありません。")
        return
    rows: list[dict[str, Any]] = []
    for phase, stats in perf.items():
        rows.append(
            {
                "phase": phase,
                "avg_time(s)": stats.get("avg_time"),
                "min_time(s)": stats.get("min_time"),
                "max_time(s)": stats.get("max_time"),
                "total_time(s)": stats.get("total_time"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_zone_counts(zone_counts: pd.DataFrame | None) -> None:
    if zone_counts is None or zone_counts.empty:
        st.info("zone_counts.csv がありません。")
        return
    df = zone_counts.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    st.line_chart(df)
    st.dataframe(zone_counts, use_container_width=True, hide_index=True)


def _render_tracks_statistics(statistics: dict[str, Any]) -> None:
    if not statistics:
        st.info("tracking_statistics.json がありません。")
        return
    st.json(statistics)


def _render_transform_preview(transformations: dict[str, Any] | list[Any] | None) -> None:
    if not transformations:
        st.info("coordinate_transformations.json がありません。")
        return
    # 変換手法とフレーム数の概要のみ表示（詳細は重いので省略）
    if isinstance(transformations, dict):
        method = transformations.get("method") or transformations.get("transform_method")
        frames = transformations.get("frames", [])
        st.write(f"変換手法: {method or '不明'} / フレーム数: {len(frames)}")
    elif isinstance(transformations, list):
        st.write(f"フレームエントリ数: {len(transformations)}")


def _render_assets(phase_dir: Path | None, graphs: list[str], floormaps: list[str], videos: list[str]) -> None:
    if phase_dir is None:
        st.info("可視化成果物ディレクトリがありません。")
        return
    if graphs:
        _render_image_grid(phase_dir / "graphs", graphs, "Graphs", max_items=6)
    if floormaps:
        st.subheader("Floormaps")
        cols = st.columns(min(4, len(floormaps)))
        for idx, rel in enumerate(floormaps[:8]):
            with cols[idx % len(cols)]:
                st.image((phase_dir / "floormaps" / rel), caption=rel, use_container_width=True)
        if len(floormaps) > 8:
            st.caption(f"...他 {len(floormaps) - 8} 件")
    if videos:
        st.subheader("Videos")
        for vid in videos:
            video_path = (phase_dir / vid).resolve()
            if not video_path.exists():
                st.warning(f"動画ファイルが見つかりません: {vid}")
                st.write(f"パス: {video_path}")
                continue

            # Streamlitのst.video()はファイルパス文字列を渡すと自動的にメディアサーバー経由で配信
            try:
                st.video(str(video_path))
            except Exception as exc:
                st.warning(f"動画の再生に失敗しました: {vid} ({exc})")
                st.write(f"パス: {video_path}")

            # ダウンロードボタン用にバイトデータを読み込む
            try:
                video_bytes = video_path.read_bytes()
                st.download_button("Download", data=video_bytes, file_name=vid, mime="video/mp4", key=f"download_{vid}")
            except Exception as exc:
                st.warning(f"ダウンロード用データの読み込みに失敗しました: {exc}")


def _render_detection_stats(statistics: dict[str, Any]) -> None:
    if not statistics:
        st.info("detection_statistics.json がありません。")
        return
    st.json(statistics)


def main() -> None:
    st.set_page_config(page_title="Aeterlink Pipeline Dashboard", layout="wide")
    st.title("Aeterlink Pipeline Dashboard")
    st.caption("パイプライン成果物の可視化ビューア")

    with st.sidebar:
        st.header("セッション選択")
        sessions_root = Path(st.text_input("sessions ルート", value=str(SESSIONS_ROOT_DEFAULT)))
        loader = SessionDataLoader(sessions_root, latest_symlink=LATEST_SYMLINK)
        sessions = loader.get_available_sessions()
        if not sessions:
            st.error("セッションが見つかりません。`output/sessions` を確認してください。")
            return
        session_id = st.selectbox("セッション", options=sessions)
        st.caption(f"選択パス: {loader.get_session_path(session_id)}")
        image_limit = st.slider("画像表示上限", min_value=4, max_value=48, value=12, step=4)

    metadata = loader.load_metadata(session_id)
    summary = loader.load_summary(session_id)
    checkpoint = loader.load_pipeline_checkpoint(session_id)
    config = loader.load_config(session_id)

    extraction = loader.load_extraction(session_id)
    detection = loader.load_detection(session_id)
    tracking = loader.load_tracking(session_id)
    transform = loader.load_transform(session_id)
    aggregation = loader.load_aggregation(session_id)
    visualization = loader.load_visualization(session_id)

    tabs = st.tabs(
        [
            "Overview",
            "Extraction",
            "Detection",
            "Tracking",
            "Transform",
            "Aggregation",
            "Visualization Assets",
            "Config & Checkpoints",
        ]
    )

    # Overview
    with tabs[0]:
        st.subheader("ステータス")
        status = summary.get("status", "unknown")
        st.write(f"セッション: `{metadata.get('session_id', session_id)}` / ステータス: **{status}**")
        if "timestamp" in summary:
            st.caption(f"完了時刻: {summary['timestamp']}")

        stats = summary.get("statistics", {})
        perf = summary.get("performance", {})
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            render_kpi_card("抽出フレーム", stats.get("frames_extracted", "-"), icon="🖼️")
        with kpi_cols[1]:
            render_kpi_card("総検出数", stats.get("total_detections", "-"), icon="🎯")
        with kpi_cols[2]:
            render_kpi_card("平均検出/フレーム", stats.get("avg_detections_per_frame", "-"), icon="📈")
        with kpi_cols[3]:
            render_kpi_card("生成フロアマップ", stats.get("floormaps_generated", "-"), icon="🗺️")

        st.divider()
        st.subheader("フェーズ進捗")
        _render_phase_status(checkpoint)

        st.subheader("パフォーマンス")
        _render_performance(perf)

    # Extraction
    with tabs[1]:
        st.subheader("抽出結果")
        if extraction["results"] is not None:
            st.dataframe(extraction["results"], hide_index=True, use_container_width=True)
        _render_image_grid(
            extraction["phase_dir"] / "frames" if extraction["phase_dir"] else None,
            extraction["frames"],
            "Frames",
            max_items=image_limit,
        )

    # Detection
    with tabs[2]:
        st.subheader("検出結果")
        _render_detection_stats(detection["statistics"])
        _render_image_grid(
            detection["phase_dir"] / "images" if detection["phase_dir"] else None,
            detection["images"],
            "Detection Images",
            max_items=image_limit,
        )

    # Tracking
    with tabs[3]:
        st.subheader("追跡結果")
        _render_tracks_statistics(tracking["statistics"])
        if tracking["tracks_csv"] is not None:
            st.dataframe(tracking["tracks_csv"], hide_index=True, use_container_width=True)
        _render_image_grid(
            tracking["phase_dir"] / "images" if tracking["phase_dir"] else None,
            tracking["images"],
            "Tracking Images",
            max_items=image_limit,
        )

    # Transform
    with tabs[4]:
        st.subheader("座標変換")
        _render_transform_preview(transform["transformations"])

    # Aggregation
    with tabs[5]:
        st.subheader("集計")
        _render_zone_counts(aggregation["zone_counts"])

    # Visualization Assets
    with tabs[6]:
        st.subheader("可視化成果物")
        _render_assets(
            visualization["phase_dir"], visualization["graphs"], visualization["floormaps"], visualization["videos"]
        )

    # Config & Checkpoints
    with tabs[7]:
        st.subheader("metadata.json")
        st.json(metadata)
        st.subheader("config（metadata から）")
        st.json(config)
        st.subheader("pipeline_checkpoint.json")
        st.json(checkpoint)


if __name__ == "__main__":
    main()
