"""Aeterlink パイプライン成果物用 Streamlit ダッシュボード."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

from src.transform.floormap_config import FloorMapConfig
from src.transform.homography import HomographyTransformer
from src.transform.piecewise_affine import PiecewiseAffineTransformer

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


def _load_transformer(config: dict[str, Any] | None) -> tuple[Any | None, dict[str, Any]]:
    """Configベースでパイプラインと同じ変換器をロード."""
    if not isinstance(config, dict):
        return None, {"reason": "config_not_dict"}

    transform_cfg = config.get("transform", {}) if isinstance(config.get("transform", {}), dict) else {}
    floormap_cfg = config.get("floormap", {}) if isinstance(config.get("floormap", {}), dict) else {}
    method = str(transform_cfg.get("method", "")).lower() or "piecewise_affine"

    try:
        fm_config = FloorMapConfig.from_config(floormap_cfg)
    except Exception as exc:  # pragma: no cover - defensive
        return None, {"reason": f"floormap_config_error: {exc}"}

    # 1) PWA/TPS 優先（現在はPWAのみ実装）
    if method == "piecewise_affine":
        model_path = transform_cfg.get("model_path")
        if model_path:
            p = Path(model_path)
            if not p.is_absolute():
                p = Path(__file__).resolve().parents[2] / p
            if p.exists():
                try:
                    transformer = PiecewiseAffineTransformer.load(
                        p, floormap_config=fm_config, distortion_corrector=None
                    )
                    info = transformer.get_info()
                    info = info if isinstance(info, dict) else {"method": "piecewise_affine"}
                    return transformer, {"method": "piecewise_affine", **info, "model_path": str(p)}
                except Exception as exc:  # pragma: no cover - defensive
                    return None, {"reason": f"pwa_load_failed: {exc}", "method": "piecewise_affine"}
        # PWAモデルが無い場合はホモグラフィにフォールバック

    # 2) ホモグラフィ
    homo_cfg = config.get("homography", {}) if isinstance(config.get("homography", {}), dict) else {}
    matrix = homo_cfg.get("matrix")
    if matrix is not None:
        try:
            transformer = HomographyTransformer(np.array(matrix, dtype=np.float64), fm_config)
            return transformer, {"method": "homography"}
        except Exception as exc:  # pragma: no cover - defensive
            return None, {"reason": f"homography_init_failed: {exc}", "method": "homography"}

    return None, {"reason": "no_transformer"}


def _render_track_floor_trajectory(
    tracks_df: pd.DataFrame | None,
    floormap_dir: Path | None,
    floormaps: list[str],
    _floormap_cfg: dict[str, Any],
    transformer_info: dict[str, Any],
    transformer: Any | None,
) -> None:
    st.subheader("フロアマップ軌跡")

    if tracks_df is None or tracks_df.empty:
        st.info("tracks.csv がありません。")
        return
    if not floormaps or floormap_dir is None:
        st.info("フロアマップ画像がありません。")
        return
    if not {"track_id", "x", "y"}.issubset(tracks_df.columns):
        st.warning("tracks.csv に必要な列がありません（track_id, x, y を期待）。")
        return

    df = tracks_df.copy()
    df = df.dropna(subset=["track_id", "x", "y"])
    if df.empty:
        st.info("描画可能な軌跡データがありません。")
        return

    # 数値型に変換
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if "frame_index" in df.columns:
        df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce")

    df = df.dropna(subset=["track_id", "x", "y"])
    track_ids = sorted(df["track_id"].unique().tolist())
    if not track_ids:
        st.info("描画可能な軌跡データがありません。")
        return

    track_id = st.selectbox("表示する track_id", track_ids, format_func=lambda v: f"{int(v)}")
    map_choice = st.selectbox("使用するフロアマップ", floormaps)
    map_path = floormap_dir / map_choice
    if not map_path.exists():
        st.warning(f"フロアマップが見つかりません: {map_path}")
        return

    try:
        img = Image.open(map_path).convert("RGB")
    except Exception as exc:
        st.error(f"フロアマップの読み込みに失敗しました: {exc}")
        return

    track_points = df[df["track_id"] == track_id]
    if "frame_index" in track_points.columns:
        track_points = track_points.sort_values("frame_index")

    coords_cam = track_points[["x", "y"]].to_numpy(dtype=np.float32)
    if coords_cam.size == 0:
        st.info("選択したトラックの座標がありません。")
        return

    coords_px = coords_cam
    applied_method = "raw_camera"
    warnings: list[str] = []

    if transformer is not None:
        results = []
        for pt in coords_cam:
            try:
                res = transformer.transform_pixel((float(pt[0]), float(pt[1])))
                results.append(res)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"変換失敗: {exc}")
                results.append(None)

        coords_list: list[tuple[float, float]] = []
        out_of_bounds = 0
        invalid = 0
        for res in results:
            if res is None or not getattr(res, "is_valid", False):
                invalid += 1
                coords_list.append((np.nan, np.nan))
                continue
            fx, fy = getattr(res, "floor_coords_px", (np.nan, np.nan))
            if fx is None or fy is None:
                invalid += 1
                coords_list.append((np.nan, np.nan))
                continue
            if not getattr(res, "is_within_bounds", True):
                out_of_bounds += 1
            coords_list.append((float(fx), float(fy)))

        coords_px = np.array(coords_list, dtype=np.float32)
        applied_method = transformer_info.get("method", "transform")
        if invalid:
            warnings.append(f"{invalid} 点が変換失敗/無効です")
        if out_of_bounds:
            warnings.append(f"{out_of_bounds} 点がフロアマップ範囲外です")
    else:
        warnings.append("変換器をロードできませんでした。生のcamera座標で描画しています。")

    width, height = img.size
    coords_px = np.clip(coords_px, [0, 0], [width - 1, height - 1])
    coords = [tuple(map(float, pt)) for pt in coords_px]
    valid_coords = [(x, y) for x, y in coords if not (np.isnan(x) or np.isnan(y))]

    draw = ImageDraw.Draw(img)
    if len(valid_coords) > 1:
        draw.line(valid_coords, fill=(0, 173, 255), width=4)
    radius = 5
    for x, y in valid_coords:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 69, 58))

    # 端に張り付いた点が多い場合は警告
    boundary_hits = sum(1 for x, y in valid_coords if x in (0.0, width - 1.0) or y in (0.0, height - 1.0))
    if boundary_hits > 0:
        st.warning(
            f"{boundary_hits}/{len(valid_coords)} 点が画像端にクリップされています。座標変換/スケールを確認してください。"
        )

    st.image(img, caption=f"track_id={int(track_id)} on {map_choice}", use_container_width=True)
    st.caption(f"軌跡点数: {len(valid_coords)}")

    transformed_df = track_points.copy()
    transformed_df["floor_x_px"] = coords_px[:, 0]
    transformed_df["floor_y_px"] = coords_px[:, 1]
    columns = [
        col
        for col in ["frame_index", "x", "y", "floor_x_px", "floor_y_px", "confidence", "zone_ids"]
        if col in transformed_df.columns
    ]
    st.dataframe(transformed_df[columns] if columns else transformed_df, hide_index=True, use_container_width=True)

    method_label = f"変換: {applied_method}"
    if transformer_info:
        if transformer_info.get("method") == "piecewise_affine":
            method_label += f" (PWA; model={transformer_info.get('model_path', '-')})"
        elif transformer_info.get("method") == "homography":
            method_label += " (Homography from config)"
    st.caption(method_label)
    for w in warnings:
        st.warning(w)


def _track_color_rgb(track_id: int) -> tuple[int, int, int]:
    """track_id に基づく安定色 (RGB)。"""
    hue = (track_id * 137) % 180
    color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
    return (r, g, b)


def _transform_points(points: np.ndarray, transformer: Any | None) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """camera座標→フロア座標への変換（統計付き）。"""
    if transformer is None or points.size == 0:
        return points, {}, ["変換器が無いためcamera座標のまま描画します。"]

    coords_list: list[tuple[float, float]] = []
    stats = {"invalid": 0, "oob": 0}
    warnings: list[str] = []

    for pt in points:
        try:
            res = transformer.transform_pixel((float(pt[0]), float(pt[1])))
        except Exception as exc:  # pragma: no cover - defensive
            stats["invalid"] += 1
            warnings.append(f"変換失敗: {exc}")
            coords_list.append((np.nan, np.nan))
            continue

        if not getattr(res, "is_valid", False):
            stats["invalid"] += 1
            coords_list.append((np.nan, np.nan))
            continue
        fx, fy = getattr(res, "floor_coords_px", (np.nan, np.nan))
        if fx is None or fy is None:
            stats["invalid"] += 1
            coords_list.append((np.nan, np.nan))
            continue
        if not getattr(res, "is_within_bounds", True):
            stats["oob"] += 1
        coords_list.append((float(fx), float(fy)))

    coords = np.array(coords_list, dtype=np.float32)
    if stats["invalid"]:
        warnings.append(f"{stats['invalid']} 点が変換失敗/無効です")
    if stats["oob"]:
        warnings.append(f"{stats['oob']} 点がフロアマップ範囲外です")
    return coords, stats, warnings


def _render_time_series_view(
    tracking: dict[str, Any],
    visualization: dict[str, Any],
    _floormap_cfg: dict[str, Any],
    transformer_info: dict[str, Any],
    transformer: Any | None,
) -> None:
    st.subheader("時系列ビュー（カメラ + フロアマップ）")

    tracks_df: pd.DataFrame | None = tracking.get("tracks_csv")
    if tracks_df is None or tracks_df.empty:
        st.info("tracks.csv がありません。")
        return

    if "frame_index" not in tracks_df.columns:
        st.warning("tracks.csv に frame_index 列がありません。")
        return

    if not visualization.get("floormaps"):
        st.info("フロアマップ画像がありません。")
        return

    floormap_dir = visualization["phase_dir"] / "floormaps" if visualization["phase_dir"] else None
    map_choice = st.selectbox("フロアマップを選択", visualization["floormaps"])
    map_path = floormap_dir / map_choice if floormap_dir else None
    if not map_path or not map_path.exists():
        st.warning("選択したフロアマップが見つかりません。")
        return

    # フレーム一覧
    frame_indices = sorted(pd.to_numeric(tracks_df["frame_index"], errors="coerce").dropna().astype(int).unique())
    if not frame_indices:
        st.info("frame_index がありません。")
        return

    # 画像との対応付け（長さ一致なら単純対応）
    images = tracking.get("images", [])
    image_map: dict[int, Path] = {}
    if tracking.get("phase_dir") and len(images) == len(frame_indices):
        base = tracking["phase_dir"] / "images"
        for idx, img_name in zip(frame_indices, sorted(images), strict=False):
            image_map[idx] = base / img_name
    elif tracking.get("phase_dir") and images:
        st.warning("tracking画像とframe_indexの件数が一致しません。画像対応はスキップします。")

    default_frame = frame_indices[0]
    current_frame = st.slider(
        "frame_index", min_value=frame_indices[0], max_value=frame_indices[-1], value=default_frame
    )
    history_len = st.slider("軌跡の履歴長", min_value=1, max_value=50, value=15, step=1)

    # フロアマップ画像読み込み
    try:
        floormap_img = Image.open(map_path).convert("RGB")
    except Exception as exc:
        st.error(f"フロアマップ読み込みに失敗: {exc}")
        return

    # 現在フレームと履歴
    current_points = tracks_df[tracks_df["frame_index"] == current_frame]
    history_points = tracks_df[tracks_df["frame_index"] <= current_frame]

    # 変換
    coords_current, _, warn_current = _transform_points(
        current_points[["x", "y"]].to_numpy(dtype=np.float32), transformer
    )
    _, _, warn_history_global = _transform_points(history_points[["x", "y"]].to_numpy(dtype=np.float32), transformer)

    width, height = floormap_img.size

    def _clip_and_clean(coords: np.ndarray) -> np.ndarray:
        coords = np.clip(coords, [0, 0], [width - 1, height - 1])
        mask = ~np.isnan(coords).any(axis=1)
        return coords[mask]

    coords_current = _clip_and_clean(coords_current)

    # 描画
    draw = ImageDraw.Draw(floormap_img)

    # trackごとに履歴を描画
    grouped = history_points.groupby("track_id")
    for tid, group in grouped:
        tid_int = int(tid)
        rgb = _track_color_rgb(tid_int)
        pts_cam = group[["x", "y"]].to_numpy(dtype=np.float32)
        pts_px, _, _ = _transform_points(pts_cam, transformer)
        pts_px = _clip_and_clean(pts_px)
        if pts_px.shape[0] < 1:
            continue
        pts_px = pts_px[-history_len:]
        if pts_px.shape[0] > 1:
            draw.line([tuple(map(float, p)) for p in pts_px], fill=rgb, width=3)
        last = pts_px[-1]
        r = 5
        draw.ellipse((last[0] - r, last[1] - r, last[0] + r, last[1] + r), fill=rgb)

    st.image(floormap_img, caption=f"frame_index={current_frame} 全トラック", use_container_width=True)

    # カメラ画像の表示（対応がある場合のみ）
    if current_frame in image_map and image_map[current_frame].exists():
        st.image(str(image_map[current_frame]), caption="tracking image", use_container_width=True)
    else:
        st.caption("対応するtracking画像が見つかりません。")

    # テーブル（現在フレーム）
    if not current_points.empty:
        coords_cur_full, _, _ = _transform_points(current_points[["x", "y"]].to_numpy(dtype=np.float32), transformer)
        cur_df = current_points.copy()
        cur_df["floor_x_px"] = coords_cur_full[:, 0]
        cur_df["floor_y_px"] = coords_cur_full[:, 1]
        cols = [
            c
            for c in ["track_id", "frame_index", "x", "y", "floor_x_px", "floor_y_px", "confidence", "zone_ids"]
            if c in cur_df.columns
        ]
        st.dataframe(cur_df[cols], hide_index=True, use_container_width=True)

    # 情報表示
    method_label = f"変換: {transformer_info.get('method', 'raw_camera')}"
    st.caption(method_label)
    for w in warn_current + warn_history_global:
        st.warning(w)


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
        floormap_dir = visualization["phase_dir"] / "floormaps" if visualization["phase_dir"] else None
        floormap_cfg = config.get("floormap", {}) if isinstance(config, dict) else {}
        transformer, transformer_info = _load_transformer(config)
        _render_track_floor_trajectory(
            tracking["tracks_csv"],
            floormap_dir,
            visualization["floormaps"],
            floormap_cfg,
            transformer_info,
            transformer,
        )
        _render_time_series_view(tracking, visualization, floormap_cfg, transformer_info, transformer)

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
