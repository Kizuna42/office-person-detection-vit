"""Streamlit-based interactive visualization tool for tracking results."""

import json
import logging
from pathlib import Path
import subprocess
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config.config_manager import ConfigManager
from src.models.data_models import Detection
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.track import Track
from src.utils.export_utils import TrajectoryExporter
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class InteractiveVisualizer:
    """インタラクティブ可視化クラス

    Streamlitを使用して追跡結果をインタラクティブに可視化します。
    """

    def __init__(self, session_dir: str | Path, config_path: str = "config.yaml"):
        """InteractiveVisualizerを初期化

        Args:
            session_dir: セッションディレクトリのパス
            config_path: 設定ファイルのパス
        """
        self.session_dir = Path(session_dir)
        self.config = ConfigManager(config_path)

        if not self.session_dir.exists():
            raise FileNotFoundError(f"セッションディレクトリが見つかりません: {session_dir}")

        logger.info(f"InteractiveVisualizer initialized: {self.session_dir}")

    def load_tracks_data(self) -> list[dict]:
        """トラックデータを読み込む

        Returns:
            トラックデータのリスト
        """
        # 複数のパスを検索
        possible_paths = [
            self.session_dir / "phase2.5_tracking" / "tracks.json",
            self.session_dir / "tracks.json",
        ]

        for tracks_file in possible_paths:
            if tracks_file.exists():
                with open(tracks_file, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("tracks", [])

        return []

    def load_floormap(self) -> np.ndarray:
        """フロアマップ画像を読み込む

        Returns:
            フロアマップ画像
        """
        floormap_path = self.config.get("floormap.image_path")
        floormap = cv2.imread(floormap_path)
        if floormap is None:
            raise ValueError(f"フロアマップ画像を読み込めません: {floormap_path}")
        return floormap

    def load_summary_data(self) -> dict | None:
        """セッションサマリーデータを読み込む

        Returns:
            サマリーデータの辞書、存在しない場合はNone
        """
        summary_file = self.session_dir / "summary.json"
        if not summary_file.exists():
            return None

        with open(summary_file, encoding="utf-8") as f:
            return json.load(f)

    def calculate_session_aggregated_stats(self, tracks_data: list[dict]) -> dict:
        """セッション集約統計を計算

        Args:
            tracks_data: トラックデータのリスト

        Returns:
            集約統計の辞書
        """
        if not tracks_data:
            return {
                "total_visitors": 0,
                "max_concurrent": 0,
                "zone_averages": {},
            }

        # 総滞在人数（ユニークなトラックID数）
        total_visitors = len(tracks_data)

        # 最大同時在室人数（全フレームでの最大同時人数）
        max_frames = max((len(track.get("trajectory", [])) for track in tracks_data), default=0)
        frame_counts = []
        for frame_idx in range(max_frames):
            frame_count = 0
            for track in tracks_data:
                trajectory = track.get("trajectory", [])
                if frame_idx < len(trajectory):
                    frame_count += 1
            frame_counts.append(frame_count)
        max_concurrent = max(frame_counts) if frame_counts else 0

        # ゾーン別平均人数
        # トラックデータからゾーン情報を取得（可能であれば）
        # 現時点ではトラックデータにゾーン情報がないため、軌跡から推定する
        zone_averages = {}
        zones = self.config.get("zones", [])
        for zone in zones:
            zone_id = zone.get("id", "")
            zone_polygon = zone.get("polygon", [])

            # 各フレームでゾーン内にいる人数をカウント
            zone_frame_counts = []
            for frame_idx in range(max_frames):
                count = 0
                for track in tracks_data:
                    trajectory = track.get("trajectory", [])
                    if frame_idx < len(trajectory):
                        pt = trajectory[frame_idx]
                        x, y = pt.get("x", 0), pt.get("y", 0)
                        # 点 in 多角形判定（簡易版）
                        if self._point_in_polygon(x, y, zone_polygon):
                            count += 1
                zone_frame_counts.append(count)

            zone_averages[zone_id] = np.mean(zone_frame_counts) if zone_frame_counts else 0.0

        return {
            "total_visitors": total_visitors,
            "max_concurrent": max_concurrent,
            "zone_averages": zone_averages,
        }

    def _point_in_polygon(self, x: float, y: float, polygon: list[list[float]]) -> bool:
        """点が多角形内にあるか判定（Ray Casting法）

        ZoneClassifierと同じアルゴリズムを使用します。

        Args:
            x: X座標
            y: Y座標
            polygon: 多角形の頂点リスト [[x1, y1], [x2, y2], ...]

        Returns:
            多角形内にある場合True
        """
        if not polygon or len(polygon) < 3:
            return False

        n = len(polygon)
        inside = False

        # 多角形の各辺について判定
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]

            # 点のy座標が辺のy座標範囲内にあるかチェック
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                # 辺が垂直でない場合
                if p1y != p2y:
                    # 交点のx座標を計算
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                # 辺が垂直、または点が交点より左側にある場合
                if p1x == p2x or x <= xinters:
                    inside = not inside

            p1x, p1y = p2x, p2y

        return inside

    def estimate_id_switches(self, tracks_data: list[dict]) -> int:
        """ID Switch数の推計

        軌跡の不連続性や短いトラック数を基にID Switchを推計します。
        実際のID SwitchはGround Truthデータが必要ですが、ここでは推計値を返します。

        Args:
            tracks_data: トラックデータのリスト

        Returns:
            推計ID Switch数
        """
        if not tracks_data:
            return 0

        # 短いトラック（軌跡長が短い）はID Switchの可能性が高い
        short_tracks = 0
        total_trajectory_length = 0

        for track in tracks_data:
            trajectory = track.get("trajectory", [])
            trajectory_length = len(trajectory)
            total_trajectory_length += trajectory_length

            # 軌跡長が平均の50%以下の場合は短いトラックとみなす
            if trajectory_length < 5:  # 最低5フレーム以下
                short_tracks += 1

        # 推計: 短いトラック数 × 0.5（経験的な係数）
        estimated_switches = int(short_tracks * 0.5)

        return estimated_switches

    def run_mot_evaluation(self, gt_path: Path | None = None) -> dict | None:
        """MOTメトリクス評価を実行

        Args:
            gt_path: Ground Truthファイルのパス（Noneの場合は自動検索）

        Returns:
            評価結果の辞書、エラー時はNone
        """
        # トラックファイルの検索
        possible_tracks_paths = [
            self.session_dir / "phase2.5_tracking" / "tracks.json",
            self.session_dir / "tracks.json",
        ]

        tracks_file = None
        for path in possible_tracks_paths:
            if path.exists():
                tracks_file = path
                break

        if tracks_file is None:
            return None

        # Ground Truthファイルの検索
        if gt_path is None:
            # output/shared/labels/result_fixed.json を探す
            gt_path = Path("output/shared/labels/result_fixed.json")
            if not gt_path.exists():
                # セッションディレクトリ内を探す
                gt_path = self.session_dir / "labels" / "result_fixed.json"
                if not gt_path.exists():
                    return None

        # フレーム数を取得
        tracks_data = self.load_tracks_data()
        max_frames = max((len(track.get("trajectory", [])) for track in tracks_data), default=0)
        if max_frames == 0:
            return None

        try:
            # MOTメトリクス評価スクリプトを実行
            script_path = Path("scripts/evaluate_mot_metrics.py")
            if not script_path.exists():
                return None

            output_file = self.session_dir / "mot_metrics.json"

            result = subprocess.run(
                [
                    "python",
                    str(script_path),
                    "--gt",
                    str(gt_path),
                    "--tracks",
                    str(tracks_file),
                    "--frames",
                    str(max_frames),
                    "--output",
                    str(output_file),
                    "--config",
                    "config.yaml",
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            if result.returncode != 0:
                logger.error(f"MOT評価エラー: {result.stderr}")
                return None

            # 結果ファイルを読み込む
            if output_file.exists():
                with open(output_file, encoding="utf-8") as f:
                    return json.load(f)

        except Exception as e:
            logger.exception(f"MOT評価実行エラー: {e}")
            return None

        return None

    def render_app(self) -> None:
        """Streamlitアプリをレンダリング"""
        st.set_page_config(
            page_title="Tracking Visualization", page_icon="📊", layout="wide", initial_sidebar_state="expanded"
        )

        # カスタムCSSでスタイリング（ミニマルデザイン）
        st.markdown(
            """
        <style>
        .main > div {
            padding-top: 1rem;
        }

        /* タイトルのスタイリング */
        h1 {
            color: #1f77b4;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            font-weight: 600;
        }

        /* 見出しのスタイリング */
        h2 {
            color: #2c3e50;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        h3 {
            color: #34495e;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        /* メトリクスカードのスタイリング（ミニマルデザイン） */
        [data-testid="stMetric"] {
            background: #f0f0f0;
            background-color: #404040;
            border: 2px solid #f8f8f8;
            border-radius: 8px;
            box-shadow: none !important;
            padding: 0.75rem 0.5rem !important;
            margin: 0.5rem auto !important;
            min-height: auto !important;
            width: 100% !important;
        }

        /* メトリクスコンテナ全体の中央揃え */
        [data-testid="stMetric"] > div,
        [data-testid="stMetricContainer"],
        [data-testid="stMetric"] div[class*="metric"],
        [data-testid="stMetric"] div[class*="stMetric"] {
            text-align: center !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.375rem !important;
            margin: 0 auto !important;
            width: 100% !important;
            background: transparent !important;
        }

        /* メトリクスのラベル（中央揃え、読みやすい色） */
        [data-testid="stMetric"] label,
        [data-testid="stMetric"] label[class*="label"],
        [data-testid="stMetric"] [class*="label"] {
            color: #495057 !important;
            font-weight: 500 !important;
            font-size: 0.8125rem !important;
            text-align: center !important;
            display: block !important;
            margin: 0 auto 0.5rem auto !important;
            width: 100% !important;
            line-height: 1.4 !important;
            background: transparent !important;
        }

        /* メトリクスの値（中央揃え、黒色で明確に） */
        [data-testid="stMetric"] [data-testid="stMarkdownContainer"],
        [data-testid="stMetric"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stMetric"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stMetric"] [data-testid="stMarkdownContainer"] div,
        [data-testid="stMetric"] [class*="value"],
        [data-testid="stMetric"] [class*="stMarkdownContainer"] {
            color: #000000 !important;
            font-weight: 600 !important;
            font-size: 1.75rem !important;
            text-align: center !important;
            margin: 0.375rem auto !important;
            display: block !important;
            line-height: 1.5 !important;
            width: 100% !important;
            background: transparent !important;
        }

        /* メトリクスのデルタ値（中央揃え） */
        [data-testid="stMetric"] [data-testid="stMarkdownContainer"] small,
        [data-testid="stMetricDelta"],
        [data-testid="stMetric"] [class*="delta"],
        [data-testid="stMetric"] small {
            color: #6c757d !important;
            font-weight: 400 !important;
            font-size: 0.8125rem !important;
            text-align: center !important;
            display: block !important;
            margin: 0.375rem auto 0 auto !important;
            width: 100% !important;
            background: transparent !important;
        }

        /* セクションの区切り（ミニマル、余白調整） */
        hr {
            margin: 1.25rem 0;
            border: none;
            border-top: 1px solid #dee2e6;
        }

        /* サイドバーの余白調整 */
        section[data-testid="stSidebar"] {
            padding-top: 1rem;
        }

        /* キャプションのスタイリング（ミニマル） */
        .stCaption {
            color: #6c757d;
            font-size: 0.8125rem;
            text-align: center;
            margin-top: 0.25rem;
        }

        /* ボタンのスタイリング（ミニマル） */
        .stButton > button {
            width: 100%;
            border-radius: 4px;
            font-weight: 500;
            padding: 0.5rem 1rem;
            margin: 0.25rem 0;
        }

        /* テーブルのスタイリング（ミニマル） */
        .dataframe {
            font-size: 0.875rem;
            margin: 0.5rem 0;
        }

        /* カラムの余白調整 */
        .element-container {
            margin-bottom: 0.75rem;
        }

        /* メトリクスカラムの中央揃え */
        [data-testid="column"] [data-testid="stMetric"] {
            margin: 0.5rem auto;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        st.title("📊 オブジェクト追跡可視化ツール")
        st.caption("リアルタイムでトラック結果を可視化・分析・比較できます")

        # サイドバー: コントロール
        with st.sidebar:
            st.header("⚙️ 設定")

            # セッション選択
            sessions_dir = Path("output/sessions")
            if sessions_dir.exists():
                sessions = sorted([d.name for d in sessions_dir.iterdir() if d.is_dir()], reverse=True)
                selected_session = st.selectbox("セッションを選択", sessions, key="session_select")
                if selected_session:
                    self.session_dir = sessions_dir / selected_session
                    st.session_state.session_dir = str(self.session_dir)

            # タブ選択
            st.divider()
            st.subheader("📑 ビュー選択")
            view_mode = st.radio(
                "表示モード",
                ["フレームビュー", "集約ビュー", "セッション比較"],
                key="view_mode",
                label_visibility="collapsed",
            )

            # フレームビュー用のフィルタ設定
            if view_mode == "フレームビュー":
                st.divider()
                st.subheader("🎨 フィルタ")
                show_trajectories = st.checkbox("軌跡を表示", value=True)
                show_ids = st.checkbox("IDを表示", value=True)

                # IDフィルタ
                tracks_data = self.load_tracks_data()
                if tracks_data:
                    track_ids = [track.get("track_id", 0) for track in tracks_data]
                    selected_ids = st.multiselect("表示するIDを選択", track_ids, default=track_ids)
                else:
                    selected_ids = []

                # ゾーンフィルタ
                zones = self.config.get("zones", [])
                zone_ids = [zone.get("id", "") for zone in zones]
                selected_zones = st.multiselect("表示するゾーンを選択", zone_ids, default=zone_ids)

                # 軌跡の長さ制限
                max_trajectory_length = st.slider("軌跡の最大長", 10, 100, 50)
            else:
                tracks_data = self.load_tracks_data()
                selected_ids = []
                selected_zones = []
                max_trajectory_length = 50
                show_trajectories = True
                show_ids = True

        # メインエリア: タブ別の表示
        if view_mode == "フレームビュー":
            self._render_frame_view(
                tracks_data, show_trajectories, show_ids, selected_ids, selected_zones, max_trajectory_length
            )
        elif view_mode == "集約ビュー":
            self._render_aggregated_view(tracks_data)
        elif view_mode == "セッション比較":
            self._render_session_comparison()

    def _render_frame_view(
        self,
        tracks_data: list[dict],
        show_trajectories: bool,
        show_ids: bool,
        selected_ids: list[int],
        selected_zones: list[str],
        max_trajectory_length: int,
    ) -> None:
        """フレームビューをレンダリング"""
        if not tracks_data:
            st.warning("⚠️ トラックデータが見つかりません")
            return

        # サマリーメトリクス（カード形式）
        max_frames = max((len(track.get("trajectory", [])) for track in tracks_data), default=0)
        if max_frames == 0:
            st.info("ℹ️ トラックデータがありません")
            return

        total_points = sum(len(track.get("trajectory", [])) for track in tracks_data)
        avg_trajectory_length = total_points / len(tracks_data) if tracks_data else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 トラック数", len(tracks_data), help="追跡中のオブジェクト数")
        with col2:
            st.metric("📊 総軌跡点数", total_points, help="全トラックの軌跡ポイント数")
        with col3:
            st.metric("📏 平均軌跡長", f"{avg_trajectory_length:.1f}", help="トラックあたりの平均軌跡長")
        with col4:
            st.metric("🎬 総フレーム数", max_frames, help="利用可能なフレーム数")

        st.divider()

        # メインコンテンツエリア
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🗺️ フロアマップ可視化")

            # フレーム情報表示
            frame_info_col1, frame_info_col2 = st.columns([3, 1])
            with frame_info_col1:
                frame_idx = st.slider(
                    "📹 フレーム選択",
                    0,
                    max_frames - 1,
                    0,
                    key="frame_slider",
                    help="スライダーでフレームを選択して軌跡を確認できます",
                )
            with frame_info_col2:
                current_frame_count = sum(1 for track in tracks_data if frame_idx < len(track.get("trajectory", [])))
                st.metric("現在の人数", current_frame_count)

            # 可視化画像を生成
            floormap = self.load_floormap()
            vis_image = self._draw_tracks(
                floormap.copy(),
                tracks_data,
                frame_idx,
                show_trajectories,
                show_ids,
                selected_ids if tracks_data else [],
                selected_zones,
                max_trajectory_length,
            )

            st.image(vis_image, use_container_width=True)

            # フレーム別人数推移グラフ
            with st.expander("📈 フレーム別人数推移グラフ", expanded=False):
                frame_counts = []
                for f_idx in range(max_frames):
                    count = sum(1 for track in tracks_data if f_idx < len(track.get("trajectory", [])))
                    frame_counts.append(count)

                chart_data = pd.DataFrame({"フレーム": range(max_frames), "人数": frame_counts})
                st.line_chart(chart_data.set_index("フレーム"), height=300, use_container_width=True)

        with col2:
            st.subheader("📊 トラック統計")

            # 軌跡長の分布
            trajectory_lengths = [len(track.get("trajectory", [])) for track in tracks_data]
            if trajectory_lengths:
                length_data = pd.DataFrame({"軌跡長": trajectory_lengths})
                st.bar_chart(length_data, height=200, use_container_width=True)

            # トラック情報テーブル
            st.divider()
            st.subheader("📋 トラック情報")

            # 上位10件を表示
            sorted_tracks = sorted(tracks_data, key=lambda x: len(x.get("trajectory", [])), reverse=True)[:10]

            track_info = []
            for track in sorted_tracks:
                track_info.append(
                    {
                        "ID": track.get("track_id", 0),
                        "軌跡長": len(track.get("trajectory", [])),
                        "年齢": track.get("age", 0),
                        "ヒット数": track.get("hits", 0),
                    }
                )

            if track_info:
                df_tracks = pd.DataFrame(track_info)
                st.dataframe(df_tracks, use_container_width=True, hide_index=True)

                # 統計サマリー
                st.caption(
                    f"📊 平均軌跡長: {df_tracks['軌跡長'].mean():.1f} | 最大: {df_tracks['軌跡長'].max()} | 最小: {df_tracks['軌跡長'].min()}"
                )

            # エクスポート機能
            st.divider()
            st.subheader("💾 データエクスポート")

            export_format = st.selectbox(
                "エクスポート形式を選択",
                ["CSV", "JSON", "画像シーケンス", "動画"],
                key="export_format",
                help="データのエクスポート形式を選択します",
            )

            # 形式別の説明
            format_descriptions = {
                "CSV": "📄 CSV形式でトラックデータをエクスポート",
                "JSON": "📋 JSON形式でトラックデータをエクスポート",
                "画像シーケンス": "🖼️ フレームごとの画像をエクスポート",
                "動画": "🎬 トラック結果を動画としてエクスポート",
            }
            st.caption(format_descriptions.get(export_format, ""))

            if st.button("🚀 エクスポート実行", type="primary", use_container_width=True):
                try:
                    exporter = TrajectoryExporter(self.session_dir / "exports")
                    tracks = self._convert_to_tracks(tracks_data)

                    with st.spinner(f"🔄 {export_format}形式でエクスポート中..."):
                        if export_format == "CSV":
                            output_path = exporter.export_csv(tracks, filename="exported_tracks.csv")
                            st.success(f"✅ CSVエクスポート完了: {output_path}")
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "📥 CSVをダウンロード", f.read(), "tracks.csv", "text/csv", use_container_width=True
                                )

                        elif export_format == "JSON":
                            output_path = exporter.export_json(tracks, filename="exported_tracks.json")
                            st.success(f"✅ JSONエクスポート完了: {output_path}")
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "📥 JSONをダウンロード",
                                    f.read(),
                                    "tracks.json",
                                    "application/json",
                                    use_container_width=True,
                                )

                        elif export_format == "画像シーケンス":
                            floormap = self.load_floormap()
                            output_paths = exporter.export_image_sequence(tracks, floormap)
                            st.success(f"✅ 画像シーケンスエクスポート完了: {len(output_paths)}フレーム")
                            st.info(f"📁 出力先: {output_paths[0].parent}")

                        elif export_format == "動画":
                            floormap = self.load_floormap()
                            output_path = exporter.export_video(tracks, floormap, filename="exported_trajectories.mp4")
                            st.success(f"✅ 動画エクスポート完了: {output_path}")
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "📥 動画をダウンロード",
                                    f.read(),
                                    "trajectories.mp4",
                                    "video/mp4",
                                    use_container_width=True,
                                )

                except Exception as e:
                    st.error(f"❌ エクスポートエラー: {e}")
                    logger.exception("Export error")

    def _render_aggregated_view(self, tracks_data: list[dict]) -> None:
        """集約ビューをレンダリング"""
        if not tracks_data:
            st.warning("⚠️ トラックデータが見つかりません")
            return

        # セッション集約統計
        st.header("📊 セッション集約統計")

        aggregated_stats = self.calculate_session_aggregated_stats(tracks_data)
        estimated_id_switches = self.estimate_id_switches(tracks_data)

        # メトリクスカード（4列）
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 総滞在人数", aggregated_stats["total_visitors"], help="ユニークなトラックID数")
        with col2:
            st.metric("👨‍👩‍👧‍👦 最大同時在室", aggregated_stats["max_concurrent"], help="全フレームでの最大同時人数")
        with col3:
            st.metric("🔄 推計ID Switch", estimated_id_switches, help="※推計値です")
        with col4:
            total_tracks = len(tracks_data)
            total_points = sum(len(track.get("trajectory", [])) for track in tracks_data)
            avg_trajectory_length = total_points / total_tracks if total_tracks > 0 else 0
            st.metric("📏 平均軌跡長", f"{avg_trajectory_length:.1f}フレーム")

        # ゾーン別平均人数
        st.divider()
        st.subheader("📍 ゾーン別平均人数")
        if aggregated_stats["zone_averages"]:
            zones = self.config.get("zones", [])
            zone_names = {zone.get("id"): zone.get("name", zone.get("id")) for zone in zones}

            # メトリクス表示
            zone_count = len(aggregated_stats["zone_averages"])
            zone_cols = st.columns(zone_count) if zone_count <= 3 else st.columns(3)

            for idx, (zone_id, avg_count) in enumerate(aggregated_stats["zone_averages"].items()):
                zone_name = zone_names.get(zone_id, zone_id)
                col_idx = idx % 3
                with zone_cols[col_idx]:
                    st.metric(f"{zone_name}", f"{avg_count:.2f}人", delta=None)

            # ゾーン別人数のグラフ
            st.divider()
            zone_chart_data = {
                zone_names.get(zone_id, zone_id): avg_count
                for zone_id, avg_count in aggregated_stats["zone_averages"].items()
            }
            st.bar_chart(zone_chart_data, height=300, use_container_width=True)

        # トラック品質指標
        st.divider()
        st.subheader("🎯 トラック品質指標")

        col1, col2, col3 = st.columns(3)
        trajectory_lengths = [len(track.get("trajectory", [])) for track in tracks_data]
        max_length = max(trajectory_lengths) if trajectory_lengths else 0
        min_length = min(trajectory_lengths) if trajectory_lengths else 0

        with col1:
            st.metric("📏 平均軌跡長", f"{avg_trajectory_length:.2f}フレーム", help="トラックあたりの平均軌跡長")
        with col2:
            st.metric("📊 軌跡長範囲", f"{min_length} - {max_length}フレーム", help="最小値から最大値までの範囲")
        with col3:
            st.metric("🔄 推計ID Switch", estimated_id_switches, help="※推計値です")

        # 軌跡長の分布グラフ
        if trajectory_lengths:
            st.divider()
            st.write("**📈 軌跡長分布**")
            length_df = pd.DataFrame({"軌跡長": trajectory_lengths})
            st.bar_chart(length_df, height=300, use_container_width=True)

        st.caption("ℹ️ ID Switch数は推計値です。正確な値はGround Truthデータが必要です。")

        # MOTメトリクス
        st.divider()
        st.subheader("🏆 MOTメトリクス")

        # 既存の評価結果を読み込む
        mot_metrics_file = self.session_dir / "mot_metrics.json"
        if mot_metrics_file.exists():
            with open(mot_metrics_file, encoding="utf-8") as f:
                mot_result = json.load(f)
                metrics = mot_result.get("metrics", {})
                if metrics:
                    # メトリクスカード
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        mota = metrics.get("MOTA", 0.0)
                        mota_target = mot_result.get("targets", {}).get("MOTA", 0.7)
                        delta_mota = mota - mota_target
                        status_icon = "✅" if mota >= mota_target else "⚠️"
                        st.metric(
                            f"{status_icon} MOTA",
                            f"{mota:.3f}",
                            delta=f"{delta_mota:+.3f}",
                            help=f"目標値: {mota_target} | Multiple Object Tracking Accuracy",
                        )

                    with col2:
                        idf1 = metrics.get("IDF1", 0.0)
                        idf1_target = mot_result.get("targets", {}).get("IDF1", 0.8)
                        delta_idf1 = idf1 - idf1_target
                        status_icon = "✅" if idf1 >= idf1_target else "⚠️"
                        st.metric(
                            f"{status_icon} IDF1",
                            f"{idf1:.3f}",
                            delta=f"{delta_idf1:+.3f}",
                            help=f"目標値: {idf1_target} | Identity F1-score",
                        )

                    with col3:
                        id_switches = metrics.get("ID_Switches", 0)
                        st.metric(
                            "🔄 ID Switches",
                            f"{id_switches:.0f}",
                            delta=None,
                            help="IDが切り替わった回数（少ないほど良い）",
                        )

                    with col4:
                        achieved = mot_result.get("achieved", {})
                        if achieved.get("MOTA") and achieved.get("IDF1"):
                            st.success("🎉 **目標達成**")
                        else:
                            st.warning("⚠️ **目標未達成**")

                    # メトリクス比較グラフ
                    st.divider()
                    st.write("**📊 メトリクス比較（目標値との比較）**")

                    # 比較チャート（現在値 vs 目標値）
                    comparison_data = {"MOTA": [mota, mota_target], "IDF1": [idf1, idf1_target]}
                    comparison_df = pd.DataFrame(comparison_data, index=["現在値", "目標値"])
                    st.bar_chart(comparison_df, height=300, use_container_width=True)

                    # 達成率の表示
                    mota_achievement = (mota / mota_target * 100) if mota_target > 0 else 0
                    idf1_achievement = (idf1 / idf1_target * 100) if idf1_target > 0 else 0

                    col1, col2 = st.columns(2)
                    with col1:
                        st.progress(min(mota_achievement / 100, 1.0))
                        st.caption(f"MOTA達成率: {mota_achievement:.1f}%")
                    with col2:
                        st.progress(min(idf1_achievement / 100, 1.0))
                        st.caption(f"IDF1達成率: {idf1_achievement:.1f}%")

                    # 詳細情報
                    with st.expander("ℹ️ メトリクス詳細情報", expanded=False):
                        st.write("**MOTA (Multiple Object Tracking Accuracy)**")
                        st.write(f"- 現在値: {mota:.3f}")
                        st.write(f"- 目標値: {mota_target}")
                        st.write(f"- 差分: {delta_mota:+.3f}")
                        st.write(f"- 達成状況: {'✅ 達成' if mota >= mota_target else '⚠️ 未達成'}")

                        st.divider()

                        st.write("**IDF1 (Identity F1-score)**")
                        st.write(f"- 現在値: {idf1:.3f}")
                        st.write(f"- 目標値: {idf1_target}")
                        st.write(f"- 差分: {delta_idf1:+.3f}")
                        st.write(f"- 達成状況: {'✅ 達成' if idf1 >= idf1_target else '⚠️ 未達成'}")

                        st.divider()

                        st.write("**ID Switches**")
                        st.write(f"- 現在値: {id_switches:.0f}")
                        st.write("- 説明: IDが切り替わった回数。少ないほど追跡の一貫性が高い")
        else:
            st.info(
                "ℹ️ MOTメトリクスはGround Truthデータが必要です。「MOTメトリクス計算」ボタンから評価を実行してください。"
            )

        # MOTメトリクス評価ボタン
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔧 MOTメトリクス計算", type="primary", use_container_width=True):
                with st.spinner("🔄 MOTメトリクスを計算中..."):
                    result = self.run_mot_evaluation()
                    if result:
                        st.success("✅ MOTメトリクス計算が完了しました")
                        st.rerun()
                    else:
                        st.error(
                            "❌ MOTメトリクス計算に失敗しました。Ground Truthファイルが見つからない可能性があります。"
                        )

        # パフォーマンス情報
        st.divider()
        st.subheader("⚡ パフォーマンス情報")
        summary_data = self.load_summary_data()
        if summary_data and "performance" in summary_data:
            perf = summary_data["performance"]

            # 主要フェーズの処理時間を取得
            phase_times = {}
            phase_names = {
                "phase1_extraction": "フェーズ1: フレーム抽出",
                "phase2_detection": "フェーズ2: 人物検出",
                "phase2.5_tracking": "フェーズ2.5: オブジェクト追跡",
                "phase3_transform": "フェーズ3: 座標変換",
                "phase4_aggregation": "フェーズ4: 集計",
                "phase5_visualization": "フェーズ5: 可視化",
            }

            for phase_key, phase_data in perf.items():
                if isinstance(phase_data, dict) and "avg_time" in phase_data:
                    phase_display_name = phase_names.get(
                        phase_key, phase_key.replace("phase", "フェーズ ").replace("_", " ")
                    )
                    phase_times[phase_display_name] = phase_data["avg_time"]

            if phase_times:
                total_time = sum(phase_times.values())

                # 総処理時間を表示
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("総処理時間", f"{total_time:.2f}秒")
                with col2:
                    avg_per_frame = total_time / len(phase_times) if phase_times else 0
                    st.metric("平均処理時間", f"{avg_per_frame:.2f}秒/フェーズ")
                with col3:
                    max_phase = max(phase_times.items(), key=lambda x: x[1]) if phase_times else None
                    if max_phase:
                        st.metric("最長フェーズ", max_phase[0].split(":")[0] if ":" in max_phase[0] else max_phase[0])

                st.divider()

                # バーチャートで表示
                st.write("**📊 フェーズ別処理時間（秒）**")
                # 降順でソート
                sorted_phases = sorted(phase_times.items(), key=lambda x: x[1], reverse=True)
                chart_data = dict(sorted_phases)
                st.bar_chart(chart_data, height=350, use_container_width=True)

                # 詳細情報を視覚的に表示
                st.write("**📋 詳細情報**")

                # フェーズごとに視覚的に表示
                for phase_name, phase_time in sorted_phases:
                    percentage = (phase_time / total_time * 100) if total_time > 0 else 0

                    # カード風の表示
                    with st.container():
                        col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
                        with col1:
                            # フェーズ名（短縮版）
                            short_name = phase_name.split(":")[1].strip() if ":" in phase_name else phase_name
                            st.write(f"**{short_name}**")
                        with col2:
                            st.write(f"⏱️ {phase_time:.2f}秒")
                        with col3:
                            st.write(f"📊 {percentage:.1f}%")
                        with col4:
                            # 処理時間に応じたアイコン
                            if percentage > 50:
                                st.write("🔴")
                            elif percentage > 25:
                                st.write("🟡")
                            else:
                                st.write("🟢")

                        # プログレスバー
                        st.progress(min(percentage / 100, 1.0))

                # データテーブル（展開可能）
                with st.expander("📊 詳細データテーブル", expanded=False):
                    table_data = []
                    for phase_name, phase_time in sorted_phases:
                        percentage = (phase_time / total_time * 100) if total_time > 0 else 0
                        table_data.append(
                            {
                                "フェーズ": phase_name,
                                "処理時間（秒）": phase_time,
                                "割合（%）": f"{percentage:.1f}%",
                            }
                        )

                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

    def _render_session_comparison(self) -> None:
        """セッション比較ビューをレンダリング"""
        st.header("📊 セッション比較")
        st.caption("複数セッションのメトリクスを比較・分析できます")

        sessions_dir = Path("output/sessions")
        if not sessions_dir.exists():
            st.error("❌ セッションディレクトリが見つかりません")
            return

        sessions = sorted([d.name for d in sessions_dir.iterdir() if d.is_dir()], reverse=True)

        if not sessions:
            st.warning("⚠️ 比較可能なセッションがありません")
            return

        # セッション数表示
        st.info(f"ℹ️ **{len(sessions)}個**のセッションが見つかりました")

        # セッション選択（複数選択可能）
        selected_sessions = st.multiselect(
            "🔍 比較するセッションを選択（複数選択可）",
            sessions,
            default=sessions[: min(5, len(sessions))] if sessions else [],
            help="比較したいセッションを選択してください（最大10個推奨）",
        )

        if not selected_sessions:
            st.info("ℹ️ 比較するセッションを選択してください")
            return

        # 各セッションのデータを読み込む
        session_data = []
        with st.spinner("📥 セッションデータを読み込み中..."):
            for session_name in selected_sessions:
                session_dir = sessions_dir / session_name
                session_info = self._load_session_info(session_dir)
                if session_info:
                    session_data.append(session_info)

        if not session_data:
            st.warning("⚠️ セッションデータの読み込みに失敗しました")
            return

        # サマリーメトリクス
        st.subheader("📊 比較サマリー")
        col1, col2, col3, col4 = st.columns(4)

        avg_tracks = sum(s.get("track_count", 0) for s in session_data) / len(session_data) if session_data else 0
        avg_visitors = sum(s.get("total_visitors", 0) for s in session_data) / len(session_data) if session_data else 0
        max_concurrent_max = max((s.get("max_concurrent", 0) for s in session_data), default=0)
        avg_id_switches = (
            sum(s.get("estimated_id_switches", 0) for s in session_data) / len(session_data) if session_data else 0
        )

        with col1:
            st.metric("📊 平均トラック数", f"{avg_tracks:.1f}")
        with col2:
            st.metric("👥 平均滞在人数", f"{avg_visitors:.1f}")
        with col3:
            st.metric("👨‍👩‍👧‍👦 最大同時在室", max_concurrent_max)
        with col4:
            st.metric("🔄 平均ID Switch", f"{avg_id_switches:.1f}")

        st.divider()

        # 比較テーブル
        st.subheader("📋 セッション比較表")

        # データフレーム用のデータ準備
        comparison_data = []
        for session in session_data:
            timestamp = session.get("timestamp", "")
            if timestamp:
                # タイムスタンプを読みやすい形式に変換
                try:
                    from datetime import datetime

                    parsed_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    formatted_time = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    formatted_time = timestamp[:19] if len(timestamp) >= 19 else timestamp
            else:
                formatted_time = ""

            row = {
                "セッションID": session["session_id"][-8:],  # 最後の8文字（日時部分）
                "日時": formatted_time,
                "トラック数": session.get("track_count", 0),
                "総滞在人数": session.get("total_visitors", 0),
                "最大同時在室": session.get("max_concurrent", 0),
                "推計ID Switch": session.get("estimated_id_switches", 0),
                "MOTA": f"{session.get('mota', 0):.3f}"
                if isinstance(session.get("mota"), int | float)
                else session.get("mota", "-"),
                "IDF1": f"{session.get('idf1', 0):.3f}"
                if isinstance(session.get("idf1"), int | float)
                else session.get("idf1", "-"),
            }
            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        # メトリクス比較グラフ
        st.divider()
        st.subheader("📈 メトリクス比較")

        # 数値データのみを抽出
        numeric_data = [s for s in session_data if isinstance(s.get("mota"), int | float)]

        if numeric_data:
            # MOTA/IDF1比較
            col1, col2 = st.columns(2)

            with col1:
                st.write("**🎯 MOTA比較**")
                mota_values = [s["mota"] for s in numeric_data]
                session_ids = [s["session_id"][-8:] for s in numeric_data]
                mota_chart = pd.DataFrame({"セッション": session_ids, "MOTA": mota_values})
                st.bar_chart(mota_chart.set_index("セッション"), height=300, use_container_width=True)

                # 目標値との比較
                mota_avg = sum(mota_values) / len(mota_values) if mota_values else 0
                mota_target = 0.7
                delta_mota = mota_avg - mota_target
                st.metric("平均MOTA", f"{mota_avg:.3f}", delta=f"{delta_mota:+.3f}", help=f"目標値: {mota_target}")

            with col2:
                st.write("**🎯 IDF1比較**")
                idf1_values = [s["idf1"] for s in numeric_data]
                idf1_chart = pd.DataFrame({"セッション": session_ids, "IDF1": idf1_values})
                st.bar_chart(idf1_chart.set_index("セッション"), height=300, use_container_width=True)

                # 目標値との比較
                idf1_avg = sum(idf1_values) / len(idf1_values) if idf1_values else 0
                idf1_target = 0.8
                delta_idf1 = idf1_avg - idf1_target
                st.metric("平均IDF1", f"{idf1_avg:.3f}", delta=f"{delta_idf1:+.3f}", help=f"目標値: {idf1_target}")

            # トラック数・滞在人数比較
            st.divider()
            st.write("**📊 トラック数・滞在人数比較**")
            col1, col2 = st.columns(2)

            with col1:
                track_counts = [s.get("track_count", 0) for s in session_data]
                track_chart = pd.DataFrame(
                    {"セッション": [s["session_id"][-8:] for s in session_data], "トラック数": track_counts}
                )
                st.bar_chart(track_chart.set_index("セッション"), height=250, use_container_width=True)

            with col2:
                visitor_counts = [s.get("total_visitors", 0) for s in session_data]
                visitor_chart = pd.DataFrame(
                    {"セッション": [s["session_id"][-8:] for s in session_data], "総滞在人数": visitor_counts}
                )
                st.bar_chart(visitor_chart.set_index("セッション"), height=250, use_container_width=True)

        # パフォーマンス比較
        perf_data = [s for s in session_data if s.get("total_time")]
        if perf_data:
            st.divider()
            st.subheader("⚡ パフォーマンス比較")

            perf_chart_data = pd.DataFrame(
                {
                    "セッション": [s["session_id"][-8:] for s in perf_data],
                    "処理時間（秒）": [s["total_time"] for s in perf_data],
                }
            )
            st.bar_chart(perf_chart_data.set_index("セッション"), height=300, use_container_width=True)

            # パフォーマンス統計
            perf_times = [s["total_time"] for s in perf_data]
            if perf_times:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ 平均処理時間", f"{sum(perf_times) / len(perf_times):.2f}秒")
                with col2:
                    st.metric("⚡ 最短処理時間", f"{min(perf_times):.2f}秒")
                with col3:
                    st.metric("🐌 最長処理時間", f"{max(perf_times):.2f}秒")

    def _load_session_info(self, session_dir: Path) -> dict | None:
        """セッション情報を読み込む

        Args:
            session_dir: セッションディレクトリのパス

        Returns:
            セッション情報の辞書、読み込み失敗時はNone
        """
        try:
            info = {"session_id": session_dir.name}

            # トラックデータ
            tracks_file = session_dir / "phase2.5_tracking" / "tracks.json"
            if not tracks_file.exists():
                tracks_file = session_dir / "tracks.json"

            if tracks_file.exists():
                with open(tracks_file, encoding="utf-8") as f:
                    tracks_data = json.load(f).get("tracks", [])
                    info["track_count"] = len(tracks_data)

                    # セッション集約統計を計算
                    old_session_dir = self.session_dir
                    self.session_dir = session_dir
                    aggregated_stats = self.calculate_session_aggregated_stats(tracks_data)
                    self.session_dir = old_session_dir

                    info["total_visitors"] = aggregated_stats["total_visitors"]
                    info["max_concurrent"] = aggregated_stats["max_concurrent"]
                    info["estimated_id_switches"] = self.estimate_id_switches(tracks_data)

            # サマリーデータ
            summary_file = session_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, encoding="utf-8") as f:
                    summary = json.load(f)
                    info["timestamp"] = summary.get("timestamp", "")

                    # パフォーマンス情報
                    if "performance" in summary:
                        perf = summary["performance"]
                        total_time = sum(
                            v.get("avg_time", 0) for k, v in perf.items() if isinstance(v, dict) and "avg_time" in v
                        )
                        info["total_time"] = total_time

            # MOTメトリクス
            mot_file = session_dir / "mot_metrics.json"
            if mot_file.exists():
                with open(mot_file, encoding="utf-8") as f:
                    mot_result = json.load(f)
                    metrics = mot_result.get("metrics", {})
                    info["mota"] = metrics.get("MOTA", "-")
                    info["idf1"] = metrics.get("IDF1", "-")
            else:
                info["mota"] = "-"
                info["idf1"] = "-"

            return info

        except Exception:
            logger.exception(f"セッション情報の読み込みエラー: {session_dir}")
            return None

    def _draw_tracks(
        self,
        image: np.ndarray,
        tracks_data: list[dict],
        frame_idx: int,
        show_trajectories: bool,
        show_ids: bool,
        selected_ids: list[int],
        selected_zones: list[str],
        max_length: int,
    ) -> np.ndarray:
        """トラックを描画

        Args:
            image: 描画対象の画像
            tracks_data: トラックデータのリスト
            frame_idx: 現在のフレームインデックス
            show_trajectories: 軌跡を表示するか
            show_ids: IDを表示するか
            selected_ids: 表示するIDのリスト
            selected_zones: 表示するゾーンのリスト
            max_length: 軌跡の最大長

        Returns:
            描画された画像
        """
        for track_data in tracks_data:
            track_id = track_data.get("track_id", 0)
            trajectory = track_data.get("trajectory", [])

            # フィルタリング
            if selected_ids and track_id not in selected_ids:
                continue

            if len(trajectory) == 0:
                continue

            # 色を生成
            hue = (track_id * 137) % 180
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(int(c) for c in color_bgr)

            # 軌跡線を描画
            if show_trajectories:
                trajectory_to_draw = trajectory[: min(frame_idx + 1, len(trajectory))]
                if max_length > 0:
                    trajectory_to_draw = trajectory_to_draw[-max_length:]

                for i in range(len(trajectory_to_draw) - 1):
                    pt1 = trajectory_to_draw[i]
                    pt2 = trajectory_to_draw[i + 1]
                    x1, y1 = int(pt1.get("x", 0)), int(pt1.get("y", 0))
                    x2, y2 = int(pt2.get("x", 0)), int(pt2.get("y", 0))
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)

            # 現在位置を描画
            if frame_idx < len(trajectory):
                pt = trajectory[frame_idx]
                x, y = int(pt.get("x", 0)), int(pt.get("y", 0))
                cv2.circle(image, (x, y), 5, color, -1)

                # IDを表示
                if show_ids:
                    cv2.putText(
                        image,
                        f"ID:{track_id}",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

        return image

    def _convert_to_tracks(self, tracks_data: list[dict]) -> list[Track]:
        """トラックデータをTrackオブジェクトに変換

        Args:
            tracks_data: トラックデータのリスト

        Returns:
            Trackオブジェクトのリスト
        """
        tracks = []
        for track_data in tracks_data:
            # Detectionオブジェクトを作成（簡易版）
            detection = Detection(
                bbox=(0, 0, 0, 0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(0, 0),
            )

            # Kalman Filterを作成
            kf = KalmanFilter()

            # Trackオブジェクトを作成
            track = Track(
                track_id=track_data.get("track_id", 0),
                detection=detection,
                kalman_filter=kf,
            )

            # 軌跡を設定
            trajectory = track_data.get("trajectory", [])
            track.trajectory = [(pt["x"], pt["y"]) for pt in trajectory]
            track.age = track_data.get("age", 1)
            track.hits = track_data.get("hits", 1)

            tracks.append(track)

        return tracks


def main():
    """メイン関数（Streamlitアプリ）"""
    setup_logging(debug_mode=False)

    # セッション状態の初期化
    if "session_dir" not in st.session_state:
        sessions_dir = Path("output/sessions")
        if sessions_dir.exists():
            sessions = [d.name for d in sessions_dir.iterdir() if d.is_dir()]
            if sessions:
                st.session_state.session_dir = str(sessions_dir / sessions[-1])
            else:
                st.error("セッションディレクトリが見つかりません")
                return
        else:
            st.error("output/sessionsディレクトリが見つかりません")
            return

    try:
        visualizer = InteractiveVisualizer(st.session_state.session_dir)
        visualizer.render_app()
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        logger.exception("Visualization error")


if __name__ == "__main__":
    main()
