"""Streamlit-based interactive visualization tool for tracking results."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.config.config_manager import ConfigManager
from src.evaluation.mot_metrics import MOTMetrics
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
        tracks_file = self.session_dir / "tracks.json"
        if not tracks_file.exists():
            return []

        with open(tracks_file, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("tracks", [])

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

    def render_app(self) -> None:
        """Streamlitアプリをレンダリング"""
        st.set_page_config(page_title="Tracking Visualization", layout="wide")

        st.title("オブジェクト追跡可視化ツール")

        # サイドバー: コントロール
        with st.sidebar:
            st.header("設定")

            # セッション選択
            sessions_dir = Path("output/sessions")
            if sessions_dir.exists():
                sessions = [d.name for d in sessions_dir.iterdir() if d.is_dir()]
                selected_session = st.selectbox("セッションを選択", sessions)
                if selected_session:
                    self.session_dir = sessions_dir / selected_session
                    st.session_state.session_dir = str(self.session_dir)

            # フィルタ設定
            st.subheader("フィルタ")
            show_trajectories = st.checkbox("軌跡を表示", value=True)
            show_ids = st.checkbox("IDを表示", value=True)

            # IDフィルタ
            tracks_data = self.load_tracks_data()
            if tracks_data:
                track_ids = [track.get("track_id", 0) for track in tracks_data]
                selected_ids = st.multiselect("表示するIDを選択", track_ids, default=track_ids)

            # ゾーンフィルタ
            zones = self.config.get("zones", [])
            zone_ids = [zone.get("id", "") for zone in zones]
            selected_zones = st.multiselect("表示するゾーンを選択", zone_ids, default=zone_ids)

            # 軌跡の長さ制限
            max_trajectory_length = st.slider("軌跡の最大長", 10, 100, 50)

        # メインエリア: 可視化
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("フロアマップ可視化")

            # フレーム選択スライダー
            if tracks_data:
                max_frames = max((len(track.get("trajectory", [])) for track in tracks_data), default=0)
                if max_frames > 0:
                    frame_idx = st.slider("フレーム", 0, max_frames - 1, 0)

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

        with col2:
            st.subheader("統計情報")

            if tracks_data:
                st.metric("トラック数", len(tracks_data))
                total_points = sum(len(track.get("trajectory", [])) for track in tracks_data)
                st.metric("総軌跡点数", total_points)
                avg_trajectory_length = total_points / len(tracks_data) if tracks_data else 0
                st.metric("平均軌跡長", f"{avg_trajectory_length:.1f}")

                # MOTメトリクス（簡易版）
                st.subheader("MOTメトリクス")
                st.info(
                    "MOTメトリクスはGround Truthデータが必要です。詳細な評価は`scripts/evaluate_mot_metrics.py`を使用してください。"
                )

                # トラック情報テーブル
                st.subheader("トラック情報")
                track_info = []
                for track in tracks_data[:10]:  # 最初の10件のみ表示
                    track_info.append(
                        {
                            "ID": track.get("track_id", 0),
                            "軌跡長": len(track.get("trajectory", [])),
                            "年齢": track.get("age", 0),
                            "ヒット数": track.get("hits", 0),
                        }
                    )
                if track_info:
                    st.dataframe(track_info)

            # エクスポート機能
            st.subheader("エクスポート")
            export_format = st.selectbox("エクスポート形式", ["CSV", "JSON", "画像シーケンス", "動画"])

            if st.button("エクスポート実行"):
                try:
                    exporter = TrajectoryExporter(self.session_dir / "exports")
                    tracks = self._convert_to_tracks(tracks_data)

                    if export_format == "CSV":
                        output_path = exporter.export_csv(tracks, filename="exported_tracks.csv")
                        st.success(f"CSVエクスポート完了: {output_path}")
                        with open(output_path, "rb") as f:
                            st.download_button("CSVをダウンロード", f.read(), "tracks.csv", "text/csv")

                    elif export_format == "JSON":
                        output_path = exporter.export_json(tracks, filename="exported_tracks.json")
                        st.success(f"JSONエクスポート完了: {output_path}")
                        with open(output_path, "rb") as f:
                            st.download_button("JSONをダウンロード", f.read(), "tracks.json", "application/json")

                    elif export_format == "画像シーケンス":
                        floormap = self.load_floormap()
                        output_paths = exporter.export_image_sequence(tracks, floormap)
                        st.success(f"画像シーケンスエクスポート完了: {len(output_paths)}フレーム")
                        st.info(f"出力先: {output_paths[0].parent}")

                    elif export_format == "動画":
                        floormap = self.load_floormap()
                        output_path = exporter.export_video(tracks, floormap, filename="exported_trajectories.mp4")
                        st.success(f"動画エクスポート完了: {output_path}")
                        with open(output_path, "rb") as f:
                            st.download_button("動画をダウンロード", f.read(), "trajectories.mp4", "video/mp4")

                except Exception as e:
                    st.error(f"エクスポートエラー: {e}")
                    logger.exception("Export error")

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
