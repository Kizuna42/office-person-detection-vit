"""メインエディタクラス: Ground Truthトラック編集ツール"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import ConfigManager
from src.transform import FloorMapConfig, HomographyTransformer
from tools.gt_editor.data_loader import (
    FrameImageLoader,
    SessionTrackLoader,
    TrackDataLoader,
    TrackGenerator,
)
from tools.gt_editor.renderer import CameraRenderer, FloormapRenderer
from tools.gt_editor.track_manager import TrackManager
from tools.gt_editor.ui_controller import UIController

logger = logging.getLogger(__name__)


class GTracksEditor:
    """Ground Truthトラック編集クラス"""

    def __init__(
        self,
        tracks_path: Path,
        floormap_path: Path,
        config_path: Path,
        session_dir: Path | None = None,
    ):
        """初期化

        Args:
            tracks_path: Ground Truthトラックファイルのパス
            floormap_path: フロアマップ画像のパス
            config_path: 設定ファイルのパス
            session_dir: セッションディレクトリ（カメラ画像表示用、オプション）
        """
        self.tracks_path = tracks_path
        self.floormap_path = floormap_path
        self.config_path = config_path
        self.session_dir = session_dir

        # データローダーを初期化
        self.track_loader = TrackDataLoader(tracks_path)
        self.tracks_data = self.track_loader.load()

        # フロアマップ画像を読み込む
        self.floormap_image = self._load_floormap()
        self.config = ConfigManager(str(config_path))

        # セッション関連のデータ
        self.session_tracks: dict[int, dict] = {}
        self.frame_images: dict[int, np.ndarray] = {}
        self.detection_images: dict[int, np.ndarray] = {}
        self.detection_results: dict[int, list[dict]] = {}
        self.frame_index_to_gt_frame: dict[int, int] = {}
        self.gt_frame_to_frame_index: dict[int, int] = {}
        self.coordinate_transformer: HomographyTransformer | None = None

        # セッションデータを読み込む
        if session_dir:
            self._load_session_data(session_dir)

        # トラックマネージャーを初期化
        self.track_manager = TrackManager(self.tracks_data, self.session_tracks)

        # 編集状態
        self.current_frame = 0
        self.selected_track_id: int | None = None
        self.selected_point_idx: int | None = None
        self.dragging = False
        self.max_frame = self.track_manager.get_max_frame()

        # レンダラーを初期化
        self.floormap_renderer = FloormapRenderer(
            self.floormap_image,
            self.tracks_data,
            self.session_tracks,
        )

        self.camera_renderer: CameraRenderer | None = None
        if session_dir:
            self.camera_renderer = CameraRenderer(
                self.frame_images,
                self.detection_images,
                self.detection_results,
                self.session_tracks,
                self.gt_frame_to_frame_index,
            )

        # UIコントローラー
        self.ui_controller = UIController(
            "Ground Truth Tracks Editor - Floormap",
            self._on_mouse,
        )

        logger.info(f"トラック数: {len(self.tracks_data)}")
        logger.info(f"最大フレーム数: {self.max_frame}")
        if session_dir:
            logger.info(f"カメラ画像: {len(self.frame_images)}フレーム読み込み済み")

    def _load_floormap(self) -> np.ndarray:
        """フロアマップ画像を読み込む"""
        image = cv2.imread(str(self.floormap_path))
        if image is None:
            raise FileNotFoundError(f"フロアマップ画像を読み込めません: {self.floormap_path}")
        return image

    def _load_session_data(self, session_dir: Path) -> None:
        """セッションデータを読み込む"""
        # セッショントラックを読み込む
        session_loader = SessionTrackLoader(session_dir)
        self.session_tracks = session_loader.load()

        # フレーム画像とマッピングを読み込む
        image_loader = FrameImageLoader(session_dir)
        image_loader.load()
        self.frame_images = image_loader.frame_images
        self.detection_images = image_loader.detection_images
        self.detection_results = image_loader.detection_results
        self.frame_index_to_gt_frame = image_loader.frame_index_to_gt_frame
        self.gt_frame_to_frame_index = image_loader.gt_frame_to_frame_index

        # 座標変換器を初期化
        homography_matrix = self.config.get("homography.matrix")
        floormap_config_dict = self.config.get("floormap", {})
        if homography_matrix:
            import numpy as np

            H = np.array(homography_matrix, dtype=np.float64)
            fm_config = FloorMapConfig.from_config(floormap_config_dict)
            self.coordinate_transformer = HomographyTransformer(H, fm_config)

        # Ground Truthトラックが空の場合のみ、セッショントラックから初期データを生成
        if len(self.tracks_data) == 0 and len(self.session_tracks) > 0:
            logger.info("Ground Truthトラックが空のため、セッショントラックから初期データを生成します")
            generator = TrackGenerator(
                self.session_tracks,
                self.gt_frame_to_frame_index,
                self.coordinate_transformer,
            )
            generated_tracks = generator.generate()
            self.tracks_data.extend(generated_tracks)
            # トラックマネージャーを再初期化
            self.track_manager = TrackManager(self.tracks_data, self.session_tracks)
        elif len(self.tracks_data) > 0:
            logger.info("既存のGround Truthトラックを読み込みました。編集を続行します。")

    def _on_mouse(self, event, x, y, flags, param) -> None:
        """マウスイベントハンドラ"""
        if event == cv2.EVENT_LBUTTONDOWN:
            nearest = self.track_manager.find_nearest_point(
                x,
                y,
                self.current_frame,
                self.floormap_image.shape[1],
                self.floormap_image.shape[0],
            )
            if nearest is not None:
                self.selected_track_id, self.selected_point_idx = nearest
                self.dragging = True
                logger.info(f"トラックID {self.selected_track_id} を選択しました")
            else:
                self.selected_track_id = None
                self.selected_point_idx = None

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.selected_track_id is not None:
                self.track_manager.update_point(self.selected_track_id, self.current_frame, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _match_track_id_from_camera(self) -> None:
        """カメラ画像上の検出結果IDに合わせてフロアマップ上のトラックIDを変更"""
        if self.selected_track_id is None:
            logger.warning("トラックが選択されていません")
            return

        track = self.track_manager.get_track_by_id(self.selected_track_id)
        if track is None:
            logger.warning("選択されたトラックが見つかりません")
            return

        trajectory = track.get("trajectory", [])
        current_point = None
        for point in trajectory:
            if point.get("frame") == self.current_frame:
                current_point = point
                break

        if current_point is None:
            logger.warning("現在フレームにポイントがありません")
            return

        floor_x = current_point.get("x", 0)
        floor_y = current_point.get("y", 0)

        frame_idx = self.gt_frame_to_frame_index.get(self.current_frame, -1)
        if frame_idx < 0 or frame_idx not in self.detection_results:
            logger.warning("現在フレームの検出結果が見つかりません")
            return

        detections = self.detection_results[frame_idx]
        if not detections:
            logger.warning("検出結果が空です")
            return

        # 最も近い検出結果を探す
        from tools.gt_editor.utils import calculate_distance

        min_distance = float("inf")
        nearest_detection = None

        for det in detections:
            floor_coords_data = det.get("floor_coords")
            if floor_coords_data is None:
                if not self.coordinate_transformer:
                    continue
                camera_coords_data = det.get("camera_coords", {})
                camera_x = camera_coords_data.get("x", 0)
                camera_y = camera_coords_data.get("y", 0)
                try:
                    result = self.coordinate_transformer.transform_pixel((camera_x, camera_y))
                    if result.is_valid and result.floor_coords_px:
                        floor_coords = result.floor_coords_px
                    else:
                        continue
                except Exception as e:
                    logger.debug(f"座標変換エラー: {e}")
                    continue
            else:
                floor_coords = (floor_coords_data.get("x", 0), floor_coords_data.get("y", 0))

            distance = calculate_distance(floor_coords[0], floor_coords[1], floor_x, floor_y)

            if distance < min_distance:
                min_distance = distance
                nearest_detection = det

        if nearest_detection is None:
            logger.warning("対応する検出結果が見つかりませんでした")
            return

        camera_track_id = nearest_detection.get("track_id")
        if camera_track_id is None:
            logger.warning("検出結果にtrack_idがありません")
            return

        threshold = 100.0
        if min_distance > threshold:
            logger.warning(
                f"検出結果との距離が遠すぎます: {min_distance:.2f}px (閾値: {threshold}px)。ID変更をスキップします。"
            )
            return

        logger.info(
            f"カメラ画像上の検出結果ID {camera_track_id} に合わせて、"
            f"フロアマップ上のトラックIDを {self.selected_track_id} から {camera_track_id} に変更します"
            f"（距離: {min_distance:.2f}px）"
        )
        if self.track_manager.change_track_id(self.selected_track_id, camera_track_id):
            self.selected_track_id = camera_track_id

    def run(self) -> None:
        """編集ツールを実行"""
        self.ui_controller.setup()

        # カメラ画像ウィンドウも作成
        if self.session_dir:
            cv2.namedWindow("Ground Truth Tracks Editor - Camera", cv2.WINDOW_NORMAL)

        logger.info("=" * 80)
        logger.info("Ground Truthトラック編集ツール")
        logger.info("=" * 80)
        logger.info("操作説明:")
        logger.info("  左矢印/右矢印: フレーム移動")
        logger.info("  クリック: トラック選択")
        logger.info("  ドラッグ: ポイント移動")
        logger.info("  1-9: トラックID変更（クイック）")
        logger.info("  i: ID入力モード（1-30）")
        logger.info("  m: カメラ画像のIDに合わせる（同一人物検出修正）")
        logger.info("  d: ポイント削除")
        logger.info("  a: 新しいトラック追加")
        logger.info("  s: 保存")
        logger.info("  q: 終了")
        logger.info("=" * 80)

        try:
            while True:
                # フロアマップを描画
                id_input_mode, id_input_buffer = self.ui_controller.get_id_input_state()
                floormap_image = self.floormap_renderer.render(
                    self.current_frame,
                    self.max_frame,
                    self.selected_track_id,
                    id_input_mode,
                    id_input_buffer,
                )
                cv2.imshow(self.ui_controller.window_name, floormap_image)

                # カメラ画像を描画
                if self.camera_renderer:
                    camera_image = self.camera_renderer.render(self.current_frame, self.max_frame)
                    if camera_image is not None:
                        cv2.imshow("Ground Truth Tracks Editor - Camera", camera_image)

                # キー入力待ち
                key = cv2.waitKey(30)

                # キー入力を処理
                should_quit = self.ui_controller.process_key(
                    key,
                    on_quit=lambda: None,
                    on_save=self._save_tracks,
                    on_add_track=self._add_new_track,
                    on_delete_point=self._delete_point,
                    on_match_id=self._match_track_id_from_camera,
                    on_change_track_id=self._change_track_id,
                    on_start_id_input=self._start_id_input,
                    on_frame_change=self._change_frame,
                )

                if should_quit:
                    break

        except KeyboardInterrupt:
            logger.info("ユーザーにより中断されました")
        finally:
            cv2.destroyAllWindows()

    def _save_tracks(self) -> None:
        """トラックデータを保存"""
        self.track_loader.save({"num_frames": self.max_frame + 1})

    def _add_new_track(self) -> None:
        """新しいトラックを追加"""
        center_x = self.floormap_image.shape[1] // 2
        center_y = self.floormap_image.shape[0] // 2
        new_id = self.track_manager.add_new_track(self.current_frame, center_x, center_y)
        self.selected_track_id = new_id
        self.selected_point_idx = 0

    def _delete_point(self) -> None:
        """選択されたポイントを削除"""
        if self.selected_track_id is None or self.selected_point_idx is None:
            logger.warning("ポイントが選択されていません")
            return

        if self.track_manager.delete_point(
            self.selected_track_id,
            self.selected_point_idx,
            self.current_frame,
        ):
            # トラックが削除された場合
            self.selected_track_id = None
            self.selected_point_idx = None

    def _change_track_id(self, new_id: int) -> None:
        """トラックIDを変更"""
        if self.selected_track_id is None:
            logger.warning("トラックが選択されていません")
            return

        if self.track_manager.change_track_id(self.selected_track_id, new_id):
            self.selected_track_id = new_id

    def _start_id_input(self) -> None:
        """ID入力モードを開始"""
        if self.selected_track_id is not None:
            self.ui_controller.id_input_mode = True
            self.ui_controller.id_input_buffer = ""
            logger.info("ID入力モード: 1-30の数字を入力してください（Enterで確定、ESCでキャンセル）")
        else:
            logger.warning("トラックが選択されていません")

    def _change_frame(self, delta: int) -> None:
        """フレームを変更

        Args:
            delta: フレーム変更量（-1 or 1）
        """
        self.current_frame = max(0, min(self.max_frame, self.current_frame + delta))
