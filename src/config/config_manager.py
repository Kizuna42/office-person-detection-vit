"""Configuration management module for the office person detection system."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """設定ファイル管理クラス

    YAML/JSON形式の設定ファイルを読み込み、検証し、設定値を提供する。

    Attributes:
        config_path: 設定ファイルのパス
        config: 読み込まれた設定データ
    """

    # 必須項目の定義
    REQUIRED_KEYS = {
        "video": ["input_path"],
        "detection": ["model_name", "confidence_threshold", "device"],
        "floormap": [
            "image_path",
            "image_width",
            "image_height",
            "image_origin_x",
            "image_origin_y",
            "image_x_mm_per_pixel",
            "image_y_mm_per_pixel",
        ],
        "homography": ["matrix"],
        "zones": [],
        "output": ["directory"],
    }

    # デフォルト設定値
    DEFAULT_CONFIG = {
        "video": {
            "input_path": "input/merged_moviefiles.mov",
            "is_timelapse": True,
            "frame_interval_minutes": 5,
            "tolerance_seconds": 10,
        },
        "detection": {
            "model_name": "facebook/detr-resnet-50",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "patch_size": 16,
            "device": "mps",
            "batch_size": 4,
        },
        "floormap": {
            "image_path": "data/floormap.png",
            "image_width": 1878,
            "image_height": 1369,
            "image_origin_x": 7,
            "image_origin_y": 9,
            "image_x_mm_per_pixel": 28.1926406926406,
            "image_y_mm_per_pixel": 28.241430700447,
        },
        "homography": {"matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
        "zones": [],
        "output": {
            "directory": "output",
            "save_detection_images": True,
            "save_floormap_images": True,
            "debug_mode": False,
        },
        "evaluation": {
            "ground_truth_path": "output/labels/result_fixed.json",
            "iou_threshold": 0.5,
        },
        "fine_tuning": {
            "enabled": False,
            "dataset_path": "data/office_dataset",
            "epochs": 50,
            "learning_rate": 0.0001,
            "batch_size": 8,
            "warmup_epochs": 5,
            "layer_decay": 0.65,
        },
    }

    def __init__(self, config_path: str = "config.yaml"):
        """ConfigManagerを初期化する

        Args:
            config_path: 設定ファイルのパス（デフォルト: config.yaml）
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む

        Returns:
            読み込まれた設定データ

        Raises:
            FileNotFoundError: 設定ファイルが存在しない場合
            ValueError: 設定ファイルの形式が不正な場合
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"設定ファイル '{self.config_path}' が見つかりません。デフォルト設定を使用します。")
            return self.DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                file_ext = Path(self.config_path).suffix.lower()

                if file_ext in [".yaml", ".yml"]:
                    config = yaml.safe_load(f)
                elif file_ext == ".json":
                    config = json.load(f)
                else:
                    raise ValueError(f"サポートされていないファイル形式: {file_ext}")

                if config is None:
                    logger.warning("設定ファイルが空です。デフォルト設定を使用します。")
                    return self.DEFAULT_CONFIG.copy()

                logger.info(f"設定ファイル '{self.config_path}' を読み込みました。")
                return config

        except yaml.YAMLError as e:
            raise ValueError(f"YAML解析エラー: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析エラー: {e}")
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")

    def validate(self) -> bool:
        """設定値の妥当性を検証する

        必須項目の存在チェック、型チェック、値の範囲チェックを実行する。

        Returns:
            検証が成功した場合True、失敗した場合False

        Raises:
            ValueError: 設定値が不正な場合
        """
        # 必須セクションの存在チェック
        for section in self.REQUIRED_KEYS.keys():
            if section not in self.config:
                raise ValueError(f"必須セクション '{section}' が設定ファイルに存在しません。")

        # 各セクションの必須項目チェック
        for section, required_keys in self.REQUIRED_KEYS.items():
            # zonesセクションはリスト型なので、辞書型チェックをスキップ
            if section == "zones":
                continue

            section_config = self.config.get(section, {})
            if not isinstance(section_config, dict):
                raise ValueError(f"セクション '{section}' は辞書型である必要があります。")
            if not isinstance(required_keys, (list, tuple)):
                raise ValueError(f"必須キー '{section}' はリストまたはタプルである必要があります。")
            for key in required_keys:
                if key not in section_config:
                    raise ValueError(f"必須項目 '{section}.{key}' が設定ファイルに存在しません。")

        # video セクションの検証
        self._validate_video_config()

        # detection セクションの検証
        self._validate_detection_config()

        # floormap セクションの検証
        self._validate_floormap_config()

        # homography セクションの検証
        self._validate_homography_config()

        # zones セクションの検証
        self._validate_zones_config()

        # camera セクションの検証（オプション）
        if "camera" in self.config:
            self._validate_camera_config()

        # output セクションの検証
        self._validate_output_config()

        logger.info("設定ファイルの検証が完了しました。")
        return True

    def _validate_video_config(self):
        """video セクションの検証"""
        video_config = self.config.get("video", {})

        # input_path の型チェック
        if not isinstance(video_config.get("input_path"), str):
            raise ValueError("video.input_path は文字列である必要があります。")

        # frame_interval_minutes の検証
        if "frame_interval_minutes" in video_config:
            interval = video_config["frame_interval_minutes"]
            if not isinstance(interval, (int, float)) or interval <= 0:
                raise ValueError("video.frame_interval_minutes は正の数値である必要があります。")

        # tolerance_seconds の検証
        if "tolerance_seconds" in video_config:
            tolerance = video_config["tolerance_seconds"]
            if not isinstance(tolerance, (int, float)) or tolerance < 0:
                raise ValueError("video.tolerance_seconds は非負の数値である必要があります。")

    def _validate_detection_config(self):
        """detection セクションの検証"""
        detection_config = self.config.get("detection", {})

        # model_name の型チェック
        if not isinstance(detection_config.get("model_name"), str):
            raise ValueError("detection.model_name は文字列である必要があります。")

        # confidence_threshold の検証
        confidence = detection_config.get("confidence_threshold")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError(
                "detection.confidence_threshold は 0.0 から 1.0 の範囲である必要があります。"
            )

        # nms_threshold の検証
        if "nms_threshold" in detection_config:
            nms = detection_config["nms_threshold"]
            if not isinstance(nms, (int, float)) or not (0.0 <= nms <= 1.0):
                raise ValueError("detection.nms_threshold は 0.0 から 1.0 の範囲である必要があります。")

        # device の検証
        device = detection_config.get("device")
        if not isinstance(device, str) or device not in ["mps", "cuda", "cpu"]:
            raise ValueError("detection.device は 'mps', 'cuda', 'cpu' のいずれかである必要があります。")

        # batch_size の検証
        if "batch_size" in detection_config:
            batch_size = detection_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("detection.batch_size は正の整数である必要があります。")

    def _validate_floormap_config(self):
        """floormap セクションの検証"""
        floormap_config = self.config.get("floormap", {})

        # image_path の型チェック
        if not isinstance(floormap_config.get("image_path"), str):
            raise ValueError("floormap.image_path は文字列である必要があります。")

        # 画像サイズの検証
        width = floormap_config.get("image_width")
        if not isinstance(width, int) or width <= 0:
            raise ValueError("floormap.image_width は正の整数である必要があります。")

        height = floormap_config.get("image_height")
        if not isinstance(height, int) or height <= 0:
            raise ValueError("floormap.image_height は正の整数である必要があります。")

        # 原点オフセットの検証
        origin_x = floormap_config.get("image_origin_x")
        if not isinstance(origin_x, (int, float)) or origin_x < 0:
            raise ValueError("floormap.image_origin_x は非負の数値である必要があります。")

        origin_y = floormap_config.get("image_origin_y")
        if not isinstance(origin_y, (int, float)) or origin_y < 0:
            raise ValueError("floormap.image_origin_y は非負の数値である必要があります。")

        # スケールの検証
        x_scale = floormap_config.get("image_x_mm_per_pixel")
        if not isinstance(x_scale, (int, float)) or x_scale <= 0:
            raise ValueError("floormap.image_x_mm_per_pixel は正の数値である必要があります。")

        y_scale = floormap_config.get("image_y_mm_per_pixel")
        if not isinstance(y_scale, (int, float)) or y_scale <= 0:
            raise ValueError("floormap.image_y_mm_per_pixel は正の数値である必要があります。")

    def _validate_homography_config(self):
        """homography セクションの検証"""
        homography_config = self.config.get("homography", {})

        # matrix の検証
        matrix = homography_config.get("matrix")
        if not isinstance(matrix, list):
            raise ValueError("homography.matrix はリストである必要があります。")

        if len(matrix) != 3:
            raise ValueError("homography.matrix は 3x3 行列である必要があります。")

        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"homography.matrix の行 {i} は長さ3のリストである必要があります。")

            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(f"homography.matrix[{i}][{j}] は数値である必要があります。")

    def _validate_zones_config(self):
        """zones セクションの検証"""
        zones = self.config.get("zones", [])

        if not isinstance(zones, list):
            raise ValueError("zones はリストである必要があります。")

        for i, zone in enumerate(zones):
            if not isinstance(zone, dict):
                raise ValueError(f"zones[{i}] は辞書である必要があります。")

            # id の検証
            if "id" not in zone:
                raise ValueError(f"zones[{i}] には 'id' が必要です。")
            if not isinstance(zone["id"], str):
                raise ValueError(f"zones[{i}].id は文字列である必要があります。")

            # polygon の検証
            if "polygon" not in zone:
                raise ValueError(f"zones[{i}] には 'polygon' が必要です。")

            polygon = zone["polygon"]
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(f"zones[{i}].polygon は少なくとも3つの頂点を持つリストである必要があります。")

            for j, point in enumerate(polygon):
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f"zones[{i}].polygon[{j}] は [x, y] 形式である必要があります。")

                if not all(isinstance(coord, (int, float)) for coord in point):
                    raise ValueError(f"zones[{i}].polygon[{j}] の座標は数値である必要があります。")

            # priority の検証（任意）
            if "priority" in zone and zone["priority"] is not None:
                priority = zone["priority"]
                if not isinstance(priority, (int, float)):
                    raise ValueError(f"zones[{i}].priority は数値である必要があります。")

    def _validate_camera_config(self):
        """camera セクションの検証"""
        camera_config = self.config.get("camera", {})

        # position_x の検証
        if "position_x" in camera_config:
            pos_x = camera_config["position_x"]
            if not isinstance(pos_x, (int, float)) or pos_x < 0:
                raise ValueError("camera.position_x は非負の数値である必要があります。")

        # position_y の検証
        if "position_y" in camera_config:
            pos_y = camera_config["position_y"]
            if not isinstance(pos_y, (int, float)) or pos_y < 0:
                raise ValueError("camera.position_y は非負の数値である必要があります。")

        # height_m の検証
        if "height_m" in camera_config:
            height = camera_config["height_m"]
            if not isinstance(height, (int, float)) or height <= 0:
                raise ValueError("camera.height_m は正の数値である必要があります。")

        # show_on_floormap の検証
        if "show_on_floormap" in camera_config:
            if not isinstance(camera_config["show_on_floormap"], bool):
                raise ValueError("camera.show_on_floormap はブール値である必要があります。")

        # marker_color の検証
        if "marker_color" in camera_config:
            marker_color = camera_config["marker_color"]
            if not isinstance(marker_color, list) or len(marker_color) != 3:
                raise ValueError("camera.marker_color は [B, G, R] 形式の3要素リストである必要があります。")
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in marker_color):
                raise ValueError("camera.marker_color の各要素は0-255の整数である必要があります。")

        # marker_size の検証
        if "marker_size" in camera_config:
            marker_size = camera_config["marker_size"]
            if not isinstance(marker_size, int) or marker_size <= 0:
                raise ValueError("camera.marker_size は正の整数である必要があります。")

    def _validate_output_config(self):
        """output セクションの検証"""
        output_config = self.config.get("output", {})

        # directory の型チェック
        if not isinstance(output_config.get("directory"), str):
            raise ValueError("output.directory は文字列である必要があります。")

        # ブール値フィールドの検証
        bool_fields = ["save_detection_images", "save_floormap_images", "debug_mode"]
        for field in bool_fields:
            if field in output_config and not isinstance(output_config[field], bool):
                raise ValueError(f"output.{field} はブール値である必要があります。")

    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得する

        ドット記法（例: 'video.input_path'）で階層的な設定値にアクセスできる。

        Args:
            key: 設定キー（ドット記法をサポート）
            default: キーが存在しない場合のデフォルト値

        Returns:
            設定値、またはデフォルト値
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """設定セクション全体を取得する

        Args:
            section: セクション名（例: 'video', 'detection'）

        Returns:
            セクションの設定データ
        """
        return self.config.get(section, {})

    def set(self, key: str, value: Any):
        """設定値を動的に変更する

        Args:
            key: 設定キー（ドット記法をサポート）
            value: 設定する値
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logger.debug(f"設定値を変更しました: {key} = {value}")

    def save(self, output_path: Optional[str] = None):
        """設定をファイルに保存する

        Args:
            output_path: 保存先パス（指定しない場合は元のパスに上書き）
        """
        save_path = output_path or self.config_path
        file_ext = Path(save_path).suffix.lower()

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                if file_ext in [".yaml", ".yml"]:
                    yaml.dump(
                        self.config, f, default_flow_style=False, allow_unicode=True
                    )
                elif file_ext == ".json":
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"サポートされていないファイル形式: {file_ext}")

            logger.info(f"設定ファイルを保存しました: {save_path}")
        except Exception as e:
            logger.error(f"設定ファイルの保存に失敗しました: {e}")
            raise
