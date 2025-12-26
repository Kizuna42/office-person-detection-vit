"""YOLOv8 based person detection module."""

import logging
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from src.models.data_models import Detection
from src.tracking.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# デフォルトのFine-tunedモデルパス
DEFAULT_MODEL_PATH = "runs/detect/person_ft/weights/best.pt"


class YOLOv8Detector:
    """YOLOv8人物検出クラス.

    ViTDetectorと同じインターフェースを提供し、
    パイプラインでの置き換えが可能。
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.25,
        device: str | None = None,
        iou_threshold: float = 0.45,
    ):
        """YOLOv8Detectorを初期化.

        Args:
            model_path: モデルファイルのパス (.pt)
            confidence_threshold: 検出の信頼度閾値 (0.0-1.0)
            device: 使用するデバイス ("mps", "cuda", "cpu", None=自動検出)
            iou_threshold: NMSのIoU閾値
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._setup_device(device)
        self.model: YOLO | None = None
        self.feature_extractor = FeatureExtractor()

        logger.info(f"YOLOv8Detector initialized with model: {model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Confidence threshold: {confidence_threshold}")

    def _setup_device(self, device: str | None = None) -> str:
        """デバイスを設定.

        Args:
            device: 指定されたデバイス名 (None の場合は自動検出)

        Returns:
            使用するデバイス名
        """
        if device is not None:
            return device

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """YOLOv8モデルをロード.

        Raises:
            RuntimeError: モデルのロードに失敗した場合
        """
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                # Fine-tunedモデルがなければベースモデルを使用
                logger.warning(f"Model not found at {model_path}, using base yolov8x.pt")
                self.model = YOLO("yolov8x.pt")
            else:
                self.model = YOLO(str(model_path))

            logger.info(f"Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load YOLOv8 model: {e}") from e

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """人物検出を実行.

        Args:
            frame: 入力画像 (numpy array, BGR format)

        Returns:
            検出結果のリスト

        Raises:
            RuntimeError: モデルがロードされていない場合
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # YOLOv8で推論
            # Fine-tunedモデル（single_cls=True）の場合はclassesフィルタを使用しない
            # ベースモデル（yolov8x.pt等）の場合はclasses=[0]でpersonのみフィルタ
            is_finetuned = "person_ft" in self.model_path or "best.pt" in self.model_path

            predict_kwargs = {
                "conf": self.confidence_threshold,
                "iou": self.iou_threshold,
                "verbose": False,
                "device": self.device,
            }

            # ベースモデルの場合のみクラスフィルタを適用
            if not is_finetuned:
                predict_kwargs["classes"] = [0]  # person only (COCO)

            results = self.model(frame, **predict_kwargs)

            # 検出結果をDetectionオブジェクトに変換
            detections = self._postprocess(results, frame.shape)

            logger.debug(f"Detected {len(detections)} persons")
            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

    def detect_with_features(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """人物検出と特徴量抽出を同時に実行.

        Args:
            frame: 入力画像 (numpy array, BGR format)

        Returns:
            (検出結果のリスト, 特徴量配列)
            特徴量配列の形状: (num_detections, feature_dim)

        Raises:
            RuntimeError: モデルがロードされていない場合
        """
        # まず検出を実行
        detections = self.detect(frame)

        # 各検出領域から特徴量を抽出
        features = self.extract_features(frame, detections)

        # 検出結果へ対応する特徴量をアサイン
        for i, det in enumerate(detections):
            if i < len(features):
                det.features = features[i]

        logger.debug(f"Detected {len(detections)} persons with features")
        return detections, features

    def extract_features(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """検出結果に対応する特徴量を抽出.

        Args:
            frame: 入力画像 (numpy array, BGR format)
            detections: 検出結果のリスト

        Returns:
            特徴量配列 (num_detections, feature_dim)
        """
        if len(detections) == 0:
            return np.array([])

        # FeatureExtractorを使用してRe-ID特徴量を抽出
        crops = []
        for det in detections:
            x, y, w, h = det.bbox
            x1, y1 = int(max(0, x)), int(max(0, y))
            x2, y2 = int(min(frame.shape[1], x + w)), int(min(frame.shape[0], y + h))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                # 無効な領域の場合はダミーを追加
                crops.append(np.zeros((64, 32, 3), dtype=np.uint8))

        # 特徴量抽出
        features = self.feature_extractor.extract_batch(crops)

        return features

    def _postprocess(self, results, image_shape: tuple[int, ...]) -> list[Detection]:
        """YOLOv8の出力を検出結果に変換.

        Args:
            results: YOLOv8の出力
            image_shape: 元画像のshape (height, width, channels)

        Returns:
            検出結果のリスト
        """
        detections = []
        _height, _width = image_shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # xyxy形式からxywh形式に変換
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])

                bbox = (x1, y1, x2 - x1, y2 - y1)
                camera_coords = self._get_foot_position(bbox)

                detection = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=0,  # person
                    class_name="person",
                    camera_coords=camera_coords,
                )
                detections.append(detection)

        return detections

    def _get_foot_position(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """バウンディングボックスから足元座標を計算.

        Args:
            bbox: バウンディングボックス (x, y, width, height)

        Returns:
            足元座標 (x, y) - バウンディングボックスの中心下端
        """
        x, y, w, h = bbox
        foot_x = x + w / 2
        foot_y = y + h
        return (foot_x, foot_y)

    def get_attention_map(self, _frame: np.ndarray, _layer_index: int = -1) -> np.ndarray | None:
        """Attention Mapを取得（互換性用、YOLOv8では未サポート）.

        Args:
            _frame: 入力画像
            _layer_index: レイヤーインデックス

        Returns:
            None (YOLOv8ではAttention Mapは利用不可)
        """
        logger.warning("Attention map is not available for YOLOv8")
        return None
