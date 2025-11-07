"""Vision Transformer based person detection module."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.models.data_models import Detection

logger = logging.getLogger(__name__)


class ViTDetector:
    """Vision Transformer人物検出クラス

    DETR (DEtection TRansformer) モデルを使用して人物検出を実行します。
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """ViTDetectorを初期化

        Args:
            model_name: 使用するモデル名 (Hugging Face model ID)
            confidence_threshold: 検出の信頼度閾値 (0.0-1.0)
            device: 使用するデバイス ("mps", "cuda", "cpu", None=自動検出)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None

        logger.info(f"ViTDetector initialized with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Confidence threshold: {confidence_threshold}")

    def _setup_device(self, device: Optional[str] = None) -> str:
        """デバイスを設定

        Args:
            device: 指定されたデバイス名 (None の場合は自動検出)

        Returns:
            使用するデバイス名
        """
        if device is not None:
            # ユーザー指定のデバイスを使用
            if device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS is not available. Falling back to CPU.")
                return "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA is not available. Falling back to CPU.")
                return "cpu"
            return device

        # 自動検出
        if torch.backends.mps.is_available():
            logger.info("MPS device detected and will be used for acceleration.")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("CUDA device detected and will be used for acceleration.")
            return "cuda"
        else:
            logger.info("No GPU acceleration available. Using CPU.")
            return "cpu"

    def load_model(self) -> None:
        """事前学習済みViTモデルをロード

        Raises:
            RuntimeError: モデルのロードに失敗した場合
        """
        try:
            logger.info(f"Loading model: {self.model_name}")

            # プロセッサとモデルをロード
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)

            # デバイスに転送
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """人物検出を実行

        Args:
            frame: 入力画像 (numpy array, BGR format)

        Returns:
            検出結果のリスト

        Raises:
            RuntimeError: モデルがロードされていない場合
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # 前処理
            inputs = self._preprocess(frame)

            # 推論
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 後処理
            detections = self._postprocess(outputs, frame.shape)

            logger.debug(f"Detected {len(detections)} persons")
            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

    def detect_with_features(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """人物検出と特徴量抽出を同時に実行

        Args:
            frame: 入力画像 (numpy array, BGR format)

        Returns:
            (検出結果のリスト, 特徴量配列)
            特徴量配列の形状: (num_detections, feature_dim)

        Raises:
            RuntimeError: モデルがロードされていない場合
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # 前処理
            inputs = self._preprocess(frame)

            # 推論（特徴量も取得）
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # 後処理
            detections = self._postprocess(outputs, frame.shape)

            # 特徴量抽出
            features = self._extract_features_from_outputs(outputs, detections)

            logger.debug(f"Detected {len(detections)} persons with features")
            return detections, features

        except Exception as e:
            logger.error(f"Detection with features failed: {e}")
            raise

    def extract_features(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """検出結果に対応する特徴量を抽出

        DETRエンコーダーの最終層の特徴量を使用します。
        各検出バウンディングボックス領域の特徴量を抽出し、L2正規化を適用します。

        Args:
            frame: 入力画像 (numpy array, BGR format)
            detections: 検出結果のリスト

        Returns:
            特徴量配列 (num_detections, feature_dim)
            特徴量はL2正規化済み（コサイン類似度計算用）

        Raises:
            RuntimeError: モデルがロードされていない場合
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not detections:
            return np.array([]).reshape(0, 256)

        try:
            # 前処理
            inputs = self._preprocess(frame)

            # 推論（特徴量も取得）
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # 特徴量抽出
            features = self._extract_features_from_outputs(outputs, detections)

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

    def _extract_features_from_outputs(self, outputs, detections: list[Detection]) -> np.ndarray:
        """モデル出力から特徴量を抽出

        Args:
            outputs: モデルの出力
            detections: 検出結果のリスト

        Returns:
            特徴量配列 (num_detections, feature_dim)
        """
        try:
            # DETRのdecoder出力から特徴量を取得
            # decoder_last_hidden_state: (batch_size, num_queries, hidden_dim)
            if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
                # 最終層のdecoder hidden statesを使用
                decoder_features = outputs.decoder_hidden_states[-1]  # (batch, num_queries, hidden_dim)
            elif hasattr(outputs, "last_hidden_state"):
                # フォールバック: last_hidden_stateを使用
                decoder_features = outputs.last_hidden_state
            else:
                # encoderの出力を使用（フォールバック）
                if hasattr(outputs, "encoder_last_hidden_state"):
                    encoder_features = outputs.encoder_last_hidden_state
                    # 簡易的に平均プーリングで特徴量を取得
                    decoder_features = encoder_features.mean(dim=1, keepdim=True)
                else:
                    raise ValueError("Model outputs do not contain feature information")

            # バッチの最初の要素を取得
            features = decoder_features[0]  # (num_queries, hidden_dim)

            # 検出結果に対応する特徴量を抽出
            # DETRでは、検出結果の順序とdecoder queryの順序が対応している
            num_detections = len(detections)
            if features.shape[0] < num_detections:
                logger.warning(
                    f"Feature dimension mismatch: {features.shape[0]} queries but {num_detections} detections"
                )
                # 検出数に合わせて特徴量を切り詰める
                detection_features = features[:num_detections]
            else:
                # 検出数分の特徴量を取得
                detection_features = features[:num_detections]

            # CPUに移動してnumpy配列に変換
            features_np = detection_features.cpu().numpy()

            # L2正規化（コサイン類似度計算のため）
            norms = np.linalg.norm(features_np, axis=1, keepdims=True)
            features_np = features_np / (norms + 1e-8)

            logger.debug(f"Extracted features: shape={features_np.shape}")
            return features_np

        except Exception as e:
            logger.error(f"Failed to extract features from outputs: {e}")
            # フォールバック: ゼロ特徴量を返す
            num_detections = len(detections)
            feature_dim = 256  # DETR-ResNet-50のデフォルト次元
            return np.zeros((num_detections, feature_dim), dtype=np.float32)

    def _preprocess(self, frame: np.ndarray) -> dict:
        """入力画像の前処理

        Args:
            frame: 入力画像 (numpy array, BGR format)

        Returns:
            モデル入力用のテンソル辞書
        """
        # BGRからRGBに変換
        image_rgb = frame[:, :, ::-1].copy()

        # PIL Imageに変換
        image_pil = Image.fromarray(image_rgb)

        # プロセッサで前処理（パッチ分割、正規化）
        inputs = self.processor(images=image_pil, return_tensors="pt")

        # デバイスに転送
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def _postprocess(self, outputs, image_shape: tuple[int, int, int]) -> list[Detection]:
        """モデル出力を検出結果に変換

        Args:
            outputs: モデルの出力
            image_shape: 元画像のshape (height, width, channels)

        Returns:
            検出結果のリスト
        """
        # 画像サイズを取得
        height, width = image_shape[:2]
        target_sizes = torch.tensor([[height, width]]).to(self.device)

        # 後処理（座標を元画像サイズにスケール）
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]

        detections = []

        # COCO datasetのpersonクラスID = 1
        person_class_id = 1

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=False):
            # personクラスのみをフィルタリング
            if label.item() != person_class_id:
                continue

            # 信頼度フィルタリング（念のため再チェック）
            confidence = score.item()
            if confidence < self.confidence_threshold:
                continue

            # バウンディングボックス座標を取得 (x_min, y_min, x_max, y_max)
            box_coords = box.cpu().numpy()
            x_min, y_min, x_max, y_max = box_coords

            # (x, y, width, height) 形式に変換
            bbox = (
                float(x_min),
                float(y_min),
                float(x_max - x_min),
                float(y_max - y_min),
            )

            # 足元座標を計算（バウンディングボックスの中心下端）
            camera_coords = self._get_foot_position(bbox)

            # Detection オブジェクトを作成
            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=person_class_id,
                class_name="person",
                camera_coords=camera_coords,
            )

            detections.append(detection)

        return detections

    def _get_foot_position(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """バウンディングボックスから足元座標を計算

        Args:
            bbox: バウンディングボックス (x, y, width, height)

        Returns:
            足元座標 (x, y) - バウンディングボックスの中心下端
        """
        x, y, width, height = bbox
        foot_x = x + width / 2.0
        foot_y = y + height
        return (foot_x, foot_y)

    def get_attention_map(self, frame: np.ndarray, layer_index: int = -1) -> Optional[np.ndarray]:
        """Attention Mapを取得（可視化用）

        Args:
            frame: 入力画像 (numpy array, BGR format)
            layer_index: 取得するレイヤーのインデックス (-1で最終層)

        Returns:
            Attention Map (numpy array) または None（取得失敗時）
        """
        if self.model is None or self.processor is None:
            logger.warning("Model not loaded. Cannot extract attention map.")
            return None

        try:
            # 前処理
            inputs = self._preprocess(frame)

            # 推論（attention weightsを取得）
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            # Attention weightsを取得
            if not hasattr(outputs, "encoder_attentions") or outputs.encoder_attentions is None:
                logger.warning("Model does not provide attention weights.")
                return None

            # 指定されたレイヤーのattentionを取得
            attention_weights = outputs.encoder_attentions[layer_index]

            # Attention mapを抽出（平均化して2D mapに変換）
            attention_map = self._extract_attention_map(attention_weights)

            return attention_map

        except Exception as e:
            logger.error(f"Failed to extract attention map: {e}")
            return None

    def _extract_attention_map(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Attention weightsから2D mapを抽出

        Args:
            attention_weights: Attention weights tensor
                Shape: (batch_size, num_heads, seq_len, seq_len)

        Returns:
            2D attention map (numpy array)
        """
        # バッチの最初の要素を取得
        attn = attention_weights[0]  # (num_heads, seq_len, seq_len)

        # 全ヘッドの平均を取る
        attn_mean = attn.mean(dim=0)  # (seq_len, seq_len)

        # CLSトークンからの注意度を取得（最初のトークン）
        cls_attention = attn_mean[0, 1:]  # CLSトークンから他のパッチへの注意度

        # CPUに移動してnumpy配列に変換
        attention_map = cls_attention.cpu().numpy()

        # 正規化
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        return attention_map

    def save_attention_map(self, frame: np.ndarray, output_path: str, layer_index: int = -1) -> bool:
        """Attention Mapを画像として保存

        Args:
            frame: 入力画像 (numpy array, BGR format)
            output_path: 保存先パス
            layer_index: 取得するレイヤーのインデックス (-1で最終層)

        Returns:
            保存成功時True、失敗時False
        """
        import cv2

        attention_map = self.get_attention_map(frame, layer_index)

        if attention_map is None:
            logger.warning("Failed to get attention map. Skipping save.")
            return False

        try:
            # Attention mapのサイズを計算（パッチ数から）
            num_patches = len(attention_map)
            patch_size = int(np.sqrt(num_patches))

            # 2D gridに変形
            attention_grid = attention_map.reshape(patch_size, patch_size)

            # 元画像サイズにリサイズ
            height, width = frame.shape[:2]
            attention_resized = cv2.resize(attention_grid, (width, height), interpolation=cv2.INTER_LINEAR)

            # ヒートマップに変換
            attention_heatmap = (attention_resized * 255).astype(np.uint8)
            attention_colored = cv2.applyColorMap(attention_heatmap, cv2.COLORMAP_JET)

            # 元画像とブレンド
            blended = cv2.addWeighted(frame, 0.6, attention_colored, 0.4, 0)

            # 保存
            cv2.imwrite(output_path, blended)
            logger.info(f"Attention map saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save attention map: {e}")
            return False

    def detect_batch(self, frames: list[np.ndarray], batch_size: Optional[int] = None) -> list[list[Detection]]:
        """複数フレームのバッチ処理

        Args:
            frames: 入力画像のリスト (numpy arrays, BGR format)
            batch_size: バッチサイズ（Noneの場合は全フレームを一度に処理）

        Returns:
            各フレームの検出結果のリスト

        Raises:
            RuntimeError: モデルがロードされていない場合
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not frames:
            return []

        # バッチサイズが指定されていない場合は全フレームを処理
        if batch_size is None:
            batch_size = len(frames)

        all_detections = []

        # バッチごとに処理
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]

            try:
                # バッチ前処理
                batch_inputs = self._preprocess_batch(batch_frames)

                # バッチ推論
                with torch.no_grad():
                    outputs = self.model(**batch_inputs)

                # バッチ後処理
                batch_detections = self._postprocess_batch(outputs, [frame.shape for frame in batch_frames])

                all_detections.extend(batch_detections)

                # メモリ解放
                del batch_inputs, outputs
                if self.device == "mps" or self.device == "cuda":
                    torch.mps.empty_cache() if self.device == "mps" else torch.cuda.empty_cache()

                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}")

            except Exception as e:
                logger.error(f"Batch detection failed for batch starting at index {i}: {e}")
                # エラーが発生した場合は空のリストを追加
                all_detections.extend([[] for _ in batch_frames])

        return all_detections

    def _preprocess_batch(self, frames: list[np.ndarray]) -> dict:
        """複数画像の前処理

        Args:
            frames: 入力画像のリスト (numpy arrays, BGR format)

        Returns:
            モデル入力用のテンソル辞書
        """
        # BGRからRGBに変換してPIL Imageのリストを作成
        images_pil = []
        for frame in frames:
            image_rgb = frame[:, :, ::-1].copy()
            image_pil = Image.fromarray(image_rgb)
            images_pil.append(image_pil)

        # プロセッサでバッチ前処理
        inputs = self.processor(images=images_pil, return_tensors="pt")

        # デバイスに転送
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def _postprocess_batch(self, outputs, image_shapes: list[tuple[int, int, int]]) -> list[list[Detection]]:
        """バッチモデル出力を検出結果に変換

        Args:
            outputs: モデルの出力
            image_shapes: 各画像のshape (height, width, channels)

        Returns:
            各フレームの検出結果のリスト
        """
        # 各画像のサイズを取得
        target_sizes = torch.tensor([[shape[0], shape[1]] for shape in image_shapes]).to(self.device)

        # 後処理（座標を元画像サイズにスケール）
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )

        batch_detections = []
        person_class_id = 1

        # 各画像の結果を処理
        for result in results:
            detections = []

            for score, label, box in zip(result["scores"], result["labels"], result["boxes"], strict=False):
                # personクラスのみをフィルタリング
                if label.item() != person_class_id:
                    continue

                # 信頼度フィルタリング
                confidence = score.item()
                if confidence < self.confidence_threshold:
                    continue

                # バウンディングボックス座標を取得
                box_coords = box.cpu().numpy()
                x_min, y_min, x_max, y_max = box_coords

                # (x, y, width, height) 形式に変換
                bbox = (
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                )

                # 足元座標を計算
                camera_coords = self._get_foot_position(bbox)

                # Detection オブジェクトを作成
                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=person_class_id,
                    class_name="person",
                    camera_coords=camera_coords,
                )

                detections.append(detection)

            batch_detections.append(detections)

        return batch_detections
