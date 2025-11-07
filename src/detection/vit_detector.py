"""Vision Transformer based person detection module."""

import logging

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
        device: str | None = None,
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

    def _setup_device(self, device: str | None = None) -> str:
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
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA is not available. Falling back to CPU.")
                return "cpu"
            return device

        # 自動検出
        if torch.backends.mps.is_available():
            logger.info("MPS device detected and will be used for acceleration.")
            return "mps"
        if torch.cuda.is_available():
            logger.info("CUDA device detected and will be used for acceleration.")
            return "cuda"
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
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e

    def detect(self, frame: np.ndarray, extract_features: bool = False) -> list[Detection]:
        """人物検出を実行

        Args:
            frame: 入力画像 (numpy array, BGR format)
            extract_features: 特徴量を抽出するか（デフォルト: False）

        Returns:
            検出結果のリスト（特徴量が含まれる場合あり）

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
                outputs = self.model(**inputs, output_hidden_states=True) if extract_features else self.model(**inputs)

            # 後処理
            detections = self._postprocess(outputs, frame.shape, extract_features=extract_features)

            logger.debug(f"Detected {len(detections)} persons")
            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

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

    def _postprocess(
        self, outputs, image_shape: tuple[int, int, int], extract_features: bool = False
    ) -> list[Detection]:
        """モデル出力を検出結果に変換

        Args:
            outputs: モデルの出力
            image_shape: 元画像のshape (height, width, channels)
            extract_features: 特徴量を抽出するか

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

        # 特徴量を抽出する場合
        encoder_features = None
        if extract_features and hasattr(outputs, "encoder_last_hidden_state"):
            encoder_features = outputs.encoder_last_hidden_state  # (batch, seq_len, hidden_dim)

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

            # 特徴量を抽出（該当する場合）
            features = None
            if extract_features and encoder_features is not None:
                features = self._extract_detection_features(encoder_features[0], bbox, image_shape[:2])

            # Detection オブジェクトを作成
            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=person_class_id,
                class_name="person",
                camera_coords=camera_coords,
                features=features,
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

    def extract_features(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        """検出結果に対応する特徴量を抽出

        DETRエンコーダーの最終層の特徴量を使用して、各検出バウンディングボックス
        領域の特徴量を抽出します。L2正規化を適用してコサイン類似度計算に適した
        形式にします。

        Args:
            frame: 入力画像 (numpy array, BGR format)
            detections: 検出結果のリスト

        Returns:
            特徴量のリスト（各要素は256次元のnumpy配列、L2正規化済み）

        Raises:
            RuntimeError: モデルがロードされていない場合
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not detections:
            return []

        try:
            # 前処理
            inputs = self._preprocess(frame)

            # 推論（エンコーダー特徴量を取得）
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # エンコーダーの最終層特徴量を取得
            if not hasattr(outputs, "encoder_last_hidden_state") or outputs.encoder_last_hidden_state is None:
                logger.warning("Encoder features not available. Returning empty features.")
                return [None] * len(detections)

            encoder_features = outputs.encoder_last_hidden_state[0]  # (seq_len, hidden_dim)

            # 各検出に対応する特徴量を抽出
            features_list = []
            height, width = frame.shape[:2]

            for detection in detections:
                feature = self._extract_detection_features(encoder_features, detection.bbox, (height, width))
                features_list.append(feature)

            logger.debug(f"Extracted features for {len(features_list)} detections")
            return features_list

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

    def _extract_detection_features(
        self, encoder_features: torch.Tensor, bbox: tuple[float, float, float, float], image_size: tuple[int, int]
    ) -> np.ndarray:
        """検出バウンディングボックスに対応する特徴量を抽出

        DETRエンコーダーの特徴量から、バウンディングボックス領域に対応する
        特徴量を抽出します。ROI Pooling的なアプローチを使用します。

        Args:
            encoder_features: エンコーダー特徴量 (seq_len, hidden_dim)
            bbox: バウンディングボックス (x, y, width, height)
            image_size: 画像サイズ (height, width)

        Returns:
            特徴量ベクトル (hidden_dim次元、L2正規化済み)
        """
        height, width = image_size
        x, y, bbox_width, bbox_height = bbox

        # バウンディングボックスの中心座標を計算
        center_x = x + bbox_width / 2.0
        center_y = y + bbox_height / 2.0

        # 正規化座標に変換（0-1範囲）
        norm_center_x = center_x / width
        norm_center_y = center_y / height

        # エンコーダー特徴量のシーケンス長からパッチ数を推定
        # DETRでは通常、画像はパッチに分割される
        seq_len = encoder_features.shape[0]
        # CLSトークンを除く（最初のトークン）
        num_patches = seq_len - 1

        # パッチグリッドのサイズを計算
        patch_size = int(np.sqrt(num_patches))
        if patch_size * patch_size != num_patches:
            # 正確な平方根でない場合、近似を使用
            patch_size = int(np.sqrt(num_patches))

        # バウンディングボックス中心に対応するパッチインデックスを計算
        patch_x = int(norm_center_x * patch_size)
        patch_y = int(norm_center_y * patch_size)
        patch_x = max(0, min(patch_x, patch_size - 1))
        patch_y = max(0, min(patch_y, patch_size - 1))

        # パッチインデックスをシーケンスインデックスに変換（+1はCLSトークンのため）
        patch_idx = patch_y * patch_size + patch_x + 1
        patch_idx = min(patch_idx, seq_len - 1)

        # 対応するパッチの特徴量を取得
        # 周辺パッチも考慮して平均化（よりロバストな特徴量）
        feature = encoder_features[patch_idx]

        # 周辺パッチも考慮（オプション）
        if patch_size > 1:
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny = patch_y + dy
                    nx = patch_x + dx
                    if 0 <= ny < patch_size and 0 <= nx < patch_size:
                        neighbor_idx = ny * patch_size + nx + 1
                        if neighbor_idx < seq_len:
                            neighbors.append(encoder_features[neighbor_idx])

            if neighbors:
                # 中心パッチと周辺パッチの平均を取る
                neighbor_tensor = torch.stack(neighbors)
                feature = (feature + neighbor_tensor.mean(dim=0)) / 2.0

        # CPUに移動してnumpy配列に変換
        feature_np = feature.cpu().numpy()

        # L2正規化（コサイン類似度計算のため）
        norm = np.linalg.norm(feature_np)
        if norm > 0:
            feature_np = feature_np / norm

        return feature_np.astype(np.float32)

    def get_attention_map(self, frame: np.ndarray, layer_index: int = -1) -> np.ndarray | None:
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

    def detect_batch(self, frames: list[np.ndarray], batch_size: int | None = None) -> list[list[Detection]]:
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
