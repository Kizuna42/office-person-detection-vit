"""Unit tests for ViTDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.detection import ViTDetector


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_load_model_success(mock_processor_cls, mock_model_cls):
    """`load_model` がモデルとプロセッサをロードし、デバイスへ転送する。"""

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    detector = ViTDetector(device="cpu")
    detector.load_model()

    mock_processor_cls.from_pretrained.assert_called_once_with(detector.model_name)
    mock_model_cls.from_pretrained.assert_called_once_with(detector.model_name)
    mock_model.to.assert_called_once_with("cpu")
    mock_model.eval.assert_called_once()
    assert detector.model is mock_model
    assert detector.processor is mock_processor


@patch("src.detection.vit_detector.ViTDetector._postprocess")
@patch("src.detection.vit_detector.ViTDetector._preprocess")
def test_detect_pipeline(mock_preprocess, mock_postprocess, sample_frame):
    """`detect` は前処理・推論・後処理を順番に呼び出す。"""

    detector = ViTDetector(device="cpu")
    detector.model = Mock()
    detector.processor = Mock()

    mock_inputs = {"pixel_values": np.zeros((1, 3, 224, 224))}
    mock_preprocess.return_value = mock_inputs

    mock_outputs = Mock()
    detector.model.return_value = mock_outputs
    mock_postprocess.return_value = []

    results = detector.detect(sample_frame)

    mock_preprocess.assert_called_once_with(sample_frame)
    detector.model.assert_called_once_with(**mock_inputs)
    mock_postprocess.assert_called_once_with(mock_outputs, sample_frame.shape)
    assert results == []


def test_detect_without_model_raises(sample_frame):
    """モデル未ロード状態で `detect` を呼ぶと `RuntimeError` を送出する。"""

    detector = ViTDetector(device="cpu")
    with pytest.raises(RuntimeError):
        detector.detect(sample_frame)


@patch("src.detection.vit_detector.ViTDetector._postprocess_batch")
@patch("src.detection.vit_detector.ViTDetector._preprocess_batch")
def test_detect_batch(mock_preprocess_batch, mock_postprocess_batch, sample_frame):
    """`detect_batch` はバッチ単位で前処理・推論・後処理を実行する。"""

    detector = ViTDetector(device="cpu")
    detector.model = Mock()
    detector.processor = Mock()

    frames = [sample_frame.copy() for _ in range(3)]

    first_batch_inputs = {"pixel_values": np.zeros((2, 3, 4, 4))}
    second_batch_inputs = {"pixel_values": np.zeros((1, 3, 4, 4))}
    mock_preprocess_batch.side_effect = [first_batch_inputs, second_batch_inputs]

    first_outputs = Mock()
    second_outputs = Mock()
    detector.model.side_effect = [first_outputs, second_outputs]
    mock_postprocess_batch.side_effect = [[["d1"], ["d2"]], [["d3"]]]

    results = detector.detect_batch(frames, batch_size=2)

    first_call_frames = mock_preprocess_batch.call_args_list[0].args[0]
    second_call_frames = mock_preprocess_batch.call_args_list[1].args[0]

    assert all(actual is expected for actual, expected in zip(first_call_frames, frames[:2], strict=False))
    assert all(actual is expected for actual, expected in zip(second_call_frames, frames[2:], strict=False))

    first_model_kwargs = detector.model.call_args_list[0].kwargs
    second_model_kwargs = detector.model.call_args_list[1].kwargs
    assert np.array_equal(first_model_kwargs["pixel_values"], first_batch_inputs["pixel_values"])
    assert np.array_equal(second_model_kwargs["pixel_values"], second_batch_inputs["pixel_values"])
    assert detector.model.call_count == 2

    first_post_call = mock_postprocess_batch.call_args_list[0].args
    second_post_call = mock_postprocess_batch.call_args_list[1].args

    assert first_post_call[0] is first_outputs
    assert first_post_call[1] == [frame.shape for frame in frames[:2]]
    assert second_post_call[0] is second_outputs
    assert second_post_call[1] == [frame.shape for frame in frames[2:]]
    assert results == [["d1"], ["d2"], ["d3"]]


def test_get_foot_position():
    """バウンディングボックス中心下端を足元とみなす。"""

    detector = ViTDetector(device="cpu")
    foot = detector._get_foot_position((100.0, 200.0, 50.0, 100.0))
    assert foot == (125.0, 300.0)


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_load_model_failure(mock_processor_cls, mock_model_cls):
    """モデルロードに失敗した場合のエラーハンドリング"""
    mock_model_cls.from_pretrained.side_effect = Exception("Model load error")

    detector = ViTDetector(device="cpu")
    with pytest.raises(RuntimeError, match="Failed to load model"):
        detector.load_model()


@patch("torch.backends.mps.is_available")
def test_device_setup_mps(mock_mps_available):
    """MPSデバイスの自動検出"""
    mock_mps_available.return_value = True

    detector = ViTDetector(device=None)
    assert detector.device == "mps"


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_device_setup_cuda(mock_cuda_available, mock_mps_available):
    """CUDAデバイスの自動検出（MPSが利用不可の場合）"""
    mock_mps_available.return_value = False
    mock_cuda_available.return_value = True

    detector = ViTDetector(device=None)
    assert detector.device == "cuda"


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_device_setup_cpu(mock_cuda_available, mock_mps_available):
    """CPUデバイスの自動検出（GPUが利用不可の場合）"""
    mock_mps_available.return_value = False
    mock_cuda_available.return_value = False

    detector = ViTDetector(device=None)
    assert detector.device == "cpu"


@patch("torch.backends.mps.is_available")
def test_device_setup_mps_fallback(mock_mps_available):
    """MPSが指定されているが利用不可の場合のフォールバック"""
    mock_mps_available.return_value = False

    detector = ViTDetector(device="mps")
    assert detector.device == "cpu"


@patch("torch.cuda.is_available")
def test_device_setup_cuda_fallback(mock_cuda_available):
    """CUDAが指定されているが利用不可の場合のフォールバック"""
    mock_cuda_available.return_value = False

    detector = ViTDetector(device="cuda")
    assert detector.device == "cpu"


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_detect_batch_without_model(mock_processor_cls, mock_model_cls, sample_frame):
    """モデル未ロード状態でdetect_batchを呼ぶとエラー"""
    detector = ViTDetector(device="cpu")

    with pytest.raises(RuntimeError, match="Model not loaded"):
        detector.detect_batch([sample_frame])


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_detect_batch_empty_frames(mock_processor_cls, mock_model_cls):
    """空のフレームリストでdetect_batchを呼ぶ"""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    detector = ViTDetector(device="cpu")
    detector.load_model()

    results = detector.detect_batch([])
    assert results == []


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_detect_batch_error_handling(mock_processor_cls, mock_model_cls, sample_frame):
    """バッチ処理でエラーが発生した場合のハンドリング"""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_model.side_effect = Exception("Batch processing error")

    detector = ViTDetector(device="cpu")
    detector.load_model()

    frames = [sample_frame.copy() for _ in range(2)]
    results = detector.detect_batch(frames)

    # エラーが発生しても空のリストが返される
    assert len(results) == 2
    assert all(len(detections) == 0 for detections in results)


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_detect_error_handling(mock_processor_cls, mock_model_cls, sample_frame):
    """detectでエラーが発生した場合のハンドリング"""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_model.side_effect = Exception("Detection error")

    detector = ViTDetector(device="cpu")
    detector.load_model()

    with pytest.raises(Exception, match="Detection error"):
        detector.detect(sample_frame)


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_postprocess_with_person_detections(mock_processor_cls, mock_model_cls, sample_frame):
    """人物検出結果の後処理"""
    import torch

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    # モックの後処理結果
    mock_result = {
        "scores": torch.tensor([0.9, 0.8, 0.3]),  # 3つの検出（最後は信頼度低い）
        "labels": torch.tensor([1, 1, 2]),  # 1=person, 2=other
        "boxes": torch.tensor(
            [
                [100.0, 200.0, 150.0, 300.0],  # person
                [200.0, 300.0, 260.0, 420.0],  # person
                [50.0, 50.0, 100.0, 100.0],  # other
            ]
        ),
    }
    mock_processor.post_process_object_detection.return_value = [mock_result]

    detector = ViTDetector(device="cpu", confidence_threshold=0.5)
    detector.load_model()

    # モック出力を作成
    mock_outputs = MagicMock()
    detections = detector._postprocess(mock_outputs, sample_frame.shape)

    # personクラスのみが検出される（信頼度0.5以上）
    assert len(detections) == 2
    assert all(det.class_id == 1 for det in detections)
    assert all(det.class_name == "person" for det in detections)
    assert all(det.confidence >= 0.5 for det in detections)


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_postprocess_no_detections(mock_processor_cls, mock_model_cls, sample_frame):
    """検出結果がない場合の後処理"""
    import torch

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    # 空の検出結果
    mock_result = {
        "scores": torch.tensor([]),
        "labels": torch.tensor([]),
        "boxes": torch.tensor([]).reshape(0, 4),
    }
    mock_processor.post_process_object_detection.return_value = [mock_result]

    detector = ViTDetector(device="cpu")
    detector.load_model()

    mock_outputs = MagicMock()
    detections = detector._postprocess(mock_outputs, sample_frame.shape)

    assert len(detections) == 0


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_postprocess_batch(mock_processor_cls, mock_model_cls, sample_frame):
    """バッチ後処理のテスト"""
    import torch

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    # 2つのフレームの検出結果
    mock_results = [
        {
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[100.0, 200.0, 150.0, 300.0]]),
        },
        {
            "scores": torch.tensor([0.8, 0.7]),
            "labels": torch.tensor([1, 1]),
            "boxes": torch.tensor(
                [
                    [200.0, 300.0, 260.0, 420.0],
                    [300.0, 400.0, 360.0, 520.0],
                ]
            ),
        },
    ]
    mock_processor.post_process_object_detection.return_value = mock_results

    detector = ViTDetector(device="cpu")
    detector.load_model()

    mock_outputs = MagicMock()
    frames = [sample_frame.copy() for _ in range(2)]
    batch_detections = detector._postprocess_batch(mock_outputs, [frame.shape for frame in frames])

    assert len(batch_detections) == 2
    assert len(batch_detections[0]) == 1
    assert len(batch_detections[1]) == 2


def test_get_foot_position_edge_cases():
    """足元座標計算のエッジケース"""
    detector = ViTDetector(device="cpu")

    # ゼロサイズのバウンディングボックス
    foot = detector._get_foot_position((100.0, 200.0, 0.0, 0.0))
    assert foot == (100.0, 200.0)

    # 負のサイズ（通常は発生しないが、テスト）
    foot = detector._get_foot_position((100.0, 200.0, -10.0, -20.0))
    assert foot == (95.0, 180.0)


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_preprocess_bgr_to_rgb(mock_processor_cls, mock_model_cls, sample_frame):
    """BGRからRGBへの変換が正しく行われる"""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    detector = ViTDetector(device="cpu")
    detector.load_model()

    inputs = detector._preprocess(sample_frame)

    # プロセッサが呼ばれていることを確認
    mock_processor.assert_called_once()
    # デバイスに転送されていることを確認
    assert all(hasattr(v, "to") or isinstance(v, dict) for v in inputs.values())


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_preprocess_batch(mock_processor_cls, mock_model_cls, sample_frame):
    """バッチ前処理のテスト"""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    detector = ViTDetector(device="cpu")
    detector.load_model()

    frames = [sample_frame.copy() for _ in range(3)]
    inputs = detector._preprocess_batch(frames)

    # プロセッサが呼ばれていることを確認
    mock_processor.assert_called_once()
    # デバイスに転送されていることを確認
    assert all(hasattr(v, "to") or isinstance(v, dict) for v in inputs.values())


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_confidence_threshold_filtering(mock_processor_cls, mock_model_cls, sample_frame):
    """信頼度閾値によるフィルタリング"""
    import torch

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    # 異なる信頼度の検出結果
    mock_result = {
        "scores": torch.tensor([0.9, 0.6, 0.4]),  # 0.5以上のみが残る
        "labels": torch.tensor([1, 1, 1]),
        "boxes": torch.tensor(
            [
                [100.0, 200.0, 150.0, 300.0],
                [200.0, 300.0, 260.0, 420.0],
                [300.0, 400.0, 360.0, 520.0],
            ]
        ),
    }
    mock_processor.post_process_object_detection.return_value = [mock_result]

    detector = ViTDetector(device="cpu", confidence_threshold=0.5)
    detector.load_model()

    mock_outputs = MagicMock()
    detections = detector._postprocess(mock_outputs, sample_frame.shape)

    # 信頼度0.5以上の検出のみが残る
    assert len(detections) == 2
    assert all(det.confidence >= 0.5 for det in detections)


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_load_model_processor_error(mock_processor_cls, mock_model_cls):
    """プロセッサのロードに失敗した場合のエラーハンドリング"""
    mock_processor_cls.from_pretrained.side_effect = Exception("Processor load error")

    detector = ViTDetector(device="cpu")
    with pytest.raises(RuntimeError, match="Failed to load model"):
        detector.load_model()


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_preprocess_error_handling(mock_processor_cls, mock_model_cls, sample_frame):
    """前処理でエラーが発生した場合のハンドリング"""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor.side_effect = Exception("Preprocessing error")

    detector = ViTDetector(device="cpu")
    detector.load_model()

    with pytest.raises(Exception, match="Preprocessing error"):
        detector._preprocess(sample_frame)


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_postprocess_error_handling(mock_processor_cls, mock_model_cls, sample_frame):
    """後処理でエラーが発生した場合のハンドリング"""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor.post_process_object_detection.side_effect = Exception("Postprocessing error")

    detector = ViTDetector(device="cpu")
    detector.load_model()

    mock_outputs = MagicMock()
    with pytest.raises(Exception, match="Postprocessing error"):
        detector._postprocess(mock_outputs, sample_frame.shape)


@patch("src.detection.vit_detector.DetrForObjectDetection")
@patch("src.detection.vit_detector.DetrImageProcessor")
def test_extract_features_error_handling(mock_processor_cls, mock_model_cls, sample_frame):
    """特徴量抽出でエラーが発生した場合のハンドリング"""
    from src.models.data_models import Detection

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_model.side_effect = Exception("Feature extraction error")

    detector = ViTDetector(device="cpu")
    detector.load_model()

    detections = [
        Detection(bbox=(100, 100, 50, 100), confidence=0.9, class_id=1, class_name="person", camera_coords=(125, 200)),
    ]

    with pytest.raises(Exception, match="Feature extraction error"):
        detector.extract_features(sample_frame, detections)


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_device_setup_explicit_cpu(mock_cuda_available, mock_mps_available):
    """明示的にCPUを指定した場合"""
    mock_mps_available.return_value = True
    mock_cuda_available.return_value = True

    detector = ViTDetector(device="cpu")
    assert detector.device == "cpu"


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_device_setup_explicit_cuda(mock_cuda_available, mock_mps_available):
    """明示的にCUDAを指定した場合（利用可能）"""
    mock_mps_available.return_value = False
    mock_cuda_available.return_value = True

    detector = ViTDetector(device="cuda")
    assert detector.device == "cuda"


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_device_setup_explicit_mps(mock_cuda_available, mock_mps_available):
    """明示的にMPSを指定した場合（利用可能）"""
    mock_mps_available.return_value = True
    mock_cuda_available.return_value = False

    detector = ViTDetector(device="mps")
    assert detector.device == "mps"
