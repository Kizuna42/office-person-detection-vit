"""Unit tests for ViTDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.vit_detector import ViTDetector


@patch("src.vit_detector.DetrForObjectDetection")
@patch("src.vit_detector.DetrImageProcessor")
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


@patch("src.vit_detector.ViTDetector._postprocess")
@patch("src.vit_detector.ViTDetector._preprocess")
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


@patch("src.vit_detector.ViTDetector._postprocess_batch")
@patch("src.vit_detector.ViTDetector._preprocess_batch")
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

    assert all(
        actual is expected for actual, expected in zip(first_call_frames, frames[:2])
    )
    assert all(
        actual is expected for actual, expected in zip(second_call_frames, frames[2:])
    )

    first_model_kwargs = detector.model.call_args_list[0].kwargs
    second_model_kwargs = detector.model.call_args_list[1].kwargs
    assert np.array_equal(
        first_model_kwargs["pixel_values"], first_batch_inputs["pixel_values"]
    )
    assert np.array_equal(
        second_model_kwargs["pixel_values"], second_batch_inputs["pixel_values"]
    )
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
