"""Test cases for memory_utils."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from src.utils.memory_utils import cleanup_resources


@pytest.fixture
def sample_logger() -> logging.Logger:
    """テスト用のロガー"""
    logger = logging.getLogger("test_memory_utils")
    logger.setLevel(logging.DEBUG)
    return logger


def test_cleanup_resources_none(sample_logger: logging.Logger):
    """リソースがNoneの場合でもエラーが発生しない"""
    cleanup_resources(video_processor=None, detector=None, logger=sample_logger)


def test_cleanup_resources_video_processor(sample_logger: logging.Logger):
    """VideoProcessorのクリーンアップが正しく動作する"""
    mock_video_processor = Mock()

    cleanup_resources(video_processor=mock_video_processor, detector=None, logger=sample_logger)

    mock_video_processor.release.assert_called_once()


def test_cleanup_resources_video_processor_error(sample_logger: logging.Logger):
    """VideoProcessorのクリーンアップでエラーが発生しても処理が続行される"""
    mock_video_processor = Mock()
    mock_video_processor.release.side_effect = Exception("Release error")

    # エラーが発生しても例外が発生しないことを確認
    cleanup_resources(video_processor=mock_video_processor, detector=None, logger=sample_logger)


@patch("src.utils.memory_utils.torch")
def test_cleanup_resources_detector_mps(mock_torch, sample_logger: logging.Logger):
    """MPSデバイスのクリーンアップが正しく動作する"""
    mock_torch.backends.mps.is_available.return_value = True

    mock_detector = Mock()
    mock_detector.device = "mps"

    cleanup_resources(video_processor=None, detector=mock_detector, logger=sample_logger)

    mock_torch.mps.empty_cache.assert_called_once()


@patch("src.utils.memory_utils.torch")
def test_cleanup_resources_detector_cuda(mock_torch, sample_logger: logging.Logger):
    """CUDAデバイスのクリーンアップが正しく動作する"""
    mock_torch.cuda.is_available.return_value = True

    mock_detector = Mock()
    mock_detector.device = "cuda"

    cleanup_resources(video_processor=None, detector=mock_detector, logger=sample_logger)

    mock_torch.cuda.empty_cache.assert_called_once()


def test_cleanup_resources_detector_cpu(sample_logger: logging.Logger):
    """CPUデバイスの場合、キャッシュクリアは実行されない"""
    mock_detector = Mock()
    mock_detector.device = "cpu"

    # エラーが発生しないことを確認
    cleanup_resources(video_processor=None, detector=mock_detector, logger=sample_logger)


@patch("src.utils.memory_utils.torch")
def test_cleanup_resources_detector_error(mock_torch, sample_logger: logging.Logger):
    """デテクターのクリーンアップでエラーが発生しても処理が続行される"""
    mock_torch.backends.mps.is_available.side_effect = Exception("MPS error")

    mock_detector = Mock()
    mock_detector.device = "mps"

    # エラーが発生しても例外が発生しないことを確認
    cleanup_resources(video_processor=None, detector=mock_detector, logger=sample_logger)


def test_cleanup_resources_no_logger():
    """ロガーがNoneの場合でもエラーが発生しない"""
    mock_video_processor = Mock()
    mock_detector = Mock()
    mock_detector.device = "cpu"

    cleanup_resources(video_processor=mock_video_processor, detector=mock_detector, logger=None)
