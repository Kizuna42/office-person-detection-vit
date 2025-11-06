"""Unit tests for VideoProcessor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.video import VideoProcessor


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """テスト用の動画ファイルパスを作成（実際のファイルは作成しない）"""
    return tmp_path / "test_video.mov"


def test_init(sample_video_path: Path):
    """VideoProcessor が正しく初期化される。"""

    processor = VideoProcessor(str(sample_video_path))
    assert processor.video_path == str(sample_video_path)
    assert processor.cap is None
    assert processor.fps is None
    assert processor.total_frames is None


def test_open_file_not_found():
    """存在しない動画ファイルを開こうとすると FileNotFoundError が発生する。"""

    processor = VideoProcessor("nonexistent_video.mov")
    with pytest.raises(FileNotFoundError):
        processor.open()


@patch("cv2.VideoCapture")
def test_open_success(mock_video_capture, sample_video_path: Path):
    """動画ファイルを正しく開ける。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    # ファイル存在チェックをモック
    with patch("os.path.exists", return_value=True):
        result = processor.open()

    assert result is True
    assert processor.cap is not None
    assert processor.fps == 30.0
    assert processor.total_frames == 100
    assert processor.width == 1280
    assert processor.height == 720


@patch("cv2.VideoCapture")
def test_open_failure(mock_video_capture, sample_video_path: Path):
    """動画ファイルを開けない場合は RuntimeError が発生する。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        with pytest.raises(RuntimeError):
            processor.open()


@patch("cv2.VideoCapture")
def test_get_frame_success(mock_video_capture, sample_video_path: Path):
    """指定フレームを正しく取得できる。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, test_frame)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    frame = processor.get_frame(10)
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    mock_cap.set.assert_called_once()
    mock_cap.read.assert_called_once()


def test_get_frame_not_opened():
    """動画が開かれていない場合は RuntimeError が発生する。"""

    processor = VideoProcessor("test_video.mov")
    with pytest.raises(RuntimeError, match="動画ファイルが開かれていません"):
        processor.get_frame(0)


@patch("cv2.VideoCapture")
def test_get_frame_out_of_range(mock_video_capture, sample_video_path: Path):
    """範囲外のフレーム番号では ValueError が発生する。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    with pytest.raises(ValueError, match="範囲外"):
        processor.get_frame(200)


@patch("cv2.VideoCapture")
def test_get_frame_negative_index(mock_video_capture, sample_video_path: Path):
    """負のフレーム番号では ValueError が発生する。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    with pytest.raises(ValueError, match="範囲外"):
        processor.get_frame(-1)


@patch("cv2.VideoCapture")
def test_get_frame_read_failure(mock_video_capture, sample_video_path: Path):
    """フレーム読み込みに失敗した場合は None が返される。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    mock_cap.read.return_value = (False, None)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    frame = processor.get_frame(10)
    assert frame is None


@patch("cv2.VideoCapture")
def test_get_current_frame_number(mock_video_capture, sample_video_path: Path):
    """現在のフレーム番号を正しく取得できる。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_POS_FRAMES: 25,
    }.get(prop, 0)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    frame_number = processor.get_current_frame_number()
    assert frame_number == 25


def test_get_current_frame_number_not_opened():
    """動画が開かれていない場合は RuntimeError が発生する。"""

    processor = VideoProcessor("test_video.mov")
    with pytest.raises(RuntimeError, match="動画ファイルが開かれていません"):
        processor.get_current_frame_number()


@patch("cv2.VideoCapture")
def test_read_next_frame(mock_video_capture, sample_video_path: Path):
    """次のフレームを順次読み込める。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, test_frame)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    ret, frame = processor.read_next_frame()
    assert ret is True
    assert frame is not None


@patch("cv2.VideoCapture")
def test_read_next_frame_eof(mock_video_capture, sample_video_path: Path):
    """動画の終端では (False, None) が返される。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)
    mock_cap.read.return_value = (False, None)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    ret, frame = processor.read_next_frame()
    assert ret is False
    assert frame is None


@patch("cv2.VideoCapture")
def test_reset(mock_video_capture, sample_video_path: Path):
    """動画を先頭に戻せる。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    result = processor.reset()
    assert result is True
    mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 0)


def test_reset_not_opened():
    """動画が開かれていない場合は False が返される。"""

    processor = VideoProcessor("test_video.mov")
    result = processor.reset()
    assert result is False


@patch("cv2.VideoCapture")
def test_release(mock_video_capture, sample_video_path: Path):
    """リソースを正しく解放できる。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        processor.open()

    processor.release()
    mock_cap.release.assert_called_once()
    assert processor.cap is None
    assert processor.fps is None


@patch("cv2.VideoCapture")
def test_context_manager(mock_video_capture, sample_video_path: Path):
    """コンテキストマネージャーとして使用できる。"""

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

    mock_video_capture.return_value = mock_cap

    processor = VideoProcessor(str(sample_video_path))
    processor.video_path = str(sample_video_path)

    with patch("os.path.exists", return_value=True):
        with processor:
            assert processor.cap is not None

    mock_cap.release.assert_called_once()
