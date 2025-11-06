"""Frame sampling strategies for timestamp-based extraction."""

import logging
from typing import Iterator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CoarseSampler:
    """粗サンプリング: 5分刻みの目標時刻の近傍を高速に特定

    10秒間隔でフレームをサンプリングし、目標時刻の近傍を特定します。
    """

    def __init__(self, video_path: str, interval_seconds: float = 10.0):
        """CoarseSamplerを初期化

        Args:
            video_path: 動画ファイルのパス
            interval_seconds: サンプリング間隔（秒）
        """
        self.video_path = video_path
        self.video: cv2.VideoCapture = None
        self.fps: float = None
        self.interval_seconds = interval_seconds
        self.interval_frames: int = None

    def _ensure_opened(self) -> None:
        """動画ファイルが開かれていることを確認"""
        if self.video is None or not self.video.isOpened():
            self.video = cv2.VideoCapture(self.video_path)
            if not self.video.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            self.interval_frames = int(self.fps * self.interval_seconds)

    def sample(self) -> Iterator[Tuple[int, np.ndarray]]:
        """フレームをサンプリング

        Yields:
            (フレーム番号, フレーム画像) のタプル
        """
        self._ensure_opened()
        frame_idx = 0
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        while frame_idx < total_frames:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video.read()

            if not ret or frame is None:
                break

            yield frame_idx, frame
            frame_idx += self.interval_frames

    def close(self) -> None:
        """リソースを解放"""
        if self.video is not None:
            self.video.release()
            self.video = None


class FineSampler:
    """精密サンプリング: 目標時刻の±10秒以内のベストフレームを特定

    目標時刻の前後30秒範囲を1秒間隔でサンプリングし、
    最も目標時刻に近いフレームを特定します。
    """

    def __init__(self, video: cv2.VideoCapture, search_window: float = 30.0):
        """FineSamplerを初期化

        Args:
            video: OpenCV VideoCaptureオブジェクト
            search_window: 探索ウィンドウ（秒）。目標時刻の前後この秒数分を探索
        """
        self.video = video
        self.search_window = search_window
        self.fps: float = None

    def _ensure_fps(self) -> None:
        """FPSが取得されていることを確認"""
        if self.fps is None:
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            if self.fps is None or self.fps <= 0:
                raise RuntimeError("Failed to get FPS from video")

    def sample_around_target(
        self, approx_frame_idx: int
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """目標時刻の前後を1秒間隔でサンプリング

        Args:
            approx_frame_idx: 近似フレーム番号（探索の中心）

        Yields:
            (フレーム番号, フレーム画像) のタプル
        """
        self._ensure_fps()
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # 探索範囲を計算
        window_frames = int(self.search_window * self.fps)
        start_frame = max(0, approx_frame_idx - window_frames)
        end_frame = min(total_frames, approx_frame_idx + window_frames)

        # 1秒間隔でサンプリング
        frame_interval = int(self.fps)

        for frame_idx in range(start_frame, end_frame, frame_interval):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video.read()

            if ret and frame is not None:
                yield frame_idx, frame


class AdaptiveSampler:
    """OCR信頼度に応じて動的にサンプリング間隔を調整

    信頼度が高い場合は間隔を広げて効率化し、
    信頼度が低い場合は間隔を狭めて精度を向上させます。
    """

    def __init__(
        self,
        base_interval: float = 10.0,
        min_interval: float = 1.0,
        max_interval: float = 30.0,
    ):
        """AdaptiveSamplerを初期化

        Args:
            base_interval: ベースサンプリング間隔（秒）
            min_interval: 最小サンプリング間隔（秒）
            max_interval: 最大サンプリング間隔（秒）
        """
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval

    def adjust_interval(self, recent_confidence: float) -> float:
        """信頼度に応じてサンプリング間隔を調整

        Args:
            recent_confidence: 最近のOCR信頼度（0.0-1.0）

        Returns:
            調整後のサンプリング間隔（秒）
        """
        if recent_confidence >= 0.9:
            # 信頼度が高い: 間隔を広げて効率化
            return min(self.base_interval * 2, self.max_interval)
        elif recent_confidence < 0.5:
            # 信頼度が低い: 間隔を狭めて精度向上
            return self.min_interval
        else:
            # 通常: ベース間隔を使用
            return self.base_interval
