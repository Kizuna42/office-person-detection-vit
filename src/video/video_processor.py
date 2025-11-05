"""Video processing module for the office person detection system."""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """動画処理クラス

    動画ファイルの読み込み、フレーム取得、リソース解放を担当する。

    Attributes:
        video_path: 動画ファイルのパス
        cap: OpenCVのVideoCaptureオブジェクト
        fps: フレームレート
        total_frames: 総フレーム数
        width: 動画の幅
        height: 動画の高さ
    """

    def __init__(self, video_path: str):
        """VideoProcessorを初期化する

        Args:
            video_path: 動画ファイルのパス
        """
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: Optional[float] = None
        self.total_frames: Optional[int] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None

    def open(self) -> bool:
        """動画ファイルを開く

        Returns:
            成功した場合True、失敗した場合False

        Raises:
            FileNotFoundError: 動画ファイルが存在しない場合
            RuntimeError: 動画ファイルの読み込みに失敗した場合
        """
        # ファイルの存在チェック
        if not os.path.exists(self.video_path):
            error_msg = f"動画ファイルが見つかりません: {self.video_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 動画ファイルを開く
        try:
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                error_msg = f"動画ファイルを開けませんでした: {self.video_path}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # 動画情報を取得
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"動画ファイルを開きました: {self.video_path}")
            logger.info(f"  解像度: {self.width}x{self.height}")
            logger.info(f"  FPS: {self.fps}")
            logger.info(f"  総フレーム数: {self.total_frames}")

            return True

        except Exception as e:
            error_msg = f"動画ファイルの読み込み中にエラーが発生しました: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """指定フレームを取得する

        Args:
            frame_number: 取得するフレーム番号（0始まり）

        Returns:
            フレーム画像（numpy配列）、失敗した場合None

        Raises:
            RuntimeError: 動画が開かれていない場合
            ValueError: フレーム番号が範囲外の場合
        """
        if self.cap is None or not self.cap.isOpened():
            error_msg = "動画ファイルが開かれていません。先にopen()を呼び出してください。"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if frame_number < 0 or (
            self.total_frames and frame_number >= self.total_frames
        ):
            error_msg = f"フレーム番号が範囲外です: {frame_number} (総フレーム数: {self.total_frames})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # 指定フレームにシーク
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # フレームを読み込む
            ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.warning(f"フレーム {frame_number} の読み込みに失敗しました")
                return None

            return frame

        except Exception as e:
            logger.error(f"フレーム {frame_number} の取得中にエラーが発生しました: {e}")
            return None

    def get_current_frame_number(self) -> int:
        """現在のフレーム番号を取得する

        Returns:
            現在のフレーム番号

        Raises:
            RuntimeError: 動画が開かれていない場合
        """
        if self.cap is None or not self.cap.isOpened():
            error_msg = "動画ファイルが開かれていません"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def read_next_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """次のフレームを順次読み込む

        Returns:
            (成功フラグ, フレーム画像) のタプル

        Raises:
            RuntimeError: 動画が開かれていない場合
        """
        if self.cap is None or not self.cap.isOpened():
            error_msg = "動画ファイルが開かれていません"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            ret, frame = self.cap.read()
            return ret, frame if ret else None
        except Exception as e:
            logger.error(f"フレームの読み込み中にエラーが発生しました: {e}")
            return False, None

    def reset(self) -> bool:
        """動画を先頭に戻す

        Returns:
            成功した場合True、失敗した場合False
        """
        if self.cap is None or not self.cap.isOpened():
            logger.warning("動画ファイルが開かれていません")
            return False

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.debug("動画を先頭に戻しました")
            return True
        except Exception as e:
            logger.error(f"動画のリセット中にエラーが発生しました: {e}")
            return False

    def release(self):
        """リソースを解放する

        動画ファイルを閉じ、関連リソースを解放する。
        """
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("動画リソースを解放しました")
            except Exception as e:
                logger.error(f"リソース解放中にエラーが発生しました: {e}")
            finally:
                self.cap = None
                self.fps = None
                self.total_frames = None
                self.width = None
                self.height = None

    def __enter__(self):
        """コンテキストマネージャのエントリ"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャの終了"""
        self.release()
        return False

    def __del__(self):
        """デストラクタ"""
        self.release()
