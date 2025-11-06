"""Integrated timestamp extraction module (V2)."""

import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from src.timestamp.ocr_engine import MultiEngineOCR
from src.timestamp.roi_extractor import TimestampROIExtractor
from src.timestamp.timestamp_parser import TimestampParser
from src.timestamp.timestamp_validator import TemporalValidator

logger = logging.getLogger(__name__)


class TimestampExtractorV2:
    """高精度タイムスタンプ抽出の統合クラス

    ROI抽出、OCR、パース、時系列検証を統合し、
    高精度なタイムスタンプ抽出を実現します。
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        roi_config: Dict[str, float] = None,
        fps: float = 30.0,
        enabled_ocr_engines: list = None
    ):
        """TimestampExtractorV2を初期化

        Args:
            confidence_threshold: 総合信頼度の閾値（0.0-1.0）
            roi_config: ROI設定（Noneの場合はデフォルト）
            fps: 動画のフレームレート
            enabled_ocr_engines: 有効にするOCRエンジンのリスト
        """
        self.roi_extractor = TimestampROIExtractor(roi_config)
        self.ocr_engine = MultiEngineOCR(enabled_engines=enabled_ocr_engines)
        self.parser = TimestampParser()
        self.validator = TemporalValidator(fps=fps)
        self.confidence_threshold = confidence_threshold

    def extract(
        self,
        frame: np.ndarray,
        frame_idx: int,
        retry_count: int = 3
    ) -> Optional[Dict[str, any]]:
        """フレームからタイムスタンプを抽出

        Args:
            frame: 入力フレーム画像（BGR形式）
            frame_idx: フレーム番号
            retry_count: リトライ回数

        Returns:
            抽出結果の辞書。以下のキーを持つ:
                - timestamp: datetimeオブジェクト
                - frame_idx: フレーム番号
                - confidence: 総合信頼度
                - ocr_text: OCR結果テキスト
                - roi_coords: ROI座標 (x, y, width, height)
            失敗した場合はNone
        """
        # ROI抽出
        roi, roi_coords = self.roi_extractor.extract_roi(frame)

        if roi.size == 0:
            logger.warning(f"Frame {frame_idx}: Empty ROI")
            return None

        for attempt in range(retry_count):
            try:
                # 前処理
                preprocessed = self.roi_extractor.preprocess_roi(roi)

                # OCR実行
                ocr_text, ocr_confidence = self.ocr_engine.extract_with_consensus(preprocessed)

                if ocr_text is None:
                    logger.warning(f"Frame {frame_idx}: OCR failed (attempt {attempt+1}/{retry_count})")
                    continue

                # パース
                timestamp, parse_confidence = self.parser.fuzzy_parse(ocr_text)

                if timestamp is None:
                    logger.warning(f"Frame {frame_idx}: Parse failed for '{ocr_text}' (attempt {attempt+1}/{retry_count})")
                    continue

                # 時系列検証
                is_valid, temporal_confidence, reason = self.validator.validate(timestamp, frame_idx)

                # 総合信頼度
                total_confidence = (ocr_confidence + parse_confidence + temporal_confidence) / 3

                if total_confidence >= self.confidence_threshold and is_valid:
                    logger.info(f"Frame {frame_idx}: {timestamp} (confidence={total_confidence:.2f})")
                    return {
                        'timestamp': timestamp,
                        'frame_idx': frame_idx,
                        'confidence': total_confidence,
                        'ocr_text': ocr_text,
                        'roi_coords': roi_coords
                    }
                else:
                    logger.debug(
                        f"Frame {frame_idx}: Low confidence ({total_confidence:.2f}), "
                        f"valid={is_valid}, {reason}"
                    )
            except Exception as e:
                logger.error(f"Frame {frame_idx}: Error during extraction (attempt {attempt+1}/{retry_count}): {e}")

        logger.error(f"Frame {frame_idx}: Failed after {retry_count} attempts")
        return None

    def reset_validator(self) -> None:
        """時系列検証器をリセット"""
        self.validator.reset()

