"""Integrated timestamp extraction module (V2)."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import logging
import threading
from typing import Any, Protocol

import numpy as np

from src.timestamp.ocr_engine import MultiEngineOCR
from src.timestamp.roi_extractor import TimestampROIExtractor
from src.timestamp.timestamp_parser import TimestampParser
from src.timestamp.timestamp_validator_v2 import TemporalValidatorV2

logger = logging.getLogger(__name__)


class TimestampValidator(Protocol):
    """タイムスタンプ検証のためのプロトコル。"""

    def validate(self, timestamp: datetime, frame_idx: int) -> tuple[bool, float, str]: ...

    def reset(self) -> None: ...


class TimestampExtractorV2:
    """高精度タイムスタンプ抽出の統合クラス

    ROI抽出、OCR、パース、時系列検証を統合し、
    高精度なタイムスタンプ抽出を実現します。
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        roi_config: dict[str, float] | None = None,
        fps: float = 30.0,
        enabled_ocr_engines: list[str] | None = None,
        use_improved_validator: bool = False,
        base_tolerance_seconds: float = 10.0,
        history_size: int = 10,
        z_score_threshold: float = 2.0,
        use_weighted_consensus: bool = False,
        use_voting_consensus: bool = False,
    ):
        """TimestampExtractorV2を初期化

        Args:
            confidence_threshold: 総合信頼度の閾値（0.0-1.0）
            roi_config: ROI設定（Noneの場合はデフォルト）
            fps: 動画のフレームレート
            enabled_ocr_engines: 有効にするOCRエンジンのリスト
            use_improved_validator: TemporalValidatorV2を使用するか（デフォルト: True）
            base_tolerance_seconds: ベース許容範囲（秒、TemporalValidatorV2用）
            history_size: 履歴サイズ（TemporalValidatorV2用）
            z_score_threshold: Z-score閾値（TemporalValidatorV2用）
            use_weighted_consensus: 重み付けスキームを使用するか（デフォルト: False）
            use_voting_consensus: 投票ロジックを使用するか（デフォルト: False）
        """
        self.roi_extractor = TimestampROIExtractor(roi_config)
        self.ocr_engine = MultiEngineOCR(
            enabled_engines=enabled_ocr_engines,
            use_weighted_consensus=use_weighted_consensus,
            use_voting_consensus=use_voting_consensus,
        )
        self.parser = TimestampParser()

        # 改善されたバリデーターを使用するか選択
        self.validator: TimestampValidator
        if use_improved_validator:
            self.validator = TemporalValidatorV2(
                fps=fps,
                base_tolerance_seconds=base_tolerance_seconds,
                history_size=history_size,
                z_score_threshold=z_score_threshold,
            )
            logger.info("Using TemporalValidatorV2 (adaptive tolerance and outlier recovery)")
        else:
            # デフォルトはV2を使用（旧版は削除）
            self.validator = TemporalValidatorV2(
                fps=fps,
                base_tolerance_seconds=base_tolerance_seconds,
                history_size=history_size,
                z_score_threshold=z_score_threshold,
            )
            logger.info("Using TemporalValidatorV2 (default)")

        self.confidence_threshold = confidence_threshold

        # OCR結果のキャッシュ（フレームハッシュをキーに）
        # maxsize=256: 最近の256フレーム分のOCR結果をキャッシュ
        self._ocr_cache: dict[str, tuple[str | None, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.Lock()  # スレッドセーフなキャッシュアクセス用

    def _compute_frame_hash(self, roi: np.ndarray) -> str:
        """ROI画像のハッシュ値を計算（キャッシュキー用）

        Args:
            roi: ROI画像

        Returns:
            MD5ハッシュ値（16進数文字列）
        """
        # ROI画像をバイト列に変換してハッシュ化
        roi_bytes = roi.tobytes()
        return hashlib.md5(roi_bytes).hexdigest()

    def extract(self, frame: np.ndarray, frame_idx: int, retry_count: int = 3) -> dict[str, Any] | None:
        """フレームからタイムスタンプを抽出（キャッシュ対応）

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

        # キャッシュキーを計算
        roi_hash = self._compute_frame_hash(roi)

        # キャッシュからOCR結果を取得（スレッドセーフ）
        with self._cache_lock:
            cached_result = self._ocr_cache.get(roi_hash)
            if cached_result is not None:
                ocr_text, ocr_confidence = cached_result
                self._cache_hits += 1
                logger.debug(f"Frame {frame_idx}: OCR結果をキャッシュから取得 (hash={roi_hash[:8]}...)")
            else:
                cached_result = None

        if cached_result is None:
            # キャッシュにない場合はOCR実行
            with self._cache_lock:
                self._cache_misses += 1
            ocr_text, ocr_confidence = None, 0.0

            for attempt in range(retry_count):
                try:
                    # 前処理
                    preprocessed = self.roi_extractor.preprocess_roi(roi)

                    # OCR実行
                    ocr_text, ocr_confidence = self.ocr_engine.extract_with_consensus(preprocessed)

                    # キャッシュに保存（成功した場合のみ、スレッドセーフ）
                    if ocr_text is not None:
                        with self._cache_lock:
                            self._ocr_cache[roi_hash] = (ocr_text, ocr_confidence)
                            # キャッシュサイズを制限（LRU方式で古いものを削除）
                            if len(self._ocr_cache) > 256:
                                # 最も古いエントリを削除（簡易的なLRU）
                                oldest_key = next(iter(self._ocr_cache))
                                del self._ocr_cache[oldest_key]
                        break
                except Exception as e:
                    logger.error(f"Frame {frame_idx}: OCR error (attempt {attempt + 1}/{retry_count}): {e}")
                    continue
        else:
            ocr_text, ocr_confidence = cached_result

        # OCR結果が取得できた場合、パースと時系列検証を実行
        if ocr_text is None:
            logger.warning(f"Frame {frame_idx}: OCR failed after {retry_count} attempts")
            return None

        for attempt in range(retry_count):
            try:
                # パース
                timestamp, parse_confidence = self.parser.fuzzy_parse(ocr_text)

                if timestamp is None:
                    logger.warning(
                        f"Frame {frame_idx}: Parse failed for '{ocr_text}' (attempt {attempt + 1}/{retry_count})"
                    )
                    continue

                # 時系列検証
                is_valid, temporal_confidence, reason = self.validator.validate(timestamp, frame_idx)

                # 総合信頼度
                total_confidence = (ocr_confidence + parse_confidence + temporal_confidence) / 3

                if total_confidence >= self.confidence_threshold and is_valid:
                    logger.info(f"Frame {frame_idx}: {timestamp} (confidence={total_confidence:.2f})")
                    return {
                        "timestamp": timestamp,
                        "frame_idx": frame_idx,
                        "confidence": total_confidence,
                        "ocr_text": ocr_text,
                        "roi_coords": roi_coords,
                    }
                # デバッグ情報を詳細に出力（時系列検証の失敗原因を確認）
                if not is_valid:
                    logger.warning(
                        f"Frame {frame_idx}: Temporal validation failed - {reason}, "
                        f"confidence={total_confidence:.2f}, "
                        f"threshold={self.confidence_threshold:.2f}"
                    )
                else:
                    logger.debug(
                        f"Frame {frame_idx}: Low confidence ({total_confidence:.2f}), valid={is_valid}, {reason}"
                    )
            except Exception as e:
                logger.error(f"Frame {frame_idx}: Error during extraction (attempt {attempt + 1}/{retry_count}): {e}")

        logger.error(f"Frame {frame_idx}: Failed after {retry_count} attempts")
        return None

    def reset_validator(self) -> None:
        """時系列検証器をリセット"""
        self.validator.reset()

    def clear_cache(self) -> None:
        """OCRキャッシュをクリア"""
        cache_size = len(self._ocr_cache)
        self._ocr_cache.clear()
        logger.info(f"OCRキャッシュをクリアしました（{cache_size}エントリ）")

    def get_cache_stats(self) -> dict:
        """キャッシュ統計情報を取得

        Returns:
            キャッシュ統計情報の辞書
        """
        with self._cache_lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0.0
            return {
                "cache_size": len(self._ocr_cache),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": hit_rate,
            }

    def extract_batch_parallel(
        self,
        frames: list[tuple[int, np.ndarray]],
        max_workers: int = 4,
        retry_count: int = 3,
    ) -> list[dict[str, Any] | None]:
        """複数フレームからタイムスタンプを並列抽出

        Args:
            frames: (frame_idx, frame) のリスト
            max_workers: 並列ワーカー数（デフォルト: 4）
            retry_count: リトライ回数

        Returns:
            抽出結果のリスト（入力順）。失敗した場合はNone
        """
        results: list[dict[str, Any] | None] = [None] * len(frames)

        def extract_single(idx_frame_tuple: tuple[int, tuple[int, np.ndarray]]) -> tuple[int, dict[str, Any] | None]:
            """単一フレームを抽出（インデックス付き）"""
            list_idx, (frame_idx, frame) = idx_frame_tuple
            result = self.extract(frame, frame_idx, retry_count=retry_count)
            return list_idx, result

        # 並列実行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_single, (i, frame_tuple)): i for i, frame_tuple in enumerate(frames)}

            for future in as_completed(futures):
                try:
                    list_idx, result = future.result()
                    results[list_idx] = result
                except Exception as e:
                    list_idx = futures[future]
                    logger.error(f"並列抽出エラー（インデックス {list_idx}）: {e}")
                    results[list_idx] = None

        return results
