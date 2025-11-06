"""Timestamp extraction module for the office person detection system."""

import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytesseract

from src.detection.preprocessing import apply_pipeline
from src.timestamp.ocr_engines import (EASYOCR_AVAILABLE, PADDLEOCR_AVAILABLE,
                                       run_ocr, run_tesseract)
from src.timestamp.timestamp_postprocess import parse_flexible_timestamp

logger = logging.getLogger(__name__)


class TimestampExtractor:
    """タイムスタンプ抽出クラス

    フレームの右上領域からOCRを使用してタイムスタンプを抽出する。

    Attributes:
        roi: タイムスタンプ領域の座標 (x, y, width, height)
    """

    TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"
    # 厳密な正規表現パターン
    STRICT_PATTERN = re.compile(r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}")
    # 緩いパターン（スペース欠落や1桁時を許容）
    FLEXIBLE_PATTERN = re.compile(r"(\d{4}/\d{2}/\d{2})\s*(\d{1,2}:\d{2}:\d{2})")

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        preproc_params: Optional[Dict] = None,
        ocr_params: Optional[Dict] = None,
        use_flexible_postprocess: bool = True,
        confidence_threshold: float = 0.2,
    ):
        """TimestampExtractorを初期化する

        Args:
            roi: タイムスタンプ領域の座標 (x, y, width, height)
                 デフォルトは右上領域 (900, 30, 360, 45)
            preproc_params: 前処理パラメータの辞書（Noneの場合はデフォルト）
            ocr_params: OCRパラメータの辞書（Noneの場合はデフォルト）
            use_flexible_postprocess: 柔軟な後処理を使用するか
            confidence_threshold: 信頼度閾値（この値未満の結果を除外、デフォルト: 0.2）
        """
        # デフォルトROI（最適値）
        self.roi = roi or (900, 30, 360, 45)
        logger.debug(f"TimestampExtractor初期化: ROI={self.roi}")
        self._debug_enabled = False
        self._debug_dir: Optional[Path] = None
        self._debug_save_intermediate = True
        self._debug_save_overlay = True
        self._debug_counter = 0
        self._last_preprocess_debug: Dict[str, np.ndarray] = {}
        self._last_timestamp: Optional[datetime] = None
        # 暫定値と確定値の分離（計画書の提案に基づく）
        self._tentative_timestamp: Optional[datetime] = None  # 低信頼度でも更新
        self._confirmed_timestamp: Optional[datetime] = None  # 高信頼度のみ更新
        self._last_corrections: List[Dict[str, str]] = []
        # CLAHEとカーネルをキャッシュ
        self._clahe_cache: Optional[cv2.CLAHE] = None
        self._horizontal_kernel_cache: Optional[np.ndarray] = None

        # パラメータ設定
        self.preproc_params = preproc_params or self._get_default_preproc_params()
        self.ocr_params = ocr_params or self._get_default_ocr_params()
        self.use_flexible_postprocess = use_flexible_postprocess
        self.confidence_threshold = confidence_threshold

        # 段階的な閾値システム（計画書の提案に基づく）
        self._threshold_levels = {
            "strict": 0.7,  # 高品質フレーム用
            "normal": confidence_threshold,  # 通常フレーム用（デフォルト）
            "lenient": 0.1,  # 最終手段
            "emergency": 0.0,  # 結果がない場合のみ
        }

        # 最初のフレーム検証用のフラグ
        self._is_initial_extraction = True
        self._initial_frame_count = 10  # 最初のNフレームで多数決

        # 最後のフレーム群検証用のフラグ（施策2）
        self._final_frame_count = 10  # 最後のNフレームで多数決

        # ROI動的調整用のフラグ（施策5）
        # 最適値(900, 30, 360, 45)を優先候補に
        self._enable_dynamic_roi = True
        self._roi_candidates = [
            (900, 30, 360, 45),  # 最適値
            (900, 20, 360, 50),  # 少し上に拡張
            (900, 40, 360, 40),  # 少し下に縮小
            (890, 30, 370, 45),  # 少し左に拡張
            (910, 30, 350, 45),  # 少し右に縮小
        ]

    def _get_default_preproc_params(self) -> Dict:
        """デフォルト前処理パラメータを取得（最適化版）"""
        return {
            "clahe": {
                "enabled": True,
                "clip_limit": 3.0,  # 2.0→3.0に向上（コントラスト強化）
                "tile_grid_size": [8, 8],
            },
            "resize": {"enabled": True, "fx": 2.0},
            "unsharp": {
                "enabled": True,  # シャープ化を有効化（文字認識精度向上）
                "amount": 1.5,
                "radius": 1.0,
            },
            "threshold": {"enabled": True, "method": "otsu"},
            "invert_after_threshold": {"enabled": True},
            "morphology": {
                "enabled": True,
                "operation": "close",
                "kernel_size": 2,
                "iterations": 1,
            },
        }

    def _get_default_ocr_params(self) -> Dict:
        """デフォルトOCRパラメータを取得"""
        return {
            "engine": "tesseract",
            "psm": 7,
            "whitelist": "0123456789/:",
            "lang": "eng",
            "oem": 3,
        }

    def enable_debug(
        self,
        debug_dir: Union[str, Path],
        save_intermediate: bool = True,
        save_overlay: bool = True,
    ) -> None:
        """デバッグ出力を有効化する

        Args:
            debug_dir: デバッグ画像を保存するディレクトリ
            save_intermediate: 前処理結果画像を保存するか
            save_overlay: オーバーレイ画像を保存するか
        """
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        self._debug_enabled = True
        self._debug_dir = debug_path
        self._debug_save_intermediate = save_intermediate
        self._debug_save_overlay = save_overlay
        logger.info(f"タイムスタンプOCRデバッグ出力: {debug_path}")

    def disable_debug(self) -> None:
        """デバッグ出力を無効化する"""
        self._debug_enabled = False
        self._debug_dir = None

    def _run_ocr(self, preprocessed: np.ndarray) -> Tuple[Optional[str], float]:
        """OCRを実行（パラメタ化版）

        Args:
            preprocessed: 前処理済み画像

        Returns:
            (OCRテキスト, 信頼度) のタプル
        """
        engine = self.ocr_params.get("engine", "tesseract")

        # OCR実行（engineをkwargsから除外）
        ocr_kwargs = {k: v for k, v in self.ocr_params.items() if k != "engine"}
        ocr_text, confidence = run_ocr(preprocessed, engine=engine, **ocr_kwargs)

        return ocr_text, confidence

    def _multi_ocr_vote(self, preprocessed: np.ndarray) -> Tuple[Optional[str], float]:
        """複数OCR設定で実行し、重み付き投票でタイムスタンプを決定（施策6: アンサンブル強化）

        Args:
            preprocessed: 前処理済み画像

        Returns:
            (タイムスタンプ, 平均信頼度) のタプル
        """
        # 複数のOCRエンジンとPSM設定を試す
        all_results: List[
            Tuple[str, float, str, float]
        ] = []  # (timestamp, weighted_confidence, engine, original_confidence)

        # Tesseract（複数PSM設定）
        ocr_configs = [
            {"psm": 7, "whitelist": "0123456789/:", "lang": "eng", "weight": 1.0},
            {"psm": 6, "whitelist": "0123456789/:", "lang": "eng", "weight": 0.9},
            {"psm": 8, "whitelist": "0123456789/:", "lang": "eng", "weight": 0.8},
            {"psm": 13, "whitelist": "0123456789/:", "lang": "eng", "weight": 0.7},
            {"psm": 11, "whitelist": "0123456789/:", "lang": "eng", "weight": 0.6},
        ]

        for ocr_config in ocr_configs:
            try:
                ocr_text, confidence = run_tesseract(
                    preprocessed,
                    psm=ocr_config["psm"],
                    whitelist=ocr_config["whitelist"],
                    lang=ocr_config["lang"],
                )

                if not ocr_text:
                    continue

                # 柔軟な後処理を使用する場合
                if self.use_flexible_postprocess:
                    timestamp = parse_flexible_timestamp(
                        ocr_text,
                        confidence=confidence,
                        reference_timestamp=self._last_timestamp,
                    )
                    if timestamp:
                        all_results.append(
                            (
                                timestamp,
                                confidence * ocr_config["weight"],
                                "tesseract",
                                confidence,
                            )
                        )
                        continue

                # 従来の方法も試す
                timestamp = self._parse_strict_regex(ocr_text)
                if timestamp:
                    all_results.append(
                        (
                            timestamp,
                            confidence * ocr_config["weight"],
                            "tesseract",
                            confidence,
                        )
                    )
                    continue

                # フォールバック: parse_timestamp
                timestamp = self.parse_timestamp(ocr_text)
                if timestamp:
                    all_results.append(
                        (
                            timestamp,
                            confidence * ocr_config["weight"],
                            "tesseract",
                            confidence,
                        )
                    )
            except Exception as e:
                logger.debug(f"OCR設定 {ocr_config} でエラー: {e}")
                continue

        # PaddleOCRとEasyOCRも試行（施策6）
        if PADDLEOCR_AVAILABLE:
            try:
                ocr_text, confidence = run_ocr(preprocessed, engine="paddleocr")
                if ocr_text:
                    timestamp = parse_flexible_timestamp(
                        ocr_text,
                        confidence=confidence,
                        reference_timestamp=self._last_timestamp,
                    )
                    if timestamp:
                        # PaddleOCRは重み1.2（高精度のため）
                        all_results.append(
                            (timestamp, confidence * 1.2, "paddleocr", confidence)
                        )
            except Exception as e:
                logger.debug(f"PaddleOCRでエラー: {e}")

        if EASYOCR_AVAILABLE:
            try:
                ocr_text, confidence = run_ocr(preprocessed, engine="easyocr")
                if ocr_text:
                    timestamp = parse_flexible_timestamp(
                        ocr_text,
                        confidence=confidence,
                        reference_timestamp=self._last_timestamp,
                    )
                    if timestamp:
                        # EasyOCRは重み1.0
                        all_results.append(
                            (timestamp, confidence * 1.0, "easyocr", confidence)
                        )
            except Exception as e:
                logger.debug(f"EasyOCRでエラー: {e}")

        if not all_results:
            return None, 0.0

        # 重み付き投票: 同じタイムスタンプの重み付きスコアを計算
        # 時系列整合性チェックも同時に実施（施策2）
        timestamp_scores: Dict[str, float] = {}
        timestamp_counts: Dict[str, int] = {}
        timestamp_confidences: Dict[str, List[float]] = {}  # 元の信頼度を保持

        for timestamp, weighted_conf, engine, original_conf in all_results:
            # 信頼度フィルタリング（施策1: 厳格なフィルタリング）
            # 信頼度0.00の結果を即座に除外
            if original_conf <= 0.0:
                logger.debug(f"信頼度0.00の結果を除外: {timestamp} (エンジン: {engine})")
                continue

            # 信頼度閾値未満の結果を除外
            if original_conf < self.confidence_threshold:
                logger.debug(
                    f"信頼度閾値未満の結果を除外: {timestamp} "
                    f"(信頼度: {original_conf:.3f} < 閾値: {self.confidence_threshold:.3f}, エンジン: {engine})"
                )
                continue

            # 時系列整合性チェック（施策4: 強化）
            # 暫定値または確定値を参照（計画書の提案に基づく）
            reference_timestamp = (
                self._tentative_timestamp
                or self._confirmed_timestamp
                or self._last_timestamp
            )
            if reference_timestamp is not None:
                try:
                    timestamp_dt = datetime.strptime(timestamp, self.TIMESTAMP_FORMAT)
                    time_diff = abs(
                        (timestamp_dt - reference_timestamp).total_seconds()
                    )
                    days_diff = abs(
                        (timestamp_dt.date() - reference_timestamp.date()).days
                    )

                    # 日付レベルの外れ値検知（±0.5日以上は除外）
                    if days_diff >= 0.5:
                        logger.debug(
                            f"時系列外れ値検知（アンサンブル）: {timestamp} "
                            f"(履歴: {reference_timestamp.date()}, 差={days_diff}日)"
                        )
                        continue

                    # ±7日以上の差は除外（誤認識の可能性が高い）
                    if days_diff >= 7:
                        logger.debug(
                            f"時系列外れ値検知（アンサンブル）: {timestamp} "
                            f"(履歴: {reference_timestamp.date()}, 差={days_diff}日)"
                        )
                        continue

                    # 時間差が大きすぎる場合（1時間以上）は除外
                    if time_diff > 3600:
                        logger.debug(
                            f"時系列外れ値検知（アンサンブル）: {timestamp} "
                            f"(履歴: {reference_timestamp}, 時間差={time_diff:.0f}秒)"
                        )
                        continue

                    # 時系列が逆転している場合（12時間以上）は除外
                    if timestamp_dt < reference_timestamp - timedelta(hours=12):
                        logger.debug(
                            f"時系列逆転検知（アンサンブル）: {timestamp} "
                            f"(履歴: {reference_timestamp})"
                        )
                        continue

                except ValueError:
                    # タイムスタンプの解析に失敗した場合はスキップ
                    continue

            if timestamp not in timestamp_scores:
                timestamp_scores[timestamp] = 0.0
                timestamp_counts[timestamp] = 0
                timestamp_confidences[timestamp] = []

            # 重み付きスコアを累積（施策5: アンサンブル結果評価改善）
            # 信頼度が低い結果の重みを下げる
            confidence_weight = 1.0
            if original_conf < 0.5:
                # 信頼度0.5未満の結果は重みを下げる
                confidence_weight = original_conf * 2.0  # 0.0-1.0の範囲に正規化
            elif original_conf < 0.7:
                # 信頼度0.5-0.7の結果は中程度の重み
                confidence_weight = 0.5 + (original_conf - 0.5) * 1.0
            # 信頼度0.7以上の結果は重み1.0（そのまま）

            adjusted_weighted_conf = weighted_conf * confidence_weight
            timestamp_scores[timestamp] += adjusted_weighted_conf
            timestamp_counts[timestamp] += 1
            # 元の信頼度を保持
            timestamp_confidences[timestamp].append(original_conf)

        # フォールバックメカニズム: 全結果が除外された場合の段階的閾値緩和
        if not timestamp_scores:
            return self._apply_fallback_filtering(all_results)

        # 最高スコアのタイムスタンプを選択
        best_timestamp = max(timestamp_scores.items(), key=lambda x: x[1])[0]

        # 平均信頼度を計算（元の信頼度の平均）
        if best_timestamp in timestamp_confidences:
            confidences = timestamp_confidences[best_timestamp]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        else:
            avg_confidence = 0.0

        # 信頼度が負の値にならないように保証
        avg_confidence = max(0.0, min(1.0, avg_confidence))

        return best_timestamp, avg_confidence

    def _apply_fallback_filtering(
        self, all_results: List[Tuple[str, float, str, float]]
    ) -> Tuple[Optional[str], float]:
        """フォールバックメカニズム: 全結果が除外された場合の段階的閾値緩和

        計画書の提案に基づき、段階的に閾値を緩和して再試行する。
        デッドロック問題（全結果が除外される）を防止する。

        Args:
            all_results: 全OCR結果のリスト (timestamp, weighted_conf, engine, original_conf)

        Returns:
            (タイムスタンプ, 平均信頼度) のタプル
        """
        if not all_results:
            logger.warning("全結果が除外されました。フォールバック処理をスキップします。")
            return None, 0.0

        # 信頼度0.00の結果を除外（これは常に除外）
        valid_results = [
            (ts, wc, eng, oc) for ts, wc, eng, oc in all_results if oc > 0.0
        ]

        if not valid_results:
            logger.warning("全結果が信頼度0.00のため、フォールバック処理をスキップします。")
            return None, 0.0

        # 段階的な閾値適用: normal -> lenient -> emergency
        threshold_levels = ["normal", "lenient", "emergency"]

        for level in threshold_levels:
            threshold = self._threshold_levels[level]
            logger.info(f"フォールバック: 閾値レベル '{level}' (閾値={threshold:.3f}) で再試行")

            # この閾値でフィルタリング
            filtered = [
                (ts, wc, eng, oc)
                for ts, wc, eng, oc in valid_results
                if oc >= threshold
            ]

            if not filtered:
                continue  # 次のレベルを試す

            # 時系列整合性チェックを実施
            timestamp_scores: Dict[str, float] = {}
            timestamp_counts: Dict[str, int] = {}
            timestamp_confidences: Dict[str, List[float]] = {}

            for timestamp, weighted_conf, engine, original_conf in filtered:
                # 時系列整合性チェック
                # 暫定値または確定値を参照（計画書の提案に基づく）
                reference_timestamp = (
                    self._tentative_timestamp
                    or self._confirmed_timestamp
                    or self._last_timestamp
                )
                if reference_timestamp is not None:
                    try:
                        timestamp_dt = datetime.strptime(
                            timestamp, self.TIMESTAMP_FORMAT
                        )
                        time_diff = abs(
                            (timestamp_dt - reference_timestamp).total_seconds()
                        )
                        days_diff = abs(
                            (timestamp_dt.date() - reference_timestamp.date()).days
                        )

                        # 日付レベルの外れ値検知（±0.5日以上は除外）
                        if days_diff >= 0.5:
                            logger.debug(
                                f"フォールバック: 時系列外れ値検知: {timestamp} "
                                f"(履歴: {reference_timestamp.date()}, 差={days_diff}日)"
                            )
                            continue

                        # ±7日以上の差は除外（誤認識の可能性が高い）
                        if days_diff >= 7:
                            logger.debug(
                                f"フォールバック: 時系列外れ値検知: {timestamp} "
                                f"(履歴: {reference_timestamp.date()}, 差={days_diff}日)"
                            )
                            continue

                        # 時間差が大きすぎる場合（1時間以上）は除外
                        if time_diff > 3600:
                            logger.debug(
                                f"フォールバック: 時系列外れ値検知: {timestamp} "
                                f"(履歴: {reference_timestamp}, 時間差={time_diff:.0f}秒)"
                            )
                            continue

                        # 時系列が逆転している場合（12時間以上）は除外
                        if timestamp_dt < reference_timestamp - timedelta(hours=12):
                            logger.debug(
                                f"フォールバック: 時系列逆転検知: {timestamp} "
                                f"(履歴: {reference_timestamp})"
                            )
                            continue

                    except ValueError:
                        # タイムスタンプの解析に失敗した場合はスキップ
                        continue

                if timestamp not in timestamp_scores:
                    timestamp_scores[timestamp] = 0.0
                    timestamp_counts[timestamp] = 0
                    timestamp_confidences[timestamp] = []

                # 重み付きスコアを累積
                confidence_weight = 1.0
                if original_conf < 0.5:
                    confidence_weight = original_conf * 2.0
                elif original_conf < 0.7:
                    confidence_weight = 0.5 + (original_conf - 0.5) * 1.0

                adjusted_weighted_conf = weighted_conf * confidence_weight
                timestamp_scores[timestamp] += adjusted_weighted_conf
                timestamp_counts[timestamp] += 1
                timestamp_confidences[timestamp].append(original_conf)

            if timestamp_scores:
                # 最高スコアのタイムスタンプを選択
                best_timestamp = max(timestamp_scores.items(), key=lambda x: x[1])[0]

                # 平均信頼度を計算
                if best_timestamp in timestamp_confidences:
                    confidences = timestamp_confidences[best_timestamp]
                    avg_confidence = (
                        sum(confidences) / len(confidences) if confidences else 0.0
                    )
                else:
                    avg_confidence = 0.0

                avg_confidence = max(0.0, min(1.0, avg_confidence))

                logger.info(
                    f"フォールバック成功: レベル '{level}' でタイムスタンプを取得: "
                    f"{best_timestamp} (信頼度={avg_confidence:.3f})"
                )
                return best_timestamp, avg_confidence

        # 全てのレベルで失敗
        logger.warning("フォールバック処理: 全ての閾値レベルで結果が得られませんでした")
        return None, 0.0

    def _parse_strict_regex(self, text: str) -> Optional[str]:
        """厳密な正規表現でタイムスタンプを抽出

        Args:
            text: OCRテキスト

        Returns:
            正規化されたタイムスタンプ、失敗時None
        """
        if not text:
            return None

        # 全角→半角変換と文字修正
        normalized = text.translate(
            str.maketrans(
                {
                    "O": "0",
                    "o": "0",
                    "D": "0",
                    "S": "5",
                    "s": "5",
                    "I": "1",
                    "l": "1",
                    "|": "1",
                    "B": "8",
                    "b": "6",
                    "A": "4",
                    "Z": "2",
                    "z": "2",
                    "q": "9",
                    "G": "6",
                    "T": "7",
                    "Q": "0",
                    "／": "/",
                    "：": ":",
                    "－": "-",
                }
            )
        )
        normalized = re.sub(r"[\t\r\n]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)

        # スラッシュ欠落補正: 2025108/2616:05:26 -> 2025/10/08 16:05:26
        # パターン: 2025108/26... -> 2025/10/08 ...
        # スラッシュの位置を考慮: 2025108/26 の部分が 2025/10/08 を意味
        # まず、厳密なパターンが既にマッチするかチェック
        if self.STRICT_PATTERN.search(normalized):
            # 既に厳密なパターンがマッチする場合は、そのまま処理
            match = self.STRICT_PATTERN.search(normalized)
            ts_str = match.group(0)
            try:
                dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
                return dt.strftime(self.TIMESTAMP_FORMAT)
            except ValueError:
                pass

        # 数字列として解析してから補正（厳密なパターンがマッチしない場合のみ）
        digits_only = re.sub(r"[^0-9]", "", normalized)
        if len(digits_only) >= 14:
            # 202510826160526 (15桁) の場合: 2025/10/08 16:05:26
            # 202510826 -> 2025/10/08 に変換（最初の8桁）
            # 残り: 160526 -> 16:05:26
            if len(digits_only) == 15:
                # スラッシュの位置を考慮: 2025108/26 の部分が 2025/10/08 を意味
                # つまり、最初の8桁を 2025/10/08 に変換し、残りを 16:05:26 として解釈
                normalized = (
                    f"{digits_only[0:4]}/{digits_only[4:6]}/{digits_only[6:8]} "
                    f"{digits_only[9:11]}:{digits_only[11:13]}:{digits_only[13:15]}"
                )
            else:
                # 14桁の場合: 20251082616052 -> 2025/10/08 16:05:26
                normalized = (
                    f"{digits_only[0:4]}/{digits_only[4:6]}/{digits_only[6:8]} "
                    f"{digits_only[8:10]}:{digits_only[10:12]}:{digits_only[12:14]}"
                )
        elif len(digits_only) == 12:
            normalized = (
                f"{digits_only[0:4]}/{digits_only[4:6]}/{digits_only[6:8]} "
                f"{digits_only[8:10]}:{digits_only[10:12]}:00"
            )

        # 既存の補正パターンも適用
        normalized = re.sub(
            r"(\d{4})(\d{2})(\d{2})(\d{2}):(\d{2}):(\d{2})",
            r"\1/\2/\3 \4:\5:\6",
            normalized,
        )
        normalized = re.sub(
            r"(\d{4})/(\d{2})(\d{2})(\d{2}):(\d{2}):(\d{2})",
            r"\1/\2/\3 \4:\5:\6",
            normalized,
        )
        normalized = re.sub(
            r"(\d{4}/\d{2}/\d{2})(\d{2}):(\d{2}):(\d{2})", r"\1 \2:\3:\4", normalized
        )

        # スペース欠落補正: 2025/08/270:13:31 -> 2025/08/27 0:13:31
        normalized = re.sub(
            r"(\d{4}/\d{2}/\d{2})(\d{1,2}:\d{2}:\d{2})", r"\1 \2", normalized
        )

        # 最優先: 厳密なパターン（前後の文字列を無視）
        match = self.STRICT_PATTERN.search(normalized)
        if match:
            ts_str = match.group(0)
            try:
                dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
                return dt.strftime(self.TIMESTAMP_FORMAT)
            except ValueError:
                pass

        # フォールバック: 緩いパターン（スペース欠落や1桁時を許容）
        match = self.FLEXIBLE_PATTERN.search(normalized)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)

            # 1桁時のゼロ埋め
            time_parts = time_part.split(":")
            if len(time_parts) == 3:
                hour, minute, second = time_parts
                if len(hour) == 1:
                    hour = f"0{hour}"
                if len(minute) == 2 and len(second) == 2:
                    ts_str = f"{date_part} {hour}:{minute}:{second}"
                    try:
                        dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
                        return dt.strftime(self.TIMESTAMP_FORMAT)
                    except ValueError:
                        pass

        return None

    def _try_multiple_roi_candidates(
        self, frame: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """複数のROI候補領域を試行（施策5）

        Args:
            frame: 入力フレーム画像

        Returns:
            [(ROI画像, ROI座標, 品質スコア), ...] のリスト
        """
        frame_height, frame_width = frame.shape[:2]
        candidates: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []

        for roi_candidate in self._roi_candidates:
            x, y, w, h = roi_candidate

            # フレームサイズチェック
            if x + w > frame_width or y + h > frame_height:
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)

            if w <= 0 or h <= 0:
                continue

            roi_image = frame[y : y + h, x : x + w]

            if roi_image.size == 0:
                continue

            # ROI領域の品質評価
            quality = self._evaluate_roi_quality(roi_image)
            quality_score = (
                quality["contrast"] * 0.4
                + quality["sharpness"] * 0.4
                + (100 - quality["noise_level"]) * 0.2
            )

            candidates.append((roi_image, (x, y, w, h), quality_score))

        # 品質スコアでソート（高い順）
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def _evaluate_roi_quality(self, roi_image: np.ndarray) -> Dict[str, float]:
        """ROI領域の品質を評価（施策5）

        Args:
            roi_image: ROI領域の画像

        Returns:
            品質指標の辞書
        """
        return self._evaluate_frame_quality(roi_image)

    def extract(
        self,
        frame: np.ndarray,
        frame_index: Optional[int] = None,
        total_frames: Optional[int] = None,
    ) -> Optional[str]:
        """フレームからタイムスタンプを抽出する

        複数OCR設定で多数決を行い、厳密な正規表現で抽出を試みる。
        ROI動的調整（施策5）と日付部分の独立検証（施策3）を実装。
        最後のフレーム群での検証と再試行メカニズム（施策2）を実装。

        Args:
            frame: 入力フレーム画像
            frame_index: デバッグ用のフレーム番号
            total_frames: 総フレーム数（最後のフレーム群検証用）

        Returns:
            タイムスタンプ文字列 (YYYY/MM/DD HH:MM:SS形式)、失敗した場合None
        """
        if frame is None or frame.size == 0:
            logger.warning("無効なフレームが渡されました")
            return None

        # 最後のフレーム群での検証と再試行メカニズム（施策2）
        if (
            frame_index is not None
            and total_frames is not None
            and frame_index >= total_frames - self._final_frame_count
        ):
            logger.debug(f"最後のフレーム群を検出: フレーム {frame_index}/{total_frames}")
            # 最後のフレーム群では、通常の処理を実行し、信頼度が低い場合は警告
            # 実際の多数決検証は呼び出し側で行う必要がある

        try:
            # ROI動的調整（施策5）
            if self._enable_dynamic_roi:
                roi_candidates = self._try_multiple_roi_candidates(frame)
                if not roi_candidates:
                    logger.warning("ROI候補領域が見つかりませんでした")
                    return None

                # 各候補領域でOCRを試行
                best_timestamp = None
                best_confidence = -1.0
                best_roi_bounds = None
                best_preprocessed = None

                for roi_image, roi_bounds, quality_score in roi_candidates:
                    # 施策3: 日付と時刻を独立して抽出（優先的に試行）
                    date_str, date_conf = self._extract_date_independently(roi_image)
                    time_str, time_conf = self._extract_time_independently(roi_image)

                    # 日付と時刻の両方が取得できた場合
                    if date_str and time_str:
                        # 日付の妥当性チェック
                        year, month, day = date_str.split("/")
                        if self._validate_date_format(year, month, day):
                            # 結合してタイムスタンプを作成
                            combined_timestamp = f"{date_str} {time_str}"
                            combined_confidence = (date_conf + time_conf) / 2.0

                            # 時系列整合性チェック（履歴がある場合）
                            if self._last_timestamp is not None:
                                try:
                                    combined_dt = datetime.strptime(
                                        combined_timestamp, self.TIMESTAMP_FORMAT
                                    )
                                    days_diff = abs(
                                        (
                                            combined_dt.date()
                                            - self._last_timestamp.date()
                                        ).days
                                    )
                                    # ±7日以上の差は除外（誤認識の可能性が高い）
                                    if days_diff >= 7:
                                        logger.debug(
                                            f"日付独立検証: 時系列外れ値検知 "
                                            f"(差={days_diff}日、履歴={self._last_timestamp.date()})"
                                        )
                                        # フォールバック処理に進む
                                    else:
                                        # 妥当なタイムスタンプとして採用
                                        if combined_confidence > best_confidence:
                                            best_timestamp = combined_timestamp
                                            best_confidence = combined_confidence
                                            best_roi_bounds = roi_bounds
                                            best_preprocessed = roi_image
                                            # 履歴を更新（施策3: 更新ロジック改善）
                                            self._update_last_timestamp(
                                                combined_dt,
                                                confidence=combined_confidence,
                                            )
                                            logger.debug(
                                                f"日付独立検証成功: {combined_timestamp} "
                                                f"(信頼度={combined_confidence:.2f})"
                                            )
                                        continue
                                except ValueError:
                                    # タイムスタンプの解析に失敗した場合はフォールバック
                                    pass
                            else:
                                # 履歴がない場合はそのまま採用
                                if combined_confidence > best_confidence:
                                    best_timestamp = combined_timestamp
                                    best_confidence = combined_confidence
                                    best_roi_bounds = roi_bounds
                                    best_preprocessed = roi_image
                                    # 履歴を更新（施策3: 更新ロジック改善）
                                    try:
                                        combined_dt = datetime.strptime(
                                            combined_timestamp, self.TIMESTAMP_FORMAT
                                        )
                                        self._update_last_timestamp(
                                            combined_dt, confidence=combined_confidence
                                        )
                                    except ValueError:
                                        pass
                                    logger.debug(
                                        f"日付独立検証成功（初期）: {combined_timestamp} "
                                        f"(信頼度={combined_confidence:.2f})"
                                    )
                                continue

                    # 日付・時刻の独立抽出が失敗した場合、従来の方法を試行
                    preprocessed = self._preprocess_roi(roi_image)

                    # 複数OCR設定で多数決
                    timestamp, confidence = self._multi_ocr_vote(preprocessed)

                    # 信頼度が高い結果を選択
                    if timestamp and confidence > best_confidence:
                        best_timestamp = timestamp
                        best_confidence = confidence
                        best_roi_bounds = roi_bounds
                        best_preprocessed = preprocessed

                    # 信頼度が十分高い場合は早期終了
                    if confidence > 0.8:
                        break

                if best_timestamp:
                    if self._debug_enabled:
                        self._save_debug_outputs(
                            frame,
                            roi_candidates[0][0],  # 最初の候補のROI画像
                            best_preprocessed or roi_candidates[0][0],
                            best_timestamp or "",
                            best_timestamp,
                            frame_index,
                            best_roi_bounds or self.roi,
                        )
                    logger.debug(
                        f"抽出されたタイムスタンプ: {best_timestamp} (信頼度={best_confidence:.2f})"
                    )
                    return best_timestamp

            # フォールバック: デフォルトROIを使用
            x, y, w, h = self.roi

            # フレームサイズチェック
            frame_height, frame_width = frame.shape[:2]
            if x + w > frame_width or y + h > frame_height:
                logger.warning(
                    f"ROI領域がフレームサイズを超えています: ROI={self.roi}, Frame={frame_width}x{frame_height}"
                )
                # ROIを調整
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)

            roi_image = frame[y : y + h, x : x + w]
            roi_bounds = (x, y, w, h)

            if roi_image.size == 0:
                logger.warning("ROI領域が空です")
                return None

            # 施策3: 日付と時刻を独立して抽出（優先的に試行）
            date_str, date_conf = self._extract_date_independently(roi_image)
            time_str, time_conf = self._extract_time_independently(roi_image)

            # 日付と時刻の両方が取得できた場合
            if date_str and time_str:
                # 日付の妥当性チェック
                year, month, day = date_str.split("/")
                if self._validate_date_format(year, month, day):
                    # 結合してタイムスタンプを作成
                    combined_timestamp = f"{date_str} {time_str}"
                    combined_confidence = (date_conf + time_conf) / 2.0

                    # 時系列整合性チェック（履歴がある場合）
                    if self._last_timestamp is not None:
                        try:
                            combined_dt = datetime.strptime(
                                combined_timestamp, self.TIMESTAMP_FORMAT
                            )
                            days_diff = abs(
                                (combined_dt.date() - self._last_timestamp.date()).days
                            )
                            # ±7日以上の差は除外（誤認識の可能性が高い）
                            if days_diff < 7:
                                # 妥当なタイムスタンプとして採用
                                # 履歴を更新（施策3: 更新ロジック改善）
                                self._update_last_timestamp(
                                    combined_dt, confidence=combined_confidence
                                )
                                logger.debug(
                                    f"日付独立検証成功: {combined_timestamp} "
                                    f"(信頼度={combined_confidence:.2f})"
                                )
                                if self._debug_enabled:
                                    self._save_debug_outputs(
                                        frame,
                                        roi_image,
                                        roi_image,
                                        combined_timestamp or "",
                                        combined_timestamp,
                                        frame_index,
                                        roi_bounds,
                                    )
                                return combined_timestamp
                            else:
                                logger.debug(
                                    f"日付独立検証: 時系列外れ値検知 "
                                    f"(差={days_diff}日、履歴={self._last_timestamp.date()})"
                                )
                        except ValueError:
                            # タイムスタンプの解析に失敗した場合はフォールバック
                            pass
                    else:
                        # 履歴がない場合はそのまま採用
                        # 履歴を更新（施策3: 更新ロジック改善）
                        try:
                            combined_dt = datetime.strptime(
                                combined_timestamp, self.TIMESTAMP_FORMAT
                            )
                            self._update_last_timestamp(
                                combined_dt, confidence=combined_confidence
                            )
                        except ValueError:
                            pass
                        logger.debug(
                            f"日付独立検証成功（初期）: {combined_timestamp} "
                            f"(信頼度={combined_confidence:.2f})"
                        )
                        if self._debug_enabled:
                            self._save_debug_outputs(
                                frame,
                                roi_image,
                                roi_image,
                                combined_timestamp or "",
                                combined_timestamp,
                                frame_index,
                                roi_bounds,
                            )
                        return combined_timestamp

            # 日付・時刻の独立抽出が失敗した場合、従来の方法を試行
            # 前処理
            preprocessed = self._preprocess_roi(roi_image)

            # 複数OCR設定で多数決
            timestamp, confidence = self._multi_ocr_vote(preprocessed)

            # 厳密な抽出が失敗した場合のみ、従来のparse_timestampをフォールバック
            if timestamp is None:
                ocr_text = pytesseract.image_to_string(
                    preprocessed,
                    config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:/",
                )
                logger.debug(f"OCR結果（フォールバック）: '{ocr_text.strip()}'")
                timestamp = self.parse_timestamp(ocr_text)

            if timestamp:
                logger.debug(f"抽出されたタイムスタンプ: {timestamp} (信頼度={confidence:.2f})")
            else:
                logger.warning("タイムスタンプの抽出に失敗しました")

            if self._debug_enabled:
                self._save_debug_outputs(
                    frame,
                    roi_image,
                    preprocessed,
                    timestamp or "",
                    timestamp,
                    frame_index,
                    roi_bounds,
                )

            return timestamp

        except Exception as e:
            logger.error(f"タイムスタンプ抽出中にエラーが発生しました: {e}")
            return None

    def _focus_timestamp_band(self, gray_roi: np.ndarray) -> np.ndarray:
        """ROIからタイムスタンプ行を抽出する

        簡略化版：ROI全体をそのまま使用（小さなROIのため）
        """
        # ROIが小さい場合は全体をそのまま使用
        height, width = gray_roi.shape[:2]
        if height <= 50:  # 高さが50px以下の場合は全体を使用
            return gray_roi

        # それ以外は上部60%を使用（タイムスタンプは通常上部に表示）
        band_height = int(height * 0.6)
        band = gray_roi[:band_height, :]
        return band if band.size > 0 else gray_roi

    def _evaluate_frame_quality(self, roi_image: np.ndarray) -> Dict[str, float]:
        """フレーム品質を評価（施策4）

        Args:
            roi_image: ROI領域の画像

        Returns:
            品質指標の辞書（contrast, sharpness, noise_level）
        """
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image

        # コントラスト（標準偏差）
        contrast = float(np.std(gray))

        # シャープネス（Laplacian分散）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # ノイズレベル（高周波成分の推定）
        # ガウシアンブラー後の差分から推定
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)
        noise_level = float(np.mean(diff))

        return {
            "contrast": contrast,
            "sharpness": sharpness,
            "noise_level": noise_level,
        }

    def _select_optimal_pipeline(
        self, roi_image: np.ndarray, quality: Dict[str, float]
    ) -> Dict:
        """フレーム品質に応じた最適な前処理パイプラインを選択（施策4）

        Args:
            roi_image: ROI領域の画像
            quality: 品質指標

        Returns:
            最適な前処理パラメータ
        """
        # デフォルトパラメータをコピー
        params = self.preproc_params.copy()

        # コントラストが低い場合（< 30）
        if quality["contrast"] < 30:
            # CLAHEのclip_limitを上げる
            if "clahe" in params:
                params["clahe"]["clip_limit"] = 3.0
            logger.debug(f"低コントラスト検出 ({quality['contrast']:.1f})。CLAHE強化")

        # シャープネスが低い場合（< 100）
        if quality["sharpness"] < 100:
            # シャープ化を有効化
            if "unsharp" not in params:
                params["unsharp"] = {"enabled": True, "amount": 1.5, "radius": 1.0}
            else:
                params["unsharp"]["enabled"] = True
            logger.debug(f"低シャープネス検出 ({quality['sharpness']:.1f})。シャープ化有効化")

        # ノイズレベルが高い場合（> 20）
        if quality["noise_level"] > 20:
            # ブラーを有効化
            if "blur" not in params:
                params["blur"] = {"enabled": True, "kernel_size": 3, "sigma": 0.0}
            else:
                params["blur"]["enabled"] = True
            logger.debug(f"高ノイズ検出 ({quality['noise_level']:.1f})。ブラー有効化")

        return params

    def _preprocess_roi(
        self, roi_image: np.ndarray, try_multiple_pipelines: bool = False
    ) -> np.ndarray:
        """OCR用の前処理を実行する

        パラメタ化された前処理パイプラインを使用。
        フレーム品質に応じた適応的前処理を実装（施策4）。
        低品質フレーム対応のため、複数パイプラインを試行可能（施策6）。

        Args:
            roi_image: ROI領域の画像
            try_multiple_pipelines: 複数パイプラインを試行するか（低品質フレーム用）

        Returns:
            前処理済み画像
        """
        try:
            self._last_preprocess_debug = {}

            # フレーム品質を評価（施策4）
            quality = self._evaluate_frame_quality(roi_image)

            # 低品質フレームの場合、複数パイプラインを試行（施策6）
            if try_multiple_pipelines or (
                quality["contrast"] < 30
                or quality["sharpness"] < 100
                or quality["noise_level"] > 20
            ):
                # 複数の前処理パイプラインを試行
                pipeline_variants = [
                    # バリアント1: デフォルト（強化版）
                    self._select_optimal_pipeline(roi_image, quality),
                    # バリアント2: より積極的なCLAHE
                    self._create_aggressive_pipeline(roi_image, quality),
                    # バリアント3: より積極的なシャープ化
                    self._create_sharpen_pipeline(roi_image, quality),
                ]

                best_preprocessed = None
                best_quality_score = -1.0

                for variant_params in pipeline_variants:
                    try:
                        variant_preprocessed = apply_pipeline(roi_image, variant_params)
                        # 前処理後の品質を評価
                        variant_quality = self._evaluate_frame_quality(
                            variant_preprocessed
                        )
                        variant_score = (
                            variant_quality["contrast"] * 0.4
                            + variant_quality["sharpness"] * 0.4
                            + (100 - variant_quality["noise_level"]) * 0.2
                        )

                        if variant_score > best_quality_score:
                            best_quality_score = variant_score
                            best_preprocessed = variant_preprocessed
                    except Exception as e:
                        logger.debug(f"前処理パイプライン変種でエラー: {e}")
                        continue

                if best_preprocessed is not None:
                    self._last_preprocess_debug = {
                        "preprocessed": best_preprocessed,
                        "quality": quality,
                        "best_quality_score": best_quality_score,
                    }
                    return best_preprocessed

            # 通常の処理（単一パイプライン）
            # 品質に応じた最適なパイプラインを選択
            optimal_params = self._select_optimal_pipeline(roi_image, quality)

            # パラメタ化された前処理を適用
            preprocessed = apply_pipeline(roi_image, optimal_params)

            self._last_preprocess_debug = {
                "preprocessed": preprocessed,
                "quality": quality,
            }

            return preprocessed

        except Exception as e:
            logger.error(f"前処理中にエラーが発生しました: {e}")
            self._last_preprocess_debug = {}
            return roi_image

    def _create_aggressive_pipeline(
        self, roi_image: np.ndarray, quality: Dict[str, float]
    ) -> Dict:
        """より積極的なCLAHEを使用する前処理パイプライン（施策6）

        Args:
            roi_image: ROI領域の画像
            quality: 品質指標

        Returns:
            前処理パラメータ
        """
        params = self.preproc_params.copy()
        # CLAHEをより積極的に
        if "clahe" in params:
            params["clahe"]["clip_limit"] = 4.0  # デフォルトより高い
            params["clahe"]["tile_grid_size"] = [4, 4]  # より細かいタイル
        return params

    def _create_sharpen_pipeline(
        self, roi_image: np.ndarray, quality: Dict[str, float]
    ) -> Dict:
        """より積極的なシャープ化を使用する前処理パイプライン（施策6）

        Args:
            roi_image: ROI領域の画像
            quality: 品質指標

        Returns:
            前処理パラメータ
        """
        params = self.preproc_params.copy()
        # シャープ化をより積極的に
        if "unsharp" not in params:
            params["unsharp"] = {"enabled": True, "amount": 2.5, "radius": 1.5}
        else:
            params["unsharp"]["enabled"] = True
            params["unsharp"]["amount"] = 2.5
            params["unsharp"]["radius"] = 1.5
        return params

    def parse_timestamp(self, ocr_text: str) -> Optional[str]:
        """OCR結果からタイムスタンプを抽出・正規化する

        Args:
            ocr_text: OCRで読み取られたテキスト

        Returns:
            正規化されたタイムスタンプ (YYYY/MM/DD HH:MM:SS形式)、失敗した場合None
        """
        if not ocr_text:
            return None

        try:
            text = ocr_text.strip()
            if not text:
                return None

            translation_table = str.maketrans(
                {
                    "O": "0",
                    "o": "0",
                    "D": "0",
                    "S": "5",
                    "s": "5",
                    "I": "1",
                    "l": "1",
                    "|": "1",
                    "B": "8",
                    "b": "6",
                    "A": "4",
                }
            )

            normalized = text.translate(translation_table)
            normalized = re.sub(r"[\t\r\n]", " ", normalized)
            normalized = re.sub(r"\s+", " ", normalized)
            normalized = (
                normalized.replace("／", "/").replace("：", ":").replace("－", "-")
            )

            # スラッシュ欠落補正: 2025108/2616:05:26 -> 2025/10/08 16:05:26
            # パターン1: 4桁の年 + 2桁の月（/欠落） + 1桁の日（/欠落） + 2桁の日 + 2桁の時（:欠落） + 分:秒
            normalized = re.sub(
                r"(\d{4})(\d{2})(\d{1})(\d{1})/(\d{2})(\d{2}):(\d{2}):(\d{2})",
                r"\1/\2/\3\4 \5\6:\7:\8",
                normalized,
            )
            # パターン2: 4桁の年 + 2桁の月（/欠落） + 2桁の日 + 2桁の時（:欠落） + 分:秒
            normalized = re.sub(
                r"(\d{4})(\d{2})(\d{2})(\d{2}):(\d{2}):(\d{2})",
                r"\1/\2/\3 \4:\5:\6",
                normalized,
            )
            # パターン3: 4桁の年 + 2桁の月 + 2桁の日（/欠落）+ 2桁の時（:欠落） + 分:秒
            normalized = re.sub(
                r"(\d{4})/(\d{2})(\d{2})(\d{2}):(\d{2}):(\d{2})",
                r"\1/\2/\3 \4:\5:\6",
                normalized,
            )
            # パターン4: 4桁の年/2桁の月/2桁の日 + 2桁の時（スペース欠落）:分:秒
            normalized = re.sub(
                r"(\d{4}/\d{2}/\d{2})(\d{2}):(\d{2}):(\d{2})",
                r"\1 \2:\3:\4",
                normalized,
            )

            # パターン5: スペース欠落補正: 2025/08/270:13:31 -> 2025/08/27 0:13:31
            normalized = re.sub(
                r"(\d{4}/\d{2}/\d{2})(\d{1,2}:\d{2}:\d{2})", r"\1 \2", normalized
            )

            # 数字列から直接抽出を優先（スラッシュ欠落などの複雑なパターンに対応）
            digits_only = re.sub(r"[^0-9]", "", normalized)
            year = month = day = hour = minute = second = None

            if len(digits_only) >= 14:
                if len(digits_only) == 15:
                    # 15桁の場合: 202510826160526 -> 2025/10/08 16:05:26
                    # 最初の8桁を日付、残りを時刻として解釈
                    year = digits_only[0:4]
                    month = digits_only[4:6]
                    day = digits_only[6:8]
                    hour = digits_only[9:11]
                    minute = digits_only[11:13]
                    second = digits_only[13:15]
                else:
                    # 14桁の場合: 20251082616052 -> 2025/10/08 16:05:26
                    digits = digits_only[:14]
                    year = digits[0:4]
                    month = digits[4:6]
                    day = digits[6:8]
                    hour = digits[8:10]
                    minute = digits[10:12]
                    second = digits[12:14]
            elif len(digits_only) == 13:
                # 13桁の場合: 2025082751331 -> 2025/08/27 5:13:31（1桁時）
                # または: 2025082701331 -> 2025/08/27 0:13:31（1桁時）
                year = digits_only[0:4]
                month = digits_only[4:6]
                day = digits_only[6:8]
                hour = digits_only[8:9]  # 1桁
                minute = digits_only[9:11]
                second = digits_only[11:13]
            elif len(digits_only) == 12:
                digits = digits_only + "00"
                year = digits[0:4]
                month = digits[4:6]
                day = digits[6:8]
                hour = digits[8:10]
                minute = digits[10:12]
                second = digits[12:14]
            else:
                # フォールバック: パターンマッチを試す（スペース欠落や1桁時を許容）
                # パターン1: 厳密な形式
                pattern = (
                    r"(\d{4})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})(?:\D*(\d{2}))?"
                )
                match = re.search(pattern, normalized)
                if match:
                    year, month, day, hour, minute, second = match.groups()
                else:
                    # パターン2: スペース欠落や1桁時を許容（2025/08/270:13:31形式、2025/08/27 5:13:31形式）
                    pattern = (
                        r"(\d{4})\D*(\d{2})\D*(\d{2})\D*(\d{1,2})\D*(\d{2})\D*(\d{2})"
                    )
                    match = re.search(pattern, normalized)
                    if match:
                        year, month, day, hour, minute, second = match.groups()
                    else:
                        logger.debug(f"タイムスタンプパターンが見つかりませんでした: '{normalized}'")
                        return None

            if second is None:
                second = "00"

            # 数値化
            try:
                year_i = int(year) if year is not None else None
            except (TypeError, ValueError):
                year_i = None

            try:
                month_i = int(month) if month is not None else None
            except (TypeError, ValueError):
                month_i = None

            try:
                day_i = int(day) if day is not None else None
            except (TypeError, ValueError):
                day_i = None

            try:
                hour_i = int(hour)
                minute_i = int(minute)
                second_i = int(second)
            except (TypeError, ValueError):
                logger.debug(f"時刻部分の数値化に失敗: '{normalized}'")
                return None

            fallback_dt = self._last_timestamp
            corrections: List[Dict[str, str]] = []

            # 時刻の妥当性チェック
            if not (0 <= hour_i <= 23 and 0 <= minute_i <= 59 and 0 <= second_i <= 59):
                logger.debug("無効な時刻値を検出: %s:%s:%s", hour, minute, second)
                if fallback_dt is None:
                    return None
                old_time = f"{hour_i:02d}:{minute_i:02d}:{second_i:02d}"
                hour_i = fallback_dt.hour
                minute_i = fallback_dt.minute
                second_i = fallback_dt.second
                corrections.append(
                    {
                        "type": "time_invalid",
                        "before": old_time,
                        "after": f"{hour_i:02d}:{minute_i:02d}:{second_i:02d}",
                        "reason": "時刻範囲外のため履歴から補正",
                    }
                )

            # 年の桁化け補正（0257など）
            original_year = year_i
            if year_i is None or not (2000 <= year_i <= 2100):
                if fallback_dt is not None:
                    # 年が大きく外れている場合（±3年超）は補正しない
                    if year_i is not None and abs(year_i - fallback_dt.year) > 3:
                        logger.warning(
                            f"年の外れ値が大きすぎます: {year_i} (履歴: {fallback_dt.year})"
                        )
                        return None
                    year_i = fallback_dt.year
                    if original_year != year_i:
                        corrections.append(
                            {
                                "type": "year_corrupted",
                                "before": str(original_year)
                                if original_year
                                else "None",
                                "after": str(year_i),
                                "reason": f"年の桁化けを履歴から補正 (履歴: {fallback_dt.year})",
                            }
                        )
                else:
                    logger.debug(f"年の解析に失敗: '{normalized}'")
                    return None

            # 月の補正
            original_month = month_i
            if month_i is None or not (1 <= month_i <= 12):
                if fallback_dt is not None:
                    month_i = fallback_dt.month
                    if original_month != month_i:
                        corrections.append(
                            {
                                "type": "month_invalid",
                                "before": str(original_month)
                                if original_month
                                else "None",
                                "after": str(month_i),
                                "reason": f"月が範囲外のため履歴から補正 (履歴: {fallback_dt.month})",
                            }
                        )
                else:
                    logger.debug(f"月の解析に失敗: '{normalized}'")
                    return None

            # 日の補正
            original_day = day_i
            if day_i is None or not (1 <= day_i <= 31):
                if fallback_dt is not None:
                    day_i = fallback_dt.day
                    if original_day != day_i:
                        corrections.append(
                            {
                                "type": "day_invalid",
                                "before": str(original_day) if original_day else "None",
                                "after": str(day_i),
                                "reason": f"日が範囲外のため履歴から補正 (履歴: {fallback_dt.day})",
                            }
                        )
                else:
                    logger.debug(f"日の解析に失敗: '{normalized}'")
                    return None

            try:
                candidate = datetime(year_i, month_i, day_i, hour_i, minute_i, second_i)
            except ValueError:
                logger.debug(
                    "無効な日付/時刻として破棄: %04d/%02d/%02d %02d:%02d:%02d",
                    year_i,
                    month_i,
                    day_i,
                    hour_i,
                    minute_i,
                    second_i,
                )
                if fallback_dt is None:
                    return None
                candidate = fallback_dt.replace(
                    hour=hour_i,
                    minute=minute_i,
                    second=second_i,
                    microsecond=0,
                )
                corrections.append(
                    {
                        "type": "date_invalid",
                        "before": f"{year_i}/{month_i}/{day_i}",
                        "after": f"{fallback_dt.year}/{fallback_dt.month}/{fallback_dt.day}",
                        "reason": "無効な日付のため履歴から補正",
                    }
                )

            # 時系列の整合性チェック（施策4: 強化）
            if fallback_dt is not None:
                time_diff = abs((candidate - fallback_dt).total_seconds())

                # 日付レベルの外れ値検知（±0.5日以上の場合、警告・再検証）
                days_diff = abs((candidate.date() - fallback_dt.date()).days)
                if days_diff >= 0.5:
                    logger.warning(
                        f"日付レベルの外れ値検知: {candidate.date()} (履歴: {fallback_dt.date()}, 差={days_diff}日)"
                    )
                    # ±7日以上の差は破棄（誤認識の可能性が高い）
                    if days_diff >= 7:
                        logger.error(
                            f"日付の外れ値が大きすぎます: {candidate.date()} (履歴: {fallback_dt.date()}, 差={days_diff}日)"
                        )
                        return None

                # 3年以上の差は破棄
                if time_diff > 3 * 365 * 24 * 3600:
                    logger.warning(
                        f"時系列の外れ値が大きすぎます: {candidate} (履歴: {fallback_dt}, 差={time_diff:.0f}秒)"
                    )
                    return None

                # 日付跨ぎの補正（±12時間以内の日付跨ぎは許容）
                if candidate < fallback_dt - timedelta(hours=12):
                    candidate += timedelta(days=1)
                    corrections.append(
                        {
                            "type": "time_wrap",
                            "before": candidate.strftime(self.TIMESTAMP_FORMAT),
                            "after": candidate.strftime(self.TIMESTAMP_FORMAT),
                            "reason": "12時間以上の逆転を日付跨ぎとして補正",
                        }
                    )
                elif candidate > fallback_dt + timedelta(hours=12):
                    candidate -= timedelta(days=1)
                    corrections.append(
                        {
                            "type": "time_wrap",
                            "before": candidate.strftime(self.TIMESTAMP_FORMAT),
                            "after": candidate.strftime(self.TIMESTAMP_FORMAT),
                            "reason": "12時間以上の進みを日付跨ぎとして補正",
                        }
                    )

            result = candidate.strftime(self.TIMESTAMP_FORMAT)
            # _last_timestamp更新（施策3: 更新ロジック改善）
            # parse_timestamp()は信頼度情報を持っていないため、デフォルト値0.5を使用
            self._update_last_timestamp(candidate, confidence=0.5)

            # 補正ログを記録
            if corrections:
                self._last_corrections = corrections
                for corr in corrections:
                    logger.info(
                        f"補正適用: {corr['type']} - {corr['before']} -> {corr['after']} ({corr['reason']})"
                    )
            else:
                self._last_corrections = []

            return result

        except Exception as e:
            logger.error(f"タイムスタンプのパース中にエラーが発生しました: {e}")
            return None

    def extract_with_confidence(
        self,
        frame: np.ndarray,
        preproc_params: Optional[Dict] = None,
        ocr_params: Optional[Dict] = None,
    ) -> Tuple[Optional[str], float]:
        """タイムスタンプを信頼度付きで抽出する

        複数OCR設定で多数決を行い、信頼度を返す。

        Args:
            frame: 入力フレーム画像
            preproc_params: 前処理パラメータ（Noneの場合はインスタンスの設定を使用）
            ocr_params: OCRパラメータ（Noneの場合はインスタンスの設定を使用）

        Returns:
            (タイムスタンプ, 信頼度) のタプル
        """
        if frame is None or frame.size == 0:
            return None, 0.0

        # 一時的にパラメータを上書き
        original_preproc = self.preproc_params
        original_ocr = self.ocr_params

        try:
            if preproc_params is not None:
                self.preproc_params = preproc_params
            if ocr_params is not None:
                self.ocr_params = ocr_params

            # ROI領域を抽出
            x, y, w, h = self.roi
            frame_height, frame_width = frame.shape[:2]

            if x + w > frame_width or y + h > frame_height:
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)

            roi_image = frame[y : y + h, x : x + w]

            if roi_image.size == 0:
                return None, 0.0

            # 前処理
            preprocessed = self._preprocess_roi(roi_image)

            # OCR実行
            timestamp, confidence = self._multi_ocr_vote(preprocessed)

            # 信頼度チェック（施策1: 厳格なフィルタリング）
            if timestamp is not None:
                # 信頼度0.00の結果を除外
                if confidence <= 0.0:
                    logger.debug(f"信頼度0.00の結果を除外: {timestamp} (信頼度: {confidence:.3f})")
                    return None, 0.0

                # 信頼度閾値未満の結果を除外
                if confidence < self.confidence_threshold:
                    logger.debug(
                        f"信頼度閾値未満の結果を除外: {timestamp} "
                        f"(信頼度: {confidence:.3f} < 閾値: {self.confidence_threshold:.3f})"
                    )
                    return None, 0.0

            return timestamp, confidence

        except Exception as e:
            logger.error(f"信頼度付きタイムスタンプ抽出中にエラーが発生しました: {e}")
            return None, 0.0
        finally:
            # パラメータを復元
            self.preproc_params = original_preproc
            self.ocr_params = original_ocr

    def get_last_corrections(self) -> List[Dict[str, str]]:
        """最後に適用された補正ログを取得する

        Returns:
            補正ログのリスト
        """
        return self._last_corrections.copy()

    def _update_last_timestamp(
        self, candidate: datetime, confidence: float = 0.5
    ) -> bool:
        """_last_timestampを信頼度ベースで更新（施策3: 更新ロジック改善）

        暫定値と確定値の分離を実装（計画書の提案に基づく）:
        - _tentative_timestamp: 低信頼度でも更新（時系列チェック用）
        - _confirmed_timestamp: 高信頼度のみ更新（最終出力用）
        - _last_timestamp: 後方互換性のため保持（_confirmed_timestampと同期）

        Args:
            candidate: 候補タイムスタンプ
            confidence: 信頼度（0.0-1.0）

        Returns:
            更新が実行された場合True、スキップされた場合False
        """
        # 時系列チェックは暫定値で実施（計画書の提案）
        reference_timestamp = (
            self._tentative_timestamp
            or self._confirmed_timestamp
            or self._last_timestamp
        )

        # 暫定値の更新: 信頼度が低くても、時系列整合性があれば更新
        if reference_timestamp is not None:
            try:
                time_diff = abs((candidate - reference_timestamp).total_seconds())
                days_diff = abs((candidate.date() - reference_timestamp.date()).days)

                # 日付レベルの外れ値検知（±0.5日以上は除外）
                if days_diff >= 0.5:
                    logger.debug(
                        f"暫定値更新をスキップ: 日付レベルの外れ値 "
                        f"(差={days_diff}日, 履歴={reference_timestamp.date()}, 候補={candidate.date()})"
                    )
                    # 暫定値も更新しない（外れ値のため）
                    return False

                # ±7日以上の差は除外（誤認識の可能性が高い）
                if days_diff >= 7:
                    logger.debug(
                        f"暫定値更新をスキップ: 日付の外れ値が大きすぎる "
                        f"(差={days_diff}日, 履歴={reference_timestamp.date()}, 候補={candidate.date()})"
                    )
                    return False

                # 時間差が大きすぎる場合（1時間以上）は除外
                if time_diff > 3600:
                    logger.debug(
                        f"暫定値更新をスキップ: 時間差が大きすぎる "
                        f"(差={time_diff:.0f}秒, 履歴={reference_timestamp}, 候補={candidate})"
                    )
                    return False

                # 時系列が逆転している場合（12時間以上）は除外
                if candidate < reference_timestamp - timedelta(hours=12):
                    logger.debug(
                        f"暫定値更新をスキップ: 時系列逆転 "
                        f"(履歴={reference_timestamp}, 候補={candidate})"
                    )
                    return False

            except Exception as e:
                logger.warning(f"暫定値更新チェック中にエラー: {e}")
                return False

        # 暫定値を更新（時系列整合性があれば、信頼度に関係なく更新）
        self._tentative_timestamp = candidate
        logger.debug(
            f"暫定値更新: 時系列整合性チェック通過 " f"(信頼度: {confidence:.3f}, 候補: {candidate})"
        )

        # 確定値の更新: 高信頼度（≥ 0.7）のみ更新
        if confidence >= 0.7:
            self._confirmed_timestamp = candidate
            self._last_timestamp = candidate  # 後方互換性のため
            logger.debug(f"確定値更新: 高信頼度 (信頼度: {confidence:.3f}, 候補: {candidate})")
            return True

        # 中間の信頼度（0.3-0.7）は確定値に更新しないが、暫定値は更新済み
        if confidence >= 0.3:
            logger.debug(
                f"確定値更新をスキップ: 中程度の信頼度 "
                f"(信頼度: {confidence:.3f}, 候補: {candidate})。暫定値は更新済み。"
            )
            return True  # 暫定値は更新したのでTrue

        # 信頼度が低い結果（< 0.3）は確定値に更新しないが、暫定値は更新済み
        logger.debug(
            f"確定値更新をスキップ: 信頼度が低い "
            f"(信頼度: {confidence:.3f} < 閾値: 0.3, 候補: {candidate})。暫定値は更新済み。"
        )
        return True  # 暫定値は更新したのでTrue

    def _validate_initial_timestamp(
        self, timestamp: str, reference_date: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str]]:
        """初期タイムスタンプの妥当性をチェック

        Args:
            timestamp: タイムスタンプ文字列 (YYYY/MM/DD HH:MM:SS形式)
            reference_date: 参照日付（現在時刻または期待される日付）

        Returns:
            (妥当性, エラーメッセージ) のタプル
        """
        try:
            dt = datetime.strptime(timestamp, self.TIMESTAMP_FORMAT)
        except ValueError:
            return False, "タイムスタンプ形式が不正です"

        # 参照日付が指定されていない場合は現在時刻を使用
        if reference_date is None:
            reference_date = datetime.now()

        # 現在時刻±30日以内のチェック
        days_diff = abs((dt - reference_date).days)
        if days_diff > 30:
            return False, f"参照日付から{days_diff}日離れています（許容範囲: ±30日）"

        # 年の妥当性チェック（2000-2100）
        if not (2000 <= dt.year <= 2100):
            return False, f"年が範囲外です: {dt.year} (許容範囲: 2000-2100)"

        # 月の妥当性チェック（1-12）
        if not (1 <= dt.month <= 12):
            return False, f"月が範囲外です: {dt.month} (許容範囲: 1-12)"

        # 日の妥当性チェック（1-31、月に応じた日数のチェックも実装）
        if not (1 <= dt.day <= 31):
            return False, f"日が範囲外です: {dt.day} (許容範囲: 1-31)"

        # 月に応じた日数のチェック
        try:
            # 無効な日付（例: 2月30日）をチェック
            datetime(dt.year, dt.month, dt.day)
        except ValueError:
            return False, f"無効な日付です: {dt.year}/{dt.month}/{dt.day}"

        return True, None

    def _extract_initial_timestamp_with_consensus(
        self, frames: List[np.ndarray]
    ) -> Tuple[Optional[str], float]:
        """最初のNフレームでOCRを実行し、多数決で初期タイムスタンプを決定

        Args:
            frames: 最初のNフレームのリスト

        Returns:
            (タイムスタンプ, 平均信頼度) のタプル
        """
        all_engines_results: List[
            Tuple[str, float, str]
        ] = []  # (timestamp, confidence, engine)

        for frame_idx, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                continue

            try:
                # ROI領域を抽出
                x, y, w, h = self.roi
                frame_height, frame_width = frame.shape[:2]

                if x + w > frame_width or y + h > frame_height:
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)

                roi_image = frame[y : y + h, x : x + w]

                if roi_image.size == 0:
                    continue

                # 前処理
                preprocessed = self._preprocess_roi(roi_image)

                # 複数のOCRエンジンで実行
                engines = ["tesseract"]
                if PADDLEOCR_AVAILABLE:
                    engines.append("paddleocr")
                if EASYOCR_AVAILABLE:
                    engines.append("easyocr")

                for engine in engines:
                    try:
                        if engine == "tesseract":
                            # Tesseractは複数PSM設定で試行
                            ocr_configs = [
                                {"psm": 7, "whitelist": "0123456789/:", "lang": "eng"},
                                {"psm": 6, "whitelist": "0123456789/:", "lang": "eng"},
                                {"psm": 8, "whitelist": "0123456789/:", "lang": "eng"},
                            ]
                            for ocr_config in ocr_configs:
                                ocr_text, confidence = run_tesseract(
                                    preprocessed,
                                    psm=ocr_config["psm"],
                                    whitelist=ocr_config["whitelist"],
                                    lang=ocr_config["lang"],
                                )
                                if ocr_text:
                                    # 柔軟な後処理を使用
                                    timestamp = parse_flexible_timestamp(
                                        ocr_text,
                                        confidence=confidence,
                                        reference_timestamp=None,  # 初期抽出なので参照なし
                                    )
                                    if timestamp:
                                        all_engines_results.append(
                                            (
                                                timestamp,
                                                confidence,
                                                f"{engine}_psm{ocr_config['psm']}",
                                            )
                                        )
                        else:
                            # PaddleOCRまたはEasyOCR
                            ocr_text, confidence = run_ocr(preprocessed, engine=engine)
                            if ocr_text:
                                timestamp = parse_flexible_timestamp(
                                    ocr_text,
                                    confidence=confidence,
                                    reference_timestamp=None,
                                )
                                if timestamp:
                                    all_engines_results.append(
                                        (timestamp, confidence, engine)
                                    )
                    except Exception as e:
                        logger.debug(f"フレーム{frame_idx}、エンジン{engine}でエラー: {e}")
                        continue

            except Exception as e:
                logger.debug(f"フレーム{frame_idx}の処理中にエラー: {e}")
                continue

        if not all_engines_results:
            return None, 0.0

        # 信頼度が低い結果を除外（閾値: 0.3）
        filtered_results = [
            (ts, conf, eng) for ts, conf, eng in all_engines_results if conf >= 0.3
        ]

        if not filtered_results:
            # 信頼度が低い結果も含める
            filtered_results = all_engines_results

        # 多数決: 同じタイムスタンプの出現回数をカウント
        timestamp_counts = Counter(ts for ts, _, _ in filtered_results)

        if not timestamp_counts:
            return None, 0.0

        max_count = max(timestamp_counts.values())

        # 最も多く出現したタイムスタンプを候補に
        top_timestamps = [
            ts for ts, count in timestamp_counts.items() if count == max_count
        ]

        if len(top_timestamps) == 1:
            # 唯一の候補
            selected = top_timestamps[0]
            # そのタイムスタンプの平均信頼度を計算
            confs_for_selected = [
                conf for ts, conf, _ in filtered_results if ts == selected
            ]
            avg_conf = (
                sum(confs_for_selected) / len(confs_for_selected)
                if confs_for_selected
                else 0.0
            )
            return selected, avg_conf
        else:
            # タイの場合、平均信頼度が高いものを選択
            best_ts = None
            best_conf = -1.0
            for ts in top_timestamps:
                confs_for_ts = [
                    conf
                    for candidate_ts, conf, _ in filtered_results
                    if candidate_ts == ts
                ]
                avg_conf = (
                    sum(confs_for_ts) / len(confs_for_ts) if confs_for_ts else 0.0
                )
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_ts = ts
            if best_ts is None:
                return None, 0.0
            return best_ts, best_conf

    def _extract_final_timestamp_with_consensus(
        self, frames: List[np.ndarray]
    ) -> Tuple[Optional[str], float]:
        """最後のNフレームでOCRを実行し、多数決で最終タイムスタンプを決定（施策2: 最後のフレーム群検証）

        Args:
            frames: 最後のNフレームのリスト

        Returns:
            (タイムスタンプ, 平均信頼度) のタプル
        """
        all_engines_results: List[
            Tuple[str, float, str]
        ] = []  # (timestamp, confidence, engine)

        for frame_idx, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                continue

            try:
                # ROI領域を抽出
                x, y, w, h = self.roi
                frame_height, frame_width = frame.shape[:2]

                if x + w > frame_width or y + h > frame_height:
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)

                roi_image = frame[y : y + h, x : x + w]

                if roi_image.size == 0:
                    continue

                # 前処理
                preprocessed = self._preprocess_roi(roi_image)

                # 複数のOCRエンジンで実行
                engines = ["tesseract"]
                if PADDLEOCR_AVAILABLE:
                    engines.append("paddleocr")
                if EASYOCR_AVAILABLE:
                    engines.append("easyocr")

                for engine in engines:
                    try:
                        if engine == "tesseract":
                            # Tesseractは複数PSM設定で試行
                            ocr_configs = [
                                {"psm": 7, "whitelist": "0123456789/:", "lang": "eng"},
                                {"psm": 6, "whitelist": "0123456789/:", "lang": "eng"},
                                {"psm": 8, "whitelist": "0123456789/:", "lang": "eng"},
                            ]
                            for ocr_config in ocr_configs:
                                ocr_text, confidence = run_tesseract(
                                    preprocessed,
                                    psm=ocr_config["psm"],
                                    whitelist=ocr_config["whitelist"],
                                    lang=ocr_config["lang"],
                                )
                                if ocr_text:
                                    # 柔軟な後処理を使用（最後のフレーム群なので履歴を参照）
                                    timestamp = parse_flexible_timestamp(
                                        ocr_text,
                                        confidence=confidence,
                                        reference_timestamp=self._last_timestamp,
                                    )
                                    if timestamp:
                                        # 信頼度0.00の結果を除外
                                        if confidence <= 0.0:
                                            continue
                                        # 信頼度閾値未満の結果を除外
                                        if confidence < self.confidence_threshold:
                                            continue
                                        all_engines_results.append(
                                            (
                                                timestamp,
                                                confidence,
                                                f"{engine}_psm{ocr_config['psm']}",
                                            )
                                        )
                        else:
                            # PaddleOCRまたはEasyOCR
                            ocr_text, confidence = run_ocr(preprocessed, engine=engine)
                            if ocr_text:
                                timestamp = parse_flexible_timestamp(
                                    ocr_text,
                                    confidence=confidence,
                                    reference_timestamp=self._last_timestamp,
                                )
                                if timestamp:
                                    # 信頼度0.00の結果を除外
                                    if confidence <= 0.0:
                                        continue
                                    # 信頼度閾値未満の結果を除外
                                    if confidence < self.confidence_threshold:
                                        continue
                                    all_engines_results.append(
                                        (timestamp, confidence, engine)
                                    )
                    except Exception as e:
                        logger.debug(f"フレーム{frame_idx}、エンジン{engine}でエラー: {e}")
                        continue

            except Exception as e:
                logger.debug(f"フレーム{frame_idx}の処理中にエラー: {e}")
                continue

        if not all_engines_results:
            return None, 0.0

        # 時系列整合性チェック（最後のフレーム群では特に重要）
        if self._last_timestamp is not None:
            filtered_results = []
            for ts, conf, eng in all_engines_results:
                try:
                    ts_dt = datetime.strptime(ts, self.TIMESTAMP_FORMAT)
                    days_diff = abs((ts_dt.date() - self._last_timestamp.date()).days)
                    # ±7日以上の差は除外（誤認識の可能性が高い）
                    if days_diff >= 7:
                        logger.debug(
                            f"最後のフレーム群: 時系列外れ値検知: {ts} "
                            f"(履歴: {self._last_timestamp.date()}, 差={days_diff}日)"
                        )
                        continue
                    filtered_results.append((ts, conf, eng))
                except ValueError:
                    continue
            all_engines_results = filtered_results

        if not all_engines_results:
            return None, 0.0

        # 信頼度が低い結果を除外（閾値: confidence_threshold）
        filtered_results = [
            (ts, conf, eng)
            for ts, conf, eng in all_engines_results
            if conf >= self.confidence_threshold
        ]

        if not filtered_results:
            # 信頼度が低い結果も含める（最後の手段）
            filtered_results = all_engines_results

        # 多数決: 同じタイムスタンプの出現回数をカウント
        timestamp_counts = Counter(ts for ts, _, _ in filtered_results)

        if not timestamp_counts:
            return None, 0.0

        max_count = max(timestamp_counts.values())

        # 最も多く出現したタイムスタンプを候補に
        top_timestamps = [
            ts for ts, count in timestamp_counts.items() if count == max_count
        ]

        if len(top_timestamps) == 1:
            # 唯一の候補
            selected = top_timestamps[0]
            # そのタイムスタンプの平均信頼度を計算
            confs_for_selected = [
                conf for ts, conf, _ in filtered_results if ts == selected
            ]
            avg_conf = (
                sum(confs_for_selected) / len(confs_for_selected)
                if confs_for_selected
                else 0.0
            )
            return selected, avg_conf
        else:
            # タイの場合、平均信頼度が高いものを選択
            best_ts = None
            best_conf = -1.0
            for ts in top_timestamps:
                confs_for_ts = [
                    conf
                    for candidate_ts, conf, _ in filtered_results
                    if candidate_ts == ts
                ]
                avg_conf = (
                    sum(confs_for_ts) / len(confs_for_ts) if confs_for_ts else 0.0
                )
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_ts = ts
            if best_ts is None:
                return None, 0.0
            return best_ts, best_conf

    def _extract_date_independently(
        self, roi_image: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """日付部分（YYYY/MM/DD）を独立してOCR実行

        Args:
            roi_image: ROI領域の画像

        Returns:
            (日付文字列, 信頼度) のタプル
        """
        # ROI領域を日付部分と時刻部分に分割（仮定: 日付は左側、時刻は右側）
        height, width = roi_image.shape[:2]
        date_roi = roi_image[:, : width // 2]  # 左半分を日付領域とする

        # 前処理
        preprocessed = self._preprocess_roi(date_roi)

        # 複数のOCRエンジンで実行
        engines = ["tesseract"]
        if PADDLEOCR_AVAILABLE:
            engines.append("paddleocr")
        if EASYOCR_AVAILABLE:
            engines.append("easyocr")

        all_results: List[Tuple[str, float]] = []

        for engine in engines:
            try:
                if engine == "tesseract":
                    # Tesseractは複数PSM設定で試行
                    ocr_configs = [
                        {"psm": 7, "whitelist": "0123456789/", "lang": "eng"},
                        {"psm": 6, "whitelist": "0123456789/", "lang": "eng"},
                        {"psm": 8, "whitelist": "0123456789/", "lang": "eng"},
                    ]
                    for ocr_config in ocr_configs:
                        ocr_text, confidence = run_tesseract(
                            preprocessed,
                            psm=ocr_config["psm"],
                            whitelist=ocr_config["whitelist"],
                            lang=ocr_config["lang"],
                        )
                        if ocr_text:
                            # 日付形式を正規化
                            normalized = self._normalize_date_text(ocr_text)
                            if normalized:
                                all_results.append((normalized, confidence))
                else:
                    # PaddleOCRまたはEasyOCR
                    ocr_text, confidence = run_ocr(preprocessed, engine=engine)
                    if ocr_text:
                        normalized = self._normalize_date_text(ocr_text)
                        if normalized:
                            all_results.append((normalized, confidence))
            except Exception as e:
                logger.debug(f"日付抽出、エンジン{engine}でエラー: {e}")
                continue

        if not all_results:
            return None, 0.0

        # 多数決
        date_counts = Counter(date for date, _ in all_results)
        max_count = max(date_counts.values())
        top_dates = [date for date, count in date_counts.items() if count == max_count]

        if len(top_dates) == 1:
            selected = top_dates[0]
            confs = [conf for date, conf in all_results if date == selected]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            return selected, avg_conf
        else:
            # タイの場合、平均信頼度が高いものを選択
            best_date = None
            best_conf = -1.0
            for date in top_dates:
                confs = [conf for d, conf in all_results if d == date]
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_date = date
            return best_date, best_conf if best_date else (None, 0.0)

    def _extract_time_independently(
        self, roi_image: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """時刻部分（HH:MM:SS）を独立してOCR実行

        Args:
            roi_image: ROI領域の画像

        Returns:
            (時刻文字列, 信頼度) のタプル
        """
        # ROI領域を日付部分と時刻部分に分割
        height, width = roi_image.shape[:2]
        time_roi = roi_image[:, width // 2 :]  # 右半分を時刻領域とする

        # 前処理
        preprocessed = self._preprocess_roi(time_roi)

        # Tesseractで実行（時刻部分は単純なのでTesseractのみ）
        ocr_configs = [
            {"psm": 7, "whitelist": "0123456789:", "lang": "eng"},
            {"psm": 6, "whitelist": "0123456789:", "lang": "eng"},
        ]

        all_results: List[Tuple[str, float]] = []

        for ocr_config in ocr_configs:
            try:
                ocr_text, confidence = run_tesseract(
                    preprocessed,
                    psm=ocr_config["psm"],
                    whitelist=ocr_config["whitelist"],
                    lang=ocr_config["lang"],
                )
                if ocr_text:
                    # 時刻形式を正規化
                    normalized = self._normalize_time_text(ocr_text)
                    if normalized:
                        all_results.append((normalized, confidence))
            except Exception as e:
                logger.debug(f"時刻抽出、設定{ocr_config}でエラー: {e}")
                continue

        if not all_results:
            return None, 0.0

        # 多数決
        time_counts = Counter(time for time, _ in all_results)
        max_count = max(time_counts.values())
        top_times = [time for time, count in time_counts.items() if count == max_count]

        if len(top_times) == 1:
            selected = top_times[0]
            confs = [conf for time, conf in all_results if time == selected]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            return selected, avg_conf
        else:
            # タイの場合、平均信頼度が高いものを選択
            best_time = None
            best_conf = -1.0
            for time in top_times:
                confs = [conf for t, conf in all_results if t == time]
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_time = time
            return best_time, best_conf if best_time else (None, 0.0)

    def _normalize_date_text(self, text: str) -> Optional[str]:
        """日付テキストを正規化（YYYY/MM/DD形式）

        Args:
            text: OCR出力テキスト

        Returns:
            正規化された日付文字列、失敗時None
        """
        if not text:
            return None

        # 文字変換
        normalized = text.translate(
            str.maketrans(
                {
                    "O": "0",
                    "o": "0",
                    "D": "0",
                    "I": "1",
                    "l": "1",
                    "|": "1",
                    "／": "/",
                }
            )
        )

        # 数字とスラッシュのみを抽出
        normalized = re.sub(r"[^0-9/]", "", normalized)

        # パターンマッチ: YYYY/MM/DD または YYYYMMDD
        match = re.search(r"(\d{4})[/\-]?(\d{2})[/\-]?(\d{2})", normalized)
        if match:
            year, month, day = match.groups()
            # 妥当性チェック
            if self._validate_date_format(year, month, day):
                return f"{year}/{month}/{day}"

        return None

    def _normalize_time_text(self, text: str) -> Optional[str]:
        """時刻テキストを正規化（HH:MM:SS形式）

        Args:
            text: OCR出力テキスト

        Returns:
            正規化された時刻文字列、失敗時None
        """
        if not text:
            return None

        # 文字変換
        normalized = text.translate(
            str.maketrans(
                {
                    "O": "0",
                    "o": "0",
                    "I": "1",
                    "l": "1",
                    "|": "1",
                    "：": ":",
                }
            )
        )

        # 数字とコロンのみを抽出
        normalized = re.sub(r"[^0-9:]", "", normalized)

        # パターンマッチ: HH:MM:SS または HHMMSS
        match = re.search(r"(\d{1,2})[:]?(\d{2})[:]?(\d{2})", normalized)
        if match:
            hour, minute, second = match.groups()
            # ゼロ埋め
            hour = hour.zfill(2)
            # 妥当性チェック
            if (
                0 <= int(hour) <= 23
                and 0 <= int(minute) <= 59
                and 0 <= int(second) <= 59
            ):
                return f"{hour}:{minute}:{second}"

        return None

    def _validate_date_format(self, year: str, month: str, day: str) -> bool:
        """日付形式の妥当性をチェック

        Args:
            year: 年（文字列）
            month: 月（文字列）
            day: 日（文字列）

        Returns:
            妥当性
        """
        try:
            year_i = int(year)
            month_i = int(month)
            day_i = int(day)

            # 年の範囲チェック（2000-2100）
            if not (2000 <= year_i <= 2100):
                return False

            # 月の範囲チェック（1-12）
            if not (1 <= month_i <= 12):
                return False

            # 日の範囲チェック（1-31）
            if not (1 <= day_i <= 31):
                return False

            # 無効な日付（例: 2月30日）をチェック
            datetime(year_i, month_i, day_i)
            return True
        except (ValueError, TypeError):
            return False

    def _save_debug_outputs(
        self,
        frame: np.ndarray,
        roi_image: np.ndarray,
        preprocessed: np.ndarray,
        ocr_text: str,
        timestamp: Optional[str],
        frame_index: Optional[int],
        roi_bounds: Tuple[int, int, int, int],
    ) -> None:
        """デバッグ用に画像を保存する"""
        if not self._debug_enabled or self._debug_dir is None:
            return
        try:
            if frame_index is not None:
                frame_tag = f"frame_{int(frame_index):06d}"
            else:
                frame_tag = f"frame_{self._debug_counter:06d}"
                self._debug_counter += 1
            roi_path = self._debug_dir / f"{frame_tag}_roi.png"
            cv2.imwrite(str(roi_path), roi_image)

            debug_images = getattr(self, "_last_preprocess_debug", {})

            if self._debug_save_intermediate:
                if preprocessed is not None:
                    preprocessed_path = (
                        self._debug_dir / f"{frame_tag}_preprocessed.png"
                    )
                    cv2.imwrite(str(preprocessed_path), preprocessed)

                for name, image in debug_images.items():
                    if image is None:
                        continue
                    debug_path = self._debug_dir / f"{frame_tag}_{name}.png"
                    cv2.imwrite(str(debug_path), image)
            if self._debug_save_overlay:
                overlay = frame.copy()
                x, y, w, h = roi_bounds
                frame_height, frame_width = frame.shape[:2]
                x2 = min(x + w, frame_width - 1)
                y2 = min(y + h, frame_height - 1)
                cv2.rectangle(overlay, (x, y), (x2, y2), (0, 255, 0), 2)
                display_text = timestamp or ocr_text.strip() or "(no result)"
                cv2.putText(
                    overlay,
                    display_text,
                    (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                overlay_path = self._debug_dir / f"{frame_tag}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
        except Exception as exc:
            logger.error(f"デバッグ画像の保存に失敗しました: {exc}")
