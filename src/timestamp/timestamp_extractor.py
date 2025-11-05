"""Timestamp extraction module for the office person detection system."""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytesseract

from src.timestamp.ocr_engines import run_ocr, run_tesseract
from src.detection.preprocessing import apply_pipeline
from src.timestamp.timestamp_postprocess import parse_flexible_timestamp
from collections import Counter

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
    ):
        """TimestampExtractorを初期化する

        Args:
            roi: タイムスタンプ領域の座標 (x, y, width, height)
                 デフォルトは右上領域 (900, 10, 350, 60)
            preproc_params: 前処理パラメータの辞書（Noneの場合はデフォルト）
            ocr_params: OCRパラメータの辞書（Noneの場合はデフォルト）
            use_flexible_postprocess: 柔軟な後処理を使用するか
        """
        # デフォルトROIを右方向・下方向に広げて日時全体を包含
        self.roi = roi or (900, 30, 360, 45)
        logger.debug(f"TimestampExtractor初期化: ROI={self.roi}")
        self._debug_enabled = False
        self._debug_dir: Optional[Path] = None
        self._debug_save_intermediate = True
        self._debug_save_overlay = True
        self._debug_counter = 0
        self._last_preprocess_debug: Dict[str, np.ndarray] = {}
        self._last_timestamp: Optional[datetime] = None
        self._last_corrections: List[Dict[str, str]] = []
        # CLAHEとカーネルをキャッシュ
        self._clahe_cache: Optional[cv2.CLAHE] = None
        self._horizontal_kernel_cache: Optional[np.ndarray] = None

        # パラメータ設定
        self.preproc_params = preproc_params or self._get_default_preproc_params()
        self.ocr_params = ocr_params or self._get_default_ocr_params()
        self.use_flexible_postprocess = use_flexible_postprocess

    def _get_default_preproc_params(self) -> Dict:
        """デフォルト前処理パラメータを取得"""
        return {
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "resize": {"enabled": True, "fx": 2.0},
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
        """複数OCR設定で実行し、多数決でタイムスタンプを決定

        Args:
            preprocessed: 前処理済み画像

        Returns:
            (タイムスタンプ, 平均信頼度) のタプル
        """
        # 複数のPSM設定を試す
        ocr_configs = [
            {'psm': 7, 'whitelist': '0123456789/:', 'lang': 'eng'},
            {'psm': 6, 'whitelist': '0123456789/:', 'lang': 'eng'},
            {'psm': 8, 'whitelist': '0123456789/:', 'lang': 'eng'},
            {'psm': 13, 'whitelist': '0123456789/:', 'lang': 'eng'},
            {'psm': 11, 'whitelist': '0123456789/:', 'lang': 'eng'},
        ]
        
        candidates: List[Tuple[str, float]] = []
        
        for ocr_config in ocr_configs:
            try:
                ocr_text, confidence = run_tesseract(
                    preprocessed,
                    psm=ocr_config['psm'],
                    whitelist=ocr_config['whitelist'],
                    lang=ocr_config['lang']
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
                        candidates.append((timestamp, confidence))
                        continue
                
                # 従来の方法も試す
                timestamp = self._parse_strict_regex(ocr_text)
                if timestamp:
                    candidates.append((timestamp, confidence))
                    continue
                
                # フォールバック: parse_timestamp
                timestamp = self.parse_timestamp(ocr_text)
                if timestamp:
                    candidates.append((timestamp, confidence))
            except Exception as e:
                logger.debug(f"OCR設定 {ocr_config} でエラー: {e}")
                continue
        
        if not candidates:
            return None, 0.0
        
        # 多数決: 同じタイムスタンプの出現回数をカウント
        timestamp_counts = Counter(ts for ts, _ in candidates)
        max_count = max(timestamp_counts.values())
        
        # 最も多く出現したタイムスタンプを候補に
        top_timestamps = [ts for ts, count in timestamp_counts.items() if count == max_count]
        
        if len(top_timestamps) == 1:
            # 唯一の候補
            selected = top_timestamps[0]
            # そのタイムスタンプの平均信頼度を計算
            confs_for_selected = [conf for ts, conf in candidates if ts == selected]
            avg_conf = sum(confs_for_selected) / len(confs_for_selected) if confs_for_selected else 0.0
            return selected, avg_conf
        else:
            # タイの場合、平均信頼度が高いものを選択
            best_ts = None
            best_conf = -1.0
            for ts in top_timestamps:
                confs_for_ts = [conf for candidate_ts, conf in candidates if candidate_ts == ts]
                avg_conf = sum(confs_for_ts) / len(confs_for_ts) if confs_for_ts else 0.0
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_ts = ts
            if best_ts is None:
                return None, 0.0
            return best_ts, best_conf

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
        # まず、数字列として解析してから補正
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

        # 最優先: 厳密なパターン
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

    def extract(
        self, frame: np.ndarray, frame_index: Optional[int] = None
    ) -> Optional[str]:
        """フレームからタイムスタンプを抽出する

        複数OCR設定で多数決を行い、厳密な正規表現で抽出を試みる。

        Args:
            frame: 入力フレーム画像
            frame_index: デバッグ用のフレーム番号

        Returns:
            タイムスタンプ文字列 (YYYY/MM/DD HH:MM:SS形式)、失敗した場合None
        """
        if frame is None or frame.size == 0:
            logger.warning("無効なフレームが渡されました")
            return None

        try:
            # ROI領域を抽出
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

            roi_image = frame[y:y + h, x:x + w]
            roi_bounds = (x, y, w, h)

            if roi_image.size == 0:
                logger.warning("ROI領域が空です")
                return None

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

    def _preprocess_roi(self, roi_image: np.ndarray) -> np.ndarray:
        """OCR用の前処理を実行する

        パラメタ化された前処理パイプラインを使用。

        Args:
            roi_image: ROI領域の画像

        Returns:
            前処理済み画像
        """
        try:
            self._last_preprocess_debug = {}

            # パラメタ化された前処理を適用
            preprocessed = apply_pipeline(roi_image, self.preproc_params)

            self._last_preprocess_debug = {
                "preprocessed": preprocessed,
            }

            return preprocessed

        except Exception as e:
            logger.error(f"前処理中にエラーが発生しました: {e}")
            self._last_preprocess_debug = {}
            return roi_image

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

            # スペース欠落補正: 2025/08/270:13:31 -> 2025/08/27 0:13:31
            normalized = re.sub(r"(\d{2})(\d{1,2}:\d{2}:\d{2})", r"\1 \2", normalized)

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
            elif len(digits_only) == 12:
                digits = digits_only + "00"
                year = digits[0:4]
                month = digits[4:6]
                day = digits[6:8]
                hour = digits[8:10]
                minute = digits[10:12]
                second = digits[12:14]
            else:
                # フォールバック: パターンマッチを試す
                pattern = (
                    r"(\d{4})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})(?:\D*(\d{2}))?"
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

            # 時系列の整合性チェック（±12時間以内の日付跨ぎは許容、±3年超は破棄）
            if fallback_dt is not None:
                time_diff = abs((candidate - fallback_dt).total_seconds())
                if time_diff > 3 * 365 * 24 * 3600:  # 3年以上の差
                    logger.warning(
                        f"時系列の外れ値が大きすぎます: {candidate} (履歴: {fallback_dt}, 差={time_diff:.0f}秒)"
                    )
                    return None

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
            self._last_timestamp = candidate

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

            roi_image = frame[y:y + h, x:x + w]

            if roi_image.size == 0:
                return None, 0.0

            # 前処理
            preprocessed = self._preprocess_roi(roi_image)

            # OCR実行
            timestamp, confidence = self._multi_ocr_vote(preprocessed)

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
