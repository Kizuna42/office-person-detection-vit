"""Timestamp extraction module for the office person detection system."""

import cv2
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pytesseract


logger = logging.getLogger(__name__)


class TimestampExtractor:
    """タイムスタンプ抽出クラス
    
    フレームの右上領域からOCRを使用してタイムスタンプを抽出する。
    
    Attributes:
        roi: タイムスタンプ領域の座標 (x, y, width, height)
    """

    TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"
    
    def __init__(self, roi: Optional[Tuple[int, int, int, int]] = None):
        """TimestampExtractorを初期化する
        
        Args:
            roi: タイムスタンプ領域の座標 (x, y, width, height)
                 デフォルトは右上領域 (900, 10, 350, 60)
        """
        # デフォルトROIを右方向・下方向に広げて日時全体を包含
        self.roi = roi or (820, 0, 460, 90)
        logger.debug(f"TimestampExtractor初期化: ROI={self.roi}")
        self._debug_enabled = False
        self._debug_dir: Optional[Path] = None
        self._debug_save_intermediate = True
        self._debug_save_overlay = True
        self._debug_counter = 0
        self._last_preprocess_debug: Dict[str, np.ndarray] = {}
        self._last_timestamp: Optional[datetime] = None
    
    def enable_debug(
        self,
        debug_dir: Union[str, Path],
        save_intermediate: bool = True,
        save_overlay: bool = True
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
    
    def extract(self, frame: np.ndarray, frame_index: Optional[int] = None) -> Optional[str]:
        """フレームからタイムスタンプを抽出する
        
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
                logger.warning(f"ROI領域がフレームサイズを超えています: ROI={self.roi}, Frame={frame_width}x{frame_height}")
                # ROIを調整
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)
            
            roi_image = frame[y:y+h, x:x+w]
            roi_bounds = (x, y, w, h)
            
            if roi_image.size == 0:
                logger.warning("ROI領域が空です")
                return None
            
            # 前処理
            preprocessed = self._preprocess_roi(roi_image)
            
            # OCR実行
            ocr_text = pytesseract.image_to_string(
                preprocessed,
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:/'
            )
            
            logger.debug(f"OCR結果: '{ocr_text.strip()}'")
            
            # タイムスタンプをパース
            timestamp = self.parse_timestamp(ocr_text)
            
            if timestamp:
                logger.debug(f"抽出されたタイムスタンプ: {timestamp}")
            else:
                logger.warning(f"タイムスタンプの抽出に失敗しました: OCR結果='{ocr_text.strip()}'")
            
            if self._debug_enabled:
                self._save_debug_outputs(
                    frame,
                    roi_image,
                    preprocessed,
                    ocr_text,
                    timestamp,
                    frame_index,
                    roi_bounds
                )
            
            return timestamp
            
        except Exception as e:
            logger.error(f"タイムスタンプ抽出中にエラーが発生しました: {e}")
            return None
    
    def _focus_timestamp_band(self, gray_roi: np.ndarray) -> np.ndarray:
        """ROIからタイムスタンプ行を抽出する"""

        height, width = gray_roi.shape[:2]
        blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect: Optional[Tuple[int, int, int, int]] = None
        best_score = -1.0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < width * 0.4:
                continue
            if h < height * 0.1 or h > height * 0.8:
                continue

            width_score = w / max(width, 1)
            top_score = max(0.0, 1.0 - (y / max(height, 1)))
            score = width_score * 0.7 + top_score * 0.3

            if score > best_score:
                best_score = score
                best_rect = (x, y, w, h)

        if best_rect is None:
            band_height = max(int(height * 0.45), 1)
            band = gray_roi[:band_height, :]
            return band if band.size > 0 else gray_roi

        x, y, w, h = best_rect
        margin_y = max(4, min(12, h // 2))
        margin_x = max(4, min(20, w // 4))
        y0 = max(0, y - margin_y)
        y1 = min(height, y + h + margin_y)
        x0 = max(0, x - margin_x)
        x1 = min(width, x + w + margin_x)

        if (x1 - x0) < width * 0.75:
            x0 = 0
            x1 = width

        band = gray_roi[y0:y1, x0:x1]
        return band if band.size > 0 else gray_roi

    def _preprocess_roi(self, roi_image: np.ndarray) -> np.ndarray:
        """OCR用の前処理を実行する
        
        二値化、ノイズ除去、コントラスト強調を行う。
        
        Args:
            roi_image: ROI領域の画像
            
        Returns:
            前処理済み画像
        """
        try:
            self._last_preprocess_debug = {}

            if len(roi_image.shape) == 3:
                gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_image.copy()

            band = self._focus_timestamp_band(gray)

            normalized = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(normalized)

            resized = cv2.resize(
                enhanced,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )

            denoised = cv2.bilateralFilter(resized, d=5, sigmaColor=75, sigmaSpace=75)

            _, binary = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            if np.mean(binary) < 127:
                binary = cv2.bitwise_not(binary)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            self._last_preprocess_debug = {
                "gray": gray,
                "timestamp_band": band,
                "normalized": normalized,
                "enhanced": enhanced,
                "resized": resized,
                "denoised": denoised,
                "binary": cleaned,
            }

            return cleaned
            
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

            translation_table = str.maketrans({
                'O': '0',
                'o': '0',
                'D': '0',
                'S': '5',
                's': '5',
                'I': '1',
                'l': '1',
                '|': '1',
                'B': '8',
                'b': '6',
                'A': '4',
            })

            normalized = text.translate(translation_table)
            normalized = re.sub(r'[\t\r\n]', ' ', normalized)
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = normalized.replace('／', '/').replace('：', ':').replace('－', '-')

            pattern = r'(\d{4})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})(?:\D*(\d{2}))?'
            match = re.search(pattern, normalized)

            year = month = day = hour = minute = second = None

            if match:
                year, month, day, hour, minute, second = match.groups()
            else:
                digits_only = re.sub(r'[^0-9]', '', normalized)
                if len(digits_only) >= 14:
                    digits = digits_only[:14]
                elif len(digits_only) == 12:
                    digits = digits_only + '00'
                else:
                    logger.debug(f"タイムスタンプパターンが見つかりませんでした: '{normalized}'")
                    return None

                year = digits[0:4]
                month = digits[4:6]
                day = digits[6:8]
                hour = digits[8:10]
                minute = digits[10:12]
                second = digits[12:14]

            if second is None:
                second = '00'

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

            if not (0 <= hour_i <= 23 and 0 <= minute_i <= 59 and 0 <= second_i <= 59):
                logger.debug(
                    "無効な時刻値を検出: %s:%s:%s", hour, minute, second
                )
                if fallback_dt is None:
                    return None
                hour_i = fallback_dt.hour
                minute_i = fallback_dt.minute
                second_i = fallback_dt.second

            if year_i is None or not (2000 <= year_i <= 2100):
                if fallback_dt is not None:
                    year_i = fallback_dt.year
                else:
                    logger.debug(f"年の解析に失敗: '{normalized}'")
                    return None

            if month_i is None or not (1 <= month_i <= 12):
                if fallback_dt is not None:
                    month_i = fallback_dt.month
                else:
                    logger.debug(f"月の解析に失敗: '{normalized}'")
                    return None

            if day_i is None or not (1 <= day_i <= 31):
                if fallback_dt is not None:
                    day_i = fallback_dt.day
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

            if fallback_dt is not None:
                if candidate < fallback_dt - timedelta(hours=12):
                    candidate += timedelta(days=1)
                elif candidate > fallback_dt + timedelta(hours=12):
                    candidate -= timedelta(days=1)

            result = candidate.strftime(self.TIMESTAMP_FORMAT)
            self._last_timestamp = candidate
            return result

        except Exception as e:
            logger.error(f"タイムスタンプのパース中にエラーが発生しました: {e}")
            return None
    
    def extract_with_confidence(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """タイムスタンプを信頼度付きで抽出する
        
        Args:
            frame: 入力フレーム画像
            
        Returns:
            (タイムスタンプ, 信頼度) のタプル
        """
        if frame is None or frame.size == 0:
            return None, 0.0
        
        try:
            # ROI領域を抽出
            x, y, w, h = self.roi
            frame_height, frame_width = frame.shape[:2]
            
            if x + w > frame_width or y + h > frame_height:
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)
            
            roi_image = frame[y:y+h, x:x+w]
            
            if roi_image.size == 0:
                return None, 0.0
            
            # 前処理
            preprocessed = self._preprocess_roi(roi_image)
            
            # OCR実行（詳細データ付き）
            ocr_data = pytesseract.image_to_data(
                preprocessed,
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:/',
                output_type=pytesseract.Output.DICT
            )
            
            # 信頼度の計算
            confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # テキスト抽出
            text = ' '.join([word for word in ocr_data['text'] if word.strip()])
            
            # タイムスタンプをパース
            timestamp = self.parse_timestamp(text)
            
            return timestamp, avg_confidence / 100.0
            
        except Exception as e:
            logger.error(f"信頼度付きタイムスタンプ抽出中にエラーが発生しました: {e}")
            return None, 0.0

    def _save_debug_outputs(
        self,
        frame: np.ndarray,
        roi_image: np.ndarray,
        preprocessed: np.ndarray,
        ocr_text: str,
        timestamp: Optional[str],
        frame_index: Optional[int],
        roi_bounds: Tuple[int, int, int, int]
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
                    preprocessed_path = self._debug_dir / f"{frame_tag}_preprocessed.png"
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
                    2
                )
                overlay_path = self._debug_dir / f"{frame_tag}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
        except Exception as exc:
            logger.error(f"デバッグ画像の保存に失敗しました: {exc}")
