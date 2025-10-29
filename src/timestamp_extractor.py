"""Timestamp extraction module for the office person detection system."""

import cv2
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pytesseract


logger = logging.getLogger(__name__)


class TimestampExtractor:
    """タイムスタンプ抽出クラス
    
    フレームの右上領域からOCRを使用してタイムスタンプを抽出する。
    
    Attributes:
        roi: タイムスタンプ領域の座標 (x, y, width, height)
    """
    
    def __init__(self, roi: Optional[Tuple[int, int, int, int]] = None):
        """TimestampExtractorを初期化する
        
        Args:
            roi: タイムスタンプ領域の座標 (x, y, width, height)
                 デフォルトは右上領域 (900, 10, 350, 60)
        """
        self.roi = roi or (900, 10, 350, 60)
        logger.debug(f"TimestampExtractor初期化: ROI={self.roi}")
        self._debug_enabled = False
        self._debug_dir: Optional[Path] = None
        self._debug_save_intermediate = True
        self._debug_save_overlay = True
        self._debug_counter = 0
    
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
            タイムスタンプ文字列 (HH:MM形式)、失敗した場合None
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
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:'
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
    
    def _preprocess_roi(self, roi_image: np.ndarray) -> np.ndarray:
        """OCR用の前処理を実行する
        
        二値化、ノイズ除去、コントラスト強調を行う。
        
        Args:
            roi_image: ROI領域の画像
            
        Returns:
            前処理済み画像
        """
        try:
            # グレースケール変換
            if len(roi_image.shape) == 3:
                gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_image.copy()
            
            # コントラスト強調（CLAHE）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # ノイズ除去（ガウシアンブラー）
            denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # 二値化（大津の方法）
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 反転（白背景に黒文字の場合）
            # OCRは黒背景に白文字を期待するため、必要に応じて反転
            mean_val = np.mean(binary)
            if mean_val > 127:
                binary = cv2.bitwise_not(binary)
            
            # モルフォロジー処理（ノイズ除去）
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"前処理中にエラーが発生しました: {e}")
            return roi_image
    
    def parse_timestamp(self, ocr_text: str) -> Optional[str]:
        """OCR結果からタイムスタンプを抽出・正規化する
        
        Args:
            ocr_text: OCRで読み取られたテキスト
            
        Returns:
            正規化されたタイムスタンプ (HH:MM形式)、失敗した場合None
        """
        if not ocr_text:
            return None
        
        try:
            # 改行やスペースを除去
            text = ocr_text.strip().replace('\n', '').replace(' ', '')
            
            # HH:MM形式のパターンを検索
            # 例: 12:30, 09:15, 23:59
            pattern = r'(\d{1,2}):(\d{2})'
            match = re.search(pattern, text)
            
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                
                # 時刻の妥当性チェック
                if 0 <= hours <= 23 and 0 <= minutes <= 59:
                    # HH:MM形式に正規化
                    timestamp = f"{hours:02d}:{minutes:02d}"
                    return timestamp
                else:
                    logger.warning(f"無効な時刻: {hours}:{minutes}")
                    return None
            
            # コロンなしの4桁数字パターンも試す（例: 1230 -> 12:30）
            pattern_no_colon = r'(\d{2})(\d{2})'
            match_no_colon = re.search(pattern_no_colon, text)
            
            if match_no_colon:
                hours = int(match_no_colon.group(1))
                minutes = int(match_no_colon.group(2))
                
                if 0 <= hours <= 23 and 0 <= minutes <= 59:
                    timestamp = f"{hours:02d}:{minutes:02d}"
                    logger.debug(f"コロンなし形式から変換: {text} -> {timestamp}")
                    return timestamp
            
            logger.debug(f"タイムスタンプパターンが見つかりませんでした: '{text}'")
            return None
            
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
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:',
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
            if self._debug_save_intermediate:
                preprocessed_path = self._debug_dir / f"{frame_tag}_preprocessed.png"
                cv2.imwrite(str(preprocessed_path), preprocessed)
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
