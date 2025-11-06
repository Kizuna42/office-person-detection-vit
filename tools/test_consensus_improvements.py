#!/usr/bin/env python
"""コンセンサスアルゴリズム改善のテストツール

重み付けスキーム、投票ロジック、フォールバックメカニズムをテストします。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.timestamp.roi_extractor import TimestampROIExtractor
from src.timestamp.ocr_engine import MultiEngineOCR
from src.timestamp.timestamp_parser import TimestampParser
from src.utils import setup_logging
from src.video import VideoProcessor

logger = logging.getLogger(__name__)


class ImprovedConsensusOCR(MultiEngineOCR):
    """改善されたコンセンサスアルゴリズム
    
    重み付けスキーム、投票ロジック、フォールバックメカニズムを実装
    """
    
    def extract_with_weighted_consensus(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        """重み付けスキームによるコンセンサス
        
        Args:
            roi: 前処理済みROI画像
        
        Returns:
            (抽出テキスト, 信頼度) のタプル
        """
        if not self.engines:
            return None, 0.0
        
        results: List[Dict] = []
        
        for engine_name, engine_func in self.engines.items():
            try:
                text = engine_func(roi)
                confidence = self._calculate_confidence(text)
                
                # エンジン別の重み（Tesseractを優先）
                weight = 1.0 if engine_name == "tesseract" else 0.8
                weighted_confidence = confidence * weight
                
                results.append({
                    "engine": engine_name,
                    "text": text.strip(),
                    "confidence": confidence,
                    "weighted_confidence": weighted_confidence,
                })
            except Exception as e:
                logger.debug(f"{engine_name} failed: {e}")
        
        if not results:
            return None, 0.0
        
        # 重み付け信頼度でソート
        results.sort(key=lambda x: x["weighted_confidence"], reverse=True)
        
        # エンジン間の一致度を評価
        if len(results) >= 2:
            top1, top2 = results[0], results[1]
            similarity = self._calculate_similarity(top1["text"], top2["text"])
            
            # 一致度が高い場合は信頼度を向上
            if similarity > 0.8:
                avg_confidence = (top1["weighted_confidence"] + top2["weighted_confidence"]) / 2
                avg_confidence = min(avg_confidence * 1.1, 1.0)  # 10%向上
                return top1["text"], avg_confidence
        
        best = results[0]
        return best["text"], best["weighted_confidence"]
    
    def extract_with_voting(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        """投票ロジックによるコンセンサス
        
        Args:
            roi: 前処理済みROI画像
        
        Returns:
            (抽出テキスト, 信頼度) のタプル
        """
        if not self.engines:
            return None, 0.0
        
        results: List[Dict] = []
        
        for engine_name, engine_func in self.engines.items():
            try:
                text = engine_func(roi)
                confidence = self._calculate_confidence(text)
                results.append({
                    "engine": engine_name,
                    "text": text.strip(),
                    "confidence": confidence,
                })
            except Exception as e:
                logger.debug(f"{engine_name} failed: {e}")
        
        if not results:
            return None, 0.0
        
        # テキストごとに投票
        text_votes: Dict[str, List[float]] = {}
        for r in results:
            text = r["text"]
            if text not in text_votes:
                text_votes[text] = []
            text_votes[text].append(r["confidence"])
        
        # 2/3以上のエンジンが一致したテキストを採用
        threshold = len(self.engines) * 2 / 3
        for text, confidences in text_votes.items():
            if len(confidences) >= threshold:
                avg_confidence = sum(confidences) / len(confidences)
                return text, avg_confidence
        
        # 2/3一致がない場合は最高信頼度を返す
        results.sort(key=lambda x: x["confidence"], reverse=True)
        best = results[0]
        return best["text"], best["confidence"]


def test_consensus_methods(
    video_path: str,
    roi_config: dict,
    frame_indices: List[int],
    output_dir: Path,
) -> Dict:
    """すべてのコンセンサス手法をテスト
    
    Args:
        video_path: 動画ファイルのパス
        roi_config: ROI設定
        frame_indices: テストするフレーム番号のリスト
        output_dir: 出力ディレクトリ
    
    Returns:
        テスト結果の辞書
    """
    # サンプルフレームを抽出
    video_processor = VideoProcessor(video_path)
    video_processor.open()
    
    try:
        frames = []
        for idx in frame_indices:
            frame = video_processor.get_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
    finally:
        video_processor.release()
    
    if not frames:
        logger.error("フレームを抽出できませんでした")
        return {}
    
    # ROI抽出器を初期化
    roi_extractor = TimestampROIExtractor(roi_config=roi_config)
    
    # タイムスタンプパーサーを初期化
    timestamp_parser = TimestampParser()
    
    # テストするコンセンサス手法
    methods = {
        "baseline": MultiEngineOCR(enabled_engines=['tesseract']),
        "weighted": ImprovedConsensusOCR(enabled_engines=['tesseract']),
        "voting": ImprovedConsensusOCR(enabled_engines=['tesseract']),
    }
    
    results = {}
    
    # 各手法でテスト
    for method_name, ocr_engine in methods.items():
        logger.info(f"コンセンサス手法 '{method_name}' をテスト中...")
        
        method_results = []
        parse_success_count = 0
        
        for frame_idx, frame in frames:
            # ROI抽出
            roi, roi_coords = roi_extractor.extract_roi(frame)
            
            # 前処理
            preprocessed = roi_extractor.preprocess_roi(roi)
            
            # OCR実行
            if method_name == "weighted":
                ocr_text, ocr_confidence = ocr_engine.extract_with_weighted_consensus(preprocessed)
            elif method_name == "voting":
                ocr_text, ocr_confidence = ocr_engine.extract_with_voting(preprocessed)
            else:
                ocr_text, ocr_confidence = ocr_engine.extract_with_consensus(preprocessed)
            
            # パース試行
            parsed_timestamp = None
            try:
                parsed_timestamp = timestamp_parser.parse(ocr_text) if ocr_text else None
                if parsed_timestamp:
                    parse_success_count += 1
            except:
                pass
            
            method_results.append({
                'frame_idx': frame_idx,
                'ocr_text': ocr_text,
                'ocr_confidence': ocr_confidence,
                'parsed': parsed_timestamp is not None,
            })
        
        # 統計を計算
        avg_confidence = sum(r['ocr_confidence'] for r in method_results) / len(method_results) if method_results else 0.0
        parse_success_rate = (parse_success_count / len(method_results) * 100) if method_results else 0.0
        
        results[method_name] = {
            'avg_confidence': avg_confidence,
            'parse_success_rate': parse_success_rate,
            'parse_success_count': parse_success_count,
            'total_count': len(method_results),
            'results': method_results,
        }
        
        logger.info(f"  {method_name}: 平均信頼度={avg_confidence:.4f}, パース成功率={parse_success_rate:.2f}%")
    
    return results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="コンセンサスアルゴリズム改善のテスト")
    parser.add_argument("--video", type=str, help="動画ファイルのパス")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--frame-indices", type=str, default="0,1000,2000,3000,4000",
                       help="テストするフレーム番号（カンマ区切り）")
    parser.add_argument("--output", type=str, help="結果出力ファイル（JSON）")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")
    
    args = parser.parse_args()
    
    # ロギング設定
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # 設定読み込み
    config = ConfigManager(args.config)
    
    # 動画パスの取得
    if args.video:
        video_path = args.video
    else:
        video_path = config.get('video.input_path')
    
    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return 1
    
    # ROI設定の取得
    timestamp_config = config.get('timestamp', {})
    extraction_config = timestamp_config.get('extraction', {})
    roi_config = extraction_config.get('roi', {})
    
    # フレーム番号のパース
    frame_indices = [int(x.strip()) for x in args.frame_indices.split(',')]
    
    # 出力ディレクトリ
    if args.output:
        output_path = Path(args.output)
        output_dir = output_path.parent
    else:
        output_dir = Path(config.get('output.directory', 'output')) / 'consensus_test'
        output_path = output_dir / 'consensus_test_results.json'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # テスト実行
    logger.info("=" * 80)
    logger.info("コンセンサスアルゴリズム改善のテスト")
    logger.info("=" * 80)
    logger.info(f"動画: {video_path}")
    logger.info(f"テストフレーム: {frame_indices}")
    logger.info("=" * 80)
    
    results = test_consensus_methods(
        video_path=video_path,
        roi_config=roi_config,
        frame_indices=frame_indices,
        output_dir=output_dir,
    )
    
    # 結果を比較
    logger.info("=" * 80)
    logger.info("コンセンサス手法比較結果")
    logger.info("=" * 80)
    
    best_method = None
    best_score = -1.0
    
    for method_name, result in results.items():
        # スコア = パース成功率 * 0.7 + 平均信頼度 * 0.3
        score = (result['parse_success_rate'] / 100.0) * 0.7 + result['avg_confidence'] * 0.3
        logger.info(f"{method_name}:")
        logger.info(f"  平均信頼度: {result['avg_confidence']:.4f}")
        logger.info(f"  パース成功率: {result['parse_success_rate']:.2f}%")
        logger.info(f"  総合スコア: {score:.4f}")
        logger.info("")
        
        if score > best_score:
            best_score = score
            best_method = method_name
    
    logger.info("=" * 80)
    if best_method:
        logger.info(f"最良の手法: {best_method} (スコア: {best_score:.4f})")
    logger.info("=" * 80)
    
    # 結果をJSONで保存
    import json
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n結果を保存しました: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

