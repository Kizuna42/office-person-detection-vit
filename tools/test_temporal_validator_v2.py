#!/usr/bin/env python
"""時系列検証強化版のテストツール

TemporalValidatorV2の精度向上効果を測定します。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# プロジェクトルートをパスに追加（直接実行可能にする）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.timestamp.roi_extractor import TimestampROIExtractor
from src.timestamp.ocr_engine import MultiEngineOCR
from src.timestamp.timestamp_parser import TimestampParser
from src.timestamp.timestamp_validator import TemporalValidator
from src.timestamp.timestamp_validator_v2 import TemporalValidatorV2
from src.utils import setup_logging
from src.video import VideoProcessor

logger = logging.getLogger(__name__)


def test_validators(
    video_path: str,
    roi_config: dict,
    frame_indices: List[int],
    fps: float = 30.0,
) -> Dict:
    """両方のバリデーターをテスト
    
    Args:
        video_path: 動画ファイルのパス
        roi_config: ROI設定
        frame_indices: テストするフレーム番号のリスト
        fps: フレームレート
    
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
    
    # コンポーネントを初期化
    roi_extractor = TimestampROIExtractor(roi_config=roi_config)
    ocr_engine = MultiEngineOCR(enabled_engines=['tesseract'])
    timestamp_parser = TimestampParser()
    
    validators = {
        "baseline": TemporalValidator(fps=fps),
        "improved": TemporalValidatorV2(fps=fps),
    }
    
    results = {}
    
    # 各バリデーターでテスト
    for validator_name, validator in validators.items():
        logger.info(f"バリデーター '{validator_name}' をテスト中...")
        
        validator.reset()
        validator_results = []
        valid_count = 0
        total_confidence = 0.0
        
        for frame_idx, frame in frames:
            # ROI抽出
            roi, roi_coords = roi_extractor.extract_roi(frame)
            
            # 前処理
            preprocessed = roi_extractor.preprocess_roi(roi)
            
            # OCR実行
            ocr_text, ocr_confidence = ocr_engine.extract_with_consensus(preprocessed)
            
            if not ocr_text:
                continue
            
            # パース
            try:
                timestamp, parse_confidence = timestamp_parser.fuzzy_parse(ocr_text)
                if not timestamp:
                    continue
            except:
                continue
            
            # 時系列検証
            is_valid, temporal_confidence, reason = validator.validate(timestamp, frame_idx)
            
            if is_valid:
                valid_count += 1
                total_confidence += temporal_confidence
            
            validator_results.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'is_valid': is_valid,
                'temporal_confidence': temporal_confidence,
                'reason': reason,
            })
        
        avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
        valid_rate = (valid_count / len(validator_results) * 100) if validator_results else 0.0
        
        results[validator_name] = {
            'valid_count': valid_count,
            'total_count': len(validator_results),
            'valid_rate': valid_rate,
            'avg_confidence': avg_confidence,
            'results': validator_results,
        }
        
        logger.info(f"  {validator_name}: 有効率={valid_rate:.2f}%, 平均信頼度={avg_confidence:.4f}")
    
    return results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="時系列検証強化版のテスト")
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
    
    # FPSの取得
    fps = config.get('video.fps', 30.0)
    
    # フレーム番号のパース
    frame_indices = [int(x.strip()) for x in args.frame_indices.split(',')]
    
    # 出力ディレクトリ
    if args.output:
        output_path = Path(args.output)
        output_dir = output_path.parent
    else:
        output_dir = Path(config.get('output.directory', 'output')) / 'temporal_validator_test'
        output_path = output_dir / 'temporal_validator_test_results.json'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # テスト実行
    logger.info("=" * 80)
    logger.info("時系列検証強化版のテスト")
    logger.info("=" * 80)
    logger.info(f"動画: {video_path}")
    logger.info(f"テストフレーム: {frame_indices}")
    logger.info("=" * 80)
    
    results = test_validators(
        video_path=video_path,
        roi_config=roi_config,
        frame_indices=frame_indices,
        fps=fps,
    )
    
    # 結果を比較
    logger.info("=" * 80)
    logger.info("バリデーター比較結果")
    logger.info("=" * 80)
    
    baseline = results.get('baseline', {})
    improved = results.get('improved', {})
    
    logger.info(f"ベースライン:")
    logger.info(f"  有効率: {baseline.get('valid_rate', 0.0):.2f}%")
    logger.info(f"  平均信頼度: {baseline.get('avg_confidence', 0.0):.4f}")
    logger.info("")
    
    logger.info(f"改善版:")
    logger.info(f"  有効率: {improved.get('valid_rate', 0.0):.2f}%")
    logger.info(f"  平均信頼度: {improved.get('avg_confidence', 0.0):.4f}")
    logger.info("")
    
    # 改善度を計算
    valid_rate_delta = improved.get('valid_rate', 0.0) - baseline.get('valid_rate', 0.0)
    confidence_delta = improved.get('avg_confidence', 0.0) - baseline.get('avg_confidence', 0.0)
    
    logger.info(f"改善度:")
    logger.info(f"  有効率: {valid_rate_delta:+.2f}%")
    logger.info(f"  平均信頼度: {confidence_delta:+.4f}")
    logger.info("")
    
    improved_flag = (valid_rate_delta > 0 or confidence_delta > 0.01)
    
    if improved_flag:
        logger.info("✅ 改善が確認されました")
    else:
        logger.info("❌ 改善が見られませんでした")
    
    logger.info("=" * 80)
    
    # 結果をJSONで保存
    import json
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n結果を保存しました: {output_path}")
    
    return 0 if improved_flag else 1


if __name__ == "__main__":
    sys.exit(main())

