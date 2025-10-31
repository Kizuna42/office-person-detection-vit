#!/usr/bin/env python
"""
オフィス人物検出システム - メインエントリーポイント

Vision Transformer (ViT) ベースの物体検出モデルを使用して、
オフィス内の定点カメラ映像から人物を検出し、
フロアマップ上でのゾーン別人数集計を実行します。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from src.config_manager import ConfigManager
from src.video_processor import VideoProcessor
from src.timestamp_extractor import TimestampExtractor
from src.frame_sampler import FrameSampler
from src.vit_detector import ViTDetector
from src.evaluation_module import EvaluationModule
from src.coordinate_transformer import CoordinateTransformer
from src.zone_classifier import ZoneClassifier
from src.aggregator import Aggregator
from src.visualizer import Visualizer
from src.floormap_visualizer import FloormapVisualizer
from src.data_models import Detection, FrameResult


# ロギング設定
def setup_logging(debug_mode: bool = False, output_dir: str = 'output'):
    """ロギングを設定する
    
    Args:
        debug_mode: デバッグモードの場合True
        output_dir: 出力ディレクトリ
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 既存のハンドラをクリア
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # コンソール出力
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # ファイル出力
    log_dir = Path(output_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    file_handler = logging.FileHandler(log_dir / 'system.log', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # ロガーに設定
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def parse_arguments():
    """コマンドライン引数をパースする
    
    Returns:
        パース済み引数
    """
    parser = argparse.ArgumentParser(
        description='オフィス人物検出システム - ViTベースの人物検出とゾーン別集計'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='設定ファイルのパス（デフォルト: config.yaml）'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='デバッグモードで実行（詳細ログ、中間結果出力）'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='精度評価モードで実行（Ground Truthとの比較）'
    )
    
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='ファインチューニングモードで実行'
    )
    
    parser.add_argument(
        '--start-time',
        type=str,
        help='開始時刻を指定（HH:MM形式）、指定しない場合は自動検出'
    )
    
    parser.add_argument(
        '--end-time',
        type=str,
        help='終了時刻を指定（HH:MM形式）、指定しない場合は自動検出'
    )
    
    return parser.parse_args()


def process_detections(
    sample_frames: List[Tuple[int, str, np.ndarray]],
    detector: ViTDetector,
    config: ConfigManager,
    logger: logging.Logger
) -> List[Tuple[int, str, List[Detection]]]:
    """人物検出処理を実行
    
    Args:
        sample_frames: サンプルフレームのリスト [(frame_num, timestamp, frame), ...]
        detector: ViTDetectorインスタンス
        config: ConfigManager インスタンス
        logger: ロガー
        
    Returns:
        検出結果のリスト [(frame_num, timestamp, detections), ...]
    """
    import gc
    
    results = []
    batch_size = config.get('detection.batch_size', 4)
    save_detection_images = config.get('output.save_detection_images', True)
    output_dir = Path(config.get('output.directory', 'output'))
    
    logger.info(f"バッチサイズ: {batch_size}")
    
    # バッチ処理
    for i in tqdm(range(0, len(sample_frames), batch_size), desc="人物検出中"):
        batch = sample_frames[i:i + batch_size]
        batch_frames = [frame for _, _, frame in batch]
        
        try:
            # バッチ検出
            batch_detections = detector.detect_batch(batch_frames, batch_size=len(batch_frames))
            
            # 結果を保存
            for j, (frame_num, timestamp, frame) in enumerate(batch):
                detections = batch_detections[j]
                results.append((frame_num, timestamp, detections))
                
                logger.info(f"フレーム #{frame_num} ({timestamp}): {len(detections)}人検出")
                
                # 検出画像を保存（オプション）
                if save_detection_images and detections:
                    save_detection_image(
                        frame, detections, timestamp,
                        output_dir / 'detections', logger
                    )
            
            # バッチ処理後のメモリ解放
            del batch_frames, batch_detections
            # 定期的にガベージコレクションを実行
            if (i // batch_size + 1) % 10 == 0:
                gc.collect()
                    
        except Exception as e:
            logger.error(f"バッチ {i//batch_size + 1} の検出処理に失敗しました: {e}", exc_info=True)
            # エラーが発生した場合は空の結果を追加
            for frame_num, timestamp, _ in batch:
                results.append((frame_num, timestamp, []))
                logger.warning(f"フレーム #{frame_num} をスキップしました")
        finally:
            # バッチ変数のクリーンアップ
            del batch
    
    return results


def save_detection_image(
    frame: np.ndarray,
    detections: List[Detection],
    timestamp: str,
    output_dir: Path,
    logger: logging.Logger
):
    """検出結果を画像として保存
    
    Args:
        frame: 入力フレーム
        detections: 検出結果のリスト
        timestamp: タイムスタンプ
        output_dir: 出力ディレクトリ
        logger: ロガー
    """
    import cv2
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # フレームをコピー（描画用）
        result_image = frame.copy()
        
        try:
            # バウンディングボックスを描画
            for detection in detections:
                x, y, w, h = detection.bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # ボックスを描画
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 信頼度を表示
                label = f"Person {detection.confidence:.2f}"
                cv2.putText(
                    result_image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                
                # 足元座標を描画
                foot_x, foot_y = detection.camera_coords
                cv2.circle(result_image, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)
            
            # ファイル名を生成
            filename = f"detection_{timestamp.replace(':', '')}.jpg"
            output_path = output_dir / filename
            
            # 保存
            cv2.imwrite(str(output_path), result_image)
            logger.debug(f"検出画像を保存しました: {output_path}")
        finally:
            # 描画用画像の参照を削除（メモリ節約）
            del result_image
        
    except Exception as e:
        logger.error(f"検出画像の保存に失敗しました: {e}")


def run_evaluation(
    detection_results: List[Tuple[int, str, List[Detection]]],
    config: ConfigManager,
    logger: logging.Logger
) -> None:
    """精度評価を実行
    
    Args:
        detection_results: 検出結果のリスト
        config: ConfigManager インスタンス
        logger: ロガー
    """
    logger.info("=" * 60)
    logger.info("精度評価を開始します")
    logger.info("=" * 60)
    
    try:
        # 評価モジュールの初期化
        gt_path = config.get('evaluation.ground_truth_path')
        iou_threshold = config.get('evaluation.iou_threshold', 0.5)
        
        evaluator = EvaluationModule(gt_path, iou_threshold)
        
        # 検出結果を辞書形式に変換
        detections_dict = {}
        for frame_num, timestamp, detections in detection_results:
            filename = f"frame_{frame_num:06d}_{timestamp.replace(':', 'h')}m.jpg"
            detections_dict[filename] = detections
        
        # 評価実行
        metrics = evaluator.evaluate(detections_dict)
        
        # レポート出力
        output_dir = Path(config.get('output.directory', 'output'))
        evaluator.export_report(metrics, str(output_dir / 'evaluation_report.csv'), format='csv')
        evaluator.export_report(metrics, str(output_dir / 'evaluation_report.json'), format='json')
        
        logger.info("=" * 60)
        logger.info("評価結果:")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall: {metrics.recall:.4f}")
        logger.info(f"  F1-score: {metrics.f1_score:.4f}")
        logger.info(f"  True Positives: {metrics.true_positives}")
        logger.info(f"  False Positives: {metrics.false_positives}")
        logger.info(f"  False Negatives: {metrics.false_negatives}")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"Ground Truthファイルが見つかりません: {e}")
    except Exception as e:
        logger.error(f"評価処理中にエラーが発生しました: {e}", exc_info=True)


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    args = parse_arguments()
    
    # 初期ロギング設定（設定ファイル読み込み前）
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("オフィス人物検出システム 起動")
    logger.info("=" * 80)
    
    video_processor = None
    detector = None
    
    try:
        # 設定ファイルの読み込み
        logger.info(f"設定ファイルを読み込んでいます: {args.config}")
        config = ConfigManager(args.config)
        
        # 設定の検証
        if not config.validate():
            logger.error("設定ファイルの検証に失敗しました")
            return 1
        
        # デバッグモードの場合、設定を上書き
        if args.debug:
            config.set('output.debug_mode', True)
            logger.info("デバッグモードが有効になりました")
        
        # ロギングを再設定（出力ディレクトリを反映）
        output_dir = config.get('output.directory', 'output')
        setup_logging(args.debug, output_dir)
        logger = logging.getLogger(__name__)
        
        # 出力ディレクトリの作成
        output_path = Path(output_dir)
        for subdir in ['detections', 'floormaps', 'graphs', 'labels']:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"出力ディレクトリ: {output_path.absolute()}")
        
        # ファインチューニングモード
        if args.fine_tune:
            logger.warning("ファインチューニングモードは未実装です")
            return 1
        
        # ========================================
        # フェーズ1: フレームサンプリング
        # ========================================
        logger.info("=" * 80)
        logger.info("フェーズ1: フレームサンプリング")
        logger.info("=" * 80)
        
        # 動画処理の初期化
        video_path = config.get('video.input_path')
        logger.info(f"動画ファイル: {video_path}")
        
        video_processor = VideoProcessor(video_path)
        video_processor.open()
        
        # タイムスタンプ抽出器の初期化
        timestamp_extractor = TimestampExtractor()
        if config.get('output.debug_mode', False):
            debug_dir = output_path / 'debug' / 'timestamps'
            timestamp_extractor.enable_debug(debug_dir)
        
        # フレームサンプラーの初期化
        interval_minutes = config.get('video.frame_interval_minutes', 5)
        tolerance_seconds = config.get('video.tolerance_seconds', 10)
        frame_sampler = FrameSampler(interval_minutes, tolerance_seconds)
        
        # フレームサンプリング実行
        logger.info("フレームサンプリングを開始します...")
        sample_frames = frame_sampler.extract_sample_frames(
            video_processor,
            timestamp_extractor,
            start_time=args.start_time,
            end_time=args.end_time
        )
        
        logger.info(f"サンプルフレーム数: {len(sample_frames)}個")
        
        # リソース解放
        video_processor.release()
        video_processor = None
        
        if not sample_frames:
            logger.error("サンプルフレームが抽出できませんでした")
            return 1
        
        # ========================================
        # フェーズ2: ViT人物検出
        # ========================================
        logger.info("=" * 80)
        logger.info("フェーズ2: ViT人物検出")
        logger.info("=" * 80)
        
        # ViT検出器の初期化
        model_name = config.get('detection.model_name')
        confidence_threshold = config.get('detection.confidence_threshold')
        device = config.get('detection.device')
        
        logger.info(f"モデル: {model_name}")
        logger.info(f"信頼度閾値: {confidence_threshold}")
        logger.info(f"デバイス: {device}")
        
        detector = ViTDetector(model_name, confidence_threshold, device)
        detector.load_model()
        
        # 人物検出実行
        detection_results = process_detections(sample_frames, detector, config, logger)
        
        # 統計情報を計算
        total_detections = sum(len(dets) for _, _, dets in detection_results)
        avg_detections = total_detections / len(detection_results) if detection_results else 0
        
        # 信頼度スコアの統計を計算
        all_confidences = []
        for _, _, detections in detection_results:
            for detection in detections:
                all_confidences.append(detection.confidence)
        
        if all_confidences:
            confidence_stats = {
                'total_detections': total_detections,
                'avg_detections_per_frame': float(avg_detections),
                'confidence': {
                    'mean': float(np.mean(all_confidences)),
                    'min': float(np.min(all_confidences)),
                    'max': float(np.max(all_confidences)),
                    'std': float(np.std(all_confidences)),
                    'median': float(np.median(all_confidences))
                },
                'frame_count': len(detection_results)
            }
        else:
            confidence_stats = {
                'total_detections': 0,
                'avg_detections_per_frame': 0.0,
                'confidence': {
                    'mean': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'std': 0.0,
                    'median': 0.0
                },
                'frame_count': len(detection_results)
            }
        
        # 統計情報を表示
        logger.info("=" * 80)
        logger.info("検出統計:")
        logger.info(f"  総検出数: {total_detections}人")
        logger.info(f"  平均検出数: {avg_detections:.2f}人/フレーム")
        if all_confidences:
            logger.info(f"  信頼度スコア統計:")
            logger.info(f"    平均: {confidence_stats['confidence']['mean']:.4f}")
            logger.info(f"    最小: {confidence_stats['confidence']['min']:.4f}")
            logger.info(f"    最大: {confidence_stats['confidence']['max']:.4f}")
            logger.info(f"    標準偏差: {confidence_stats['confidence']['std']:.4f}")
            logger.info(f"    中央値: {confidence_stats['confidence']['median']:.4f}")
        logger.info("=" * 80)
        
        # 統計情報をJSONファイルに出力
        detection_stats_path = output_path / 'detection_statistics.json'
        try:
            with open(detection_stats_path, 'w', encoding='utf-8') as f:
                json.dump(confidence_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"検出統計情報をJSONに出力しました: {detection_stats_path}")
        except Exception as e:
            logger.error(f"検出統計情報のJSON出力に失敗しました: {e}")
        
        # ========================================
        # フェーズ3: 座標変換とゾーン判定
        # ========================================
        logger.info("=" * 80)
        logger.info("フェーズ3: 座標変換とゾーン判定")
        logger.info("=" * 80)
        
        # CoordinateTransformerの初期化
        homography_matrix = config.get('homography.matrix')
        floormap_config = config.get('floormap')
        
        if homography_matrix is None:
            logger.error("ホモグラフィ行列が設定されていません")
            return 1
        
        coordinate_transformer = CoordinateTransformer(homography_matrix, floormap_config)
        logger.info("CoordinateTransformer initialized")
        
        # ZoneClassifierの初期化
        zones = config.get('zones', [])
        if not zones:
            logger.warning("ゾーン定義が設定されていません")
        
        zone_classifier = ZoneClassifier(zones, allow_overlap=False)
        logger.info(
            "ZoneClassifier initialized with %d zones (allow_overlap=%s)",
            len(zones),
            False,
        )
        
        # 座標変換とゾーン判定を実行
        frame_results: List[FrameResult] = []
        
        for frame_num, timestamp, detections in tqdm(detection_results, desc="座標変換・ゾーン判定中"):
            # 各検出結果に対して座標変換とゾーン判定を適用
            for detection in detections:
                # フロアマップ座標に変換（ピクセル単位）
                try:
                    floor_coords = coordinate_transformer.transform(detection.camera_coords)
                    detection.floor_coords = floor_coords
                    
                    # mm単位にも変換
                    floor_coords_mm = coordinate_transformer.pixel_to_mm(floor_coords)
                    detection.floor_coords_mm = floor_coords_mm
                    
                    # 座標がフロアマップ範囲内かチェック
                    if not coordinate_transformer.is_within_bounds(floor_coords):
                        logger.debug(f"座標が範囲外: {floor_coords}")
                    
                    # ゾーン判定
                    zone_ids = zone_classifier.classify(floor_coords)
                    detection.zone_ids = zone_ids
                    
                except Exception as e:
                    logger.error(f"座標変換/ゾーン判定エラー: {e}")
                    detection.floor_coords = None
                    detection.floor_coords_mm = None
                    detection.zone_ids = []
            
            # FrameResultを作成（集計は次のフェーズで実行）
            frame_result = FrameResult(
                frame_number=frame_num,
                timestamp=timestamp,
                detections=detections,
                zone_counts={}  # 後で集計
            )
            frame_results.append(frame_result)
        
        logger.info(f"座標変換とゾーン判定が完了: {len(frame_results)}フレーム")
        
        # 座標変換結果をJSON形式で出力
        coordinate_data = []
        for frame_result in frame_results:
            frame_data = {
                'frame_number': frame_result.frame_number,
                'timestamp': frame_result.timestamp,
                'detections': []
            }
            
            for detection in frame_result.detections:
                detection_data = {
                    'bbox': {
                        'x': float(detection.bbox[0]),
                        'y': float(detection.bbox[1]),
                        'width': float(detection.bbox[2]),
                        'height': float(detection.bbox[3])
                    },
                    'confidence': float(detection.confidence),
                    'camera_coords': {
                        'x': float(detection.camera_coords[0]),
                        'y': float(detection.camera_coords[1])
                    }
                }
                
                # フロアマップ座標が存在する場合のみ追加
                if detection.floor_coords is not None:
                    detection_data['floor_coords'] = {
                        'x': float(detection.floor_coords[0]),
                        'y': float(detection.floor_coords[1])
                    }
                
                if detection.floor_coords_mm is not None:
                    detection_data['floor_coords_mm'] = {
                        'x': float(detection.floor_coords_mm[0]),
                        'y': float(detection.floor_coords_mm[1])
                    }
                
                if detection.zone_ids:
                    detection_data['zone_ids'] = detection.zone_ids
                
                frame_data['detections'].append(detection_data)
            
            coordinate_data.append(frame_data)
        
        # JSONファイルに出力
        coordinate_output_path = output_path / 'coordinate_transformations.json'
        try:
            with open(coordinate_output_path, 'w', encoding='utf-8') as f:
                json.dump(coordinate_data, f, indent=2, ensure_ascii=False)
            logger.info(f"座標変換結果をJSONに出力しました: {coordinate_output_path}")
            logger.info(f"  出力フレーム数: {len(coordinate_data)}")
            logger.info(f"  総検出数: {sum(len(f['detections']) for f in coordinate_data)}")
        except Exception as e:
            logger.error(f"座標変換結果のJSON出力に失敗しました: {e}")
        
        # ========================================
        # フェーズ4: 集計とレポート生成
        # ========================================
        logger.info("=" * 80)
        logger.info("フェーズ4: 集計とレポート生成")
        logger.info("=" * 80)
        
        # Aggregatorの初期化
        aggregator = Aggregator()
        
        # フレームごとに集計
        for frame_result in tqdm(frame_results, desc="集計中"):
            zone_counts = aggregator.aggregate_frame(
                frame_result.timestamp,
                frame_result.detections
            )
            # FrameResultにゾーンカウントを設定
            frame_result.zone_counts = zone_counts
        
        # 統計情報を表示
        statistics = aggregator.get_statistics()
        logger.info("=" * 80)
        logger.info("集計統計:")
        for zone_id, stats in statistics.items():
            zone_name = next((z['name'] for z in zones if z['id'] == zone_id), zone_id)
            logger.info(f"  {zone_name} ({zone_id}):")
            logger.info(f"    平均: {stats['average']:.2f}人")
            logger.info(f"    最大: {stats['max']}人")
            logger.info(f"    最小: {stats['min']}人")
        logger.info("=" * 80)
        
        # CSV出力
        csv_path = output_path / 'zone_counts.csv'
        aggregator.export_csv(str(csv_path))
        logger.info(f"集計結果をCSVに出力しました: {csv_path}")
        
        # ========================================
        # フェーズ5: 可視化
        # ========================================
        logger.info("=" * 80)
        logger.info("フェーズ5: 可視化")
        logger.info("=" * 80)
        
        # Visualizerの初期化
        visualizer = Visualizer(debug_mode=args.debug)
        
        # 時系列グラフの生成
        time_series_path = output_path / 'graphs' / 'time_series.png'
        if visualizer.plot_time_series(aggregator, str(time_series_path)):
            logger.info(f"時系列グラフを生成しました: {time_series_path}")
        
        # 統計グラフの生成
        statistics_path = output_path / 'graphs' / 'statistics.png'
        if visualizer.plot_zone_statistics(aggregator, str(statistics_path)):
            logger.info(f"統計グラフを生成しました: {statistics_path}")
        
        # ヒートマップの生成
        heatmap_path = output_path / 'graphs' / 'heatmap.png'
        if visualizer.plot_heatmap(aggregator, str(heatmap_path)):
            logger.info(f"ヒートマップを生成しました: {heatmap_path}")
        
        # FloormapVisualizerの初期化と可視化
        save_floormap_images = config.get('output.save_floormap_images', True)
        
        if save_floormap_images:
            floormap_path = config.get('floormap.image_path')
            camera_config = config.get('camera', {})
            
            try:
                floormap_visualizer = FloormapVisualizer(
                    floormap_path,
                    floormap_config,
                    zones,
                    camera_config
                )
                
                # 各フレームのフロアマップ画像を生成
                for frame_result in tqdm(frame_results, desc="フロアマップ可視化中"):
                    # フロアマップ上に描画
                    floormap_image = floormap_visualizer.visualize_frame(
                        frame_result,
                        draw_zones=True,
                        draw_labels=True
                    )
                    
                    # 保存
                    floormap_output = output_path / 'floormaps' / f"floormap_{frame_result.timestamp.replace(':', '')}.png"
                    floormap_visualizer.save_visualization(floormap_image, str(floormap_output))
                
                # 凡例を生成
                legend_image = floormap_visualizer.create_legend()
                legend_path = output_path / 'floormaps' / 'legend.png'
                floormap_visualizer.save_visualization(legend_image, str(legend_path))
                logger.info(f"フロアマップ凡例を生成しました: {legend_path}")
                
            except FileNotFoundError as e:
                logger.warning(f"フロアマップ画像が見つかりません: {e}")
            except Exception as e:
                logger.error(f"フロアマップ可視化エラー: {e}", exc_info=True)
        
        # ========================================
        # 精度評価（オプション）
        # ========================================
        if args.evaluate:
            run_evaluation(detection_results, config, logger)
        
        logger.info("=" * 80)
        logger.info("処理が正常に完了しました")
        logger.info("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return 1
    except ValueError as e:
        logger.error(f"設定エラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("処理が中断されました")
        return 130
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
        return 1
    finally:
        import gc
        
        # リソースのクリーンアップ
        if video_processor is not None:
            try:
                video_processor.release()
            except Exception as e:
                logger.error(f"リソース解放中にエラーが発生しました: {e}")
        
        # メモリ解放
        if detector is not None:
            try:
                import torch
                if detector.device in ['mps', 'cuda']:
                    if detector.device == 'mps' and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif detector.device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"メモリ解放中にエラーが発生しました: {e}")
        
        # 大きなデータ構造のクリーンアップ
        try:
            # ローカル変数を削除（大きなデータ構造）
            if 'sample_frames' in locals():
                del sample_frames
            if 'detection_results' in locals():
                del detection_results
            if 'frame_results' in locals():
                del frame_results
            if 'aggregator' in locals():
                del aggregator
            
            # ガベージコレクションを実行
            gc.collect()
            logger.debug("メモリクリーンアップを実行しました")
        except Exception as e:
            logger.debug(f"メモリクリーンアップ中にエラーが発生しました: {e}")


if __name__ == "__main__":
    sys.exit(main())

