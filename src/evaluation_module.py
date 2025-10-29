"""Evaluation module for person detection accuracy assessment."""

import json
import csv
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.data_models import Detection, EvaluationMetrics

logger = logging.getLogger(__name__)


class EvaluationModule:
    """精度評価クラス
    
    Ground Truthデータと検出結果を比較し、精度指標を計算する。
    
    Attributes:
        ground_truth_path: Ground Truthデータファイルのパス
        iou_threshold: IoU閾値（True Positiveの判定基準）
        ground_truth: 読み込まれたGround Truthデータ
    """
    
    def __init__(self, ground_truth_path: str, iou_threshold: float = 0.5):
        """
        Args:
            ground_truth_path: Ground Truthデータファイルのパス（COCO形式JSON）
            iou_threshold: IoU閾値（デフォルト: 0.5）
        """
        self.ground_truth_path = ground_truth_path
        self.iou_threshold = iou_threshold
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self) -> Dict:
        """Ground Truthデータを読み込む
        
        Returns:
            Ground Truthデータ（COCO形式）
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            json.JSONDecodeError: JSON形式が不正な場合
        """
        try:
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Ground Truthデータを読み込みました: {self.ground_truth_path}")
            logger.info(f"画像数: {len(data.get('images', []))}, "
                       f"アノテーション数: {len(data.get('annotations', []))}")
            
            return data
            
        except FileNotFoundError:
            logger.error(f"Ground Truthファイルが見つかりません: {self.ground_truth_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Ground TruthファイルのJSON形式が不正です: {e}")
            raise
    
    def _get_annotations_by_image(self, image_id: int) -> List[Dict]:
        """指定された画像IDのアノテーションを取得
        
        Args:
            image_id: 画像ID
            
        Returns:
            アノテーションのリスト
        """
        annotations = []
        for ann in self.ground_truth.get('annotations', []):
            if ann['image_id'] == image_id:
                annotations.append(ann)
        return annotations
    
    def _get_image_by_filename(self, filename: str) -> Optional[Dict]:
        """ファイル名から画像情報を取得
        
        Args:
            filename: ファイル名
            
        Returns:
            画像情報、見つからない場合はNone
        """
        for img in self.ground_truth.get('images', []):
            if img['file_name'] == filename:
                return img
        return None
    
    def calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """2つのバウンディングボックスのIoU（Intersection over Union）を計算
        
        Args:
            bbox1: バウンディングボックス1 (x, y, width, height)
            bbox2: バウンディングボックス2 (x, y, width, height)
            
        Returns:
            IoU値（0.0～1.0）
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 各ボックスの右下座標を計算
        x1_max = x1 + w1
        y1_max = y1 + h1
        x2_max = x2 + w2
        y2_max = y2 + h2
        
        # 交差領域の座標を計算
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        # 交差領域の面積を計算
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        
        # 各ボックスの面積を計算
        area1 = w1 * h1
        area2 = w2 * h2
        
        # 和集合の面積を計算
        union_area = area1 + area2 - inter_area
        
        # IoUを計算（ゼロ除算を回避）
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou
    
    def evaluate(self, detections: Dict[str, List[Detection]]) -> EvaluationMetrics:
        """検出結果を評価し、精度指標を計算
        
        Args:
            detections: 検出結果の辞書 {filename: [Detection, ...]}
            
        Returns:
            評価指標（Precision, Recall, F1-score等）
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # 各画像について評価
        for filename, det_list in detections.items():
            # Ground Truthから画像情報を取得
            image_info = self._get_image_by_filename(filename)
            if image_info is None:
                logger.warning(f"Ground Truthに画像が見つかりません: {filename}")
                continue
            
            # Ground Truthのアノテーションを取得
            gt_annotations = self._get_annotations_by_image(image_info['id'])
            
            # マッチング済みのGround Truthアノテーションを追跡
            matched_gt = set()
            
            # 各検出結果について、最もIoUが高いGround Truthとマッチング
            for detection in det_list:
                best_iou = 0.0
                best_gt_idx = -1
                
                for idx, gt_ann in enumerate(gt_annotations):
                    # 人物クラス（category_id=0）のみを対象
                    if gt_ann['category_id'] != 0:
                        continue
                    
                    # IoUを計算
                    iou = self.calculate_iou(detection.bbox, tuple(gt_ann['bbox']))
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                
                # IoU閾値以上で、まだマッチングされていないGround Truthとマッチング
                if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
            
            # マッチングされなかったGround Truthは偽陰性
            person_gt_count = sum(1 for ann in gt_annotations if ann['category_id'] == 0)
            false_negatives += person_gt_count - len(matched_gt)
        
        # 精度指標を計算
        metrics = self.calculate_metrics(true_positives, false_positives, false_negatives)
        
        logger.info(f"評価完了 - TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
        logger.info(f"Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}, "
                   f"F1-score: {metrics.f1_score:.4f}")
        
        return metrics
    
    def calculate_metrics(self, tp: int, fp: int, fn: int) -> EvaluationMetrics:
        """Precision, Recall, F1-scoreを計算
        
        Args:
            tp: True Positiveの数
            fp: False Positiveの数
            fn: False Negativeの数
            
        Returns:
            評価指標
        """
        # Precisionを計算（ゼロ除算を回避）
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recallを計算（ゼロ除算を回避）
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-scoreを計算（ゼロ除算を回避）
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            confidence_threshold=0.5  # デフォルト値、実際の値は外部から設定可能
        )
    
    def export_report(self, metrics: EvaluationMetrics, output_path: str, 
                     format: str = 'csv') -> None:
        """評価レポートをファイルに出力
        
        Args:
            metrics: 評価指標
            output_path: 出力ファイルパス
            format: 出力形式（'csv' または 'json'）
            
        Raises:
            ValueError: 不正な出力形式が指定された場合
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            self._export_csv(metrics, output_path)
        elif format == 'json':
            self._export_json(metrics, output_path)
        else:
            raise ValueError(f"不正な出力形式: {format}。'csv'または'json'を指定してください。")
        
        logger.info(f"評価レポートを出力しました: {output_path}")
    
    def _export_csv(self, metrics: EvaluationMetrics, output_path: str) -> None:
        """評価レポートをCSV形式で出力
        
        Args:
            metrics: 評価指標
            output_path: 出力ファイルパス
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow(['Metric', 'Value'])
            
            # データ
            writer.writerow(['Precision', f'{metrics.precision:.4f}'])
            writer.writerow(['Recall', f'{metrics.recall:.4f}'])
            writer.writerow(['F1-score', f'{metrics.f1_score:.4f}'])
            writer.writerow(['True Positives', metrics.true_positives])
            writer.writerow(['False Positives', metrics.false_positives])
            writer.writerow(['False Negatives', metrics.false_negatives])
            writer.writerow(['Confidence Threshold', metrics.confidence_threshold])
            writer.writerow(['IoU Threshold', self.iou_threshold])
    
    def _export_json(self, metrics: EvaluationMetrics, output_path: str) -> None:
        """評価レポートをJSON形式で出力
        
        Args:
            metrics: 評価指標
            output_path: 出力ファイルパス
        """
        report = {
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'true_positives': metrics.true_positives,
            'false_positives': metrics.false_positives,
            'false_negatives': metrics.false_negatives,
            'confidence_threshold': metrics.confidence_threshold,
            'iou_threshold': self.iou_threshold
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
