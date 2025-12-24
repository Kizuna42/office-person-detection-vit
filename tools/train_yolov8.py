#!/usr/bin/env python3
"""YOLOv8 人物検出モデルのFine-tuningスクリプト.

事前学習済みYOLOv8xをドメイン特化データでFine-tuningする。
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_yolov8(
    data_yaml: str,
    model_name: str = "yolov8x.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    project: str = "runs/detect",
    name: str = "person_detection",
) -> Path:
    """YOLOv8をFine-tuning.

    Args:
        data_yaml: データセット定義YAMLのパス
        model_name: 事前学習済みモデル名
        epochs: エポック数
        imgsz: 入力画像サイズ
        batch: バッチサイズ
        project: 出力プロジェクトディレクトリ
        name: 実験名

    Returns:
        学習済みモデルのパス
    """
    logger.info(f"モデルをロード: {model_name}")
    model = YOLO(model_name)

    logger.info("Fine-tuning開始")
    logger.info(f"  データ: {data_yaml}")
    logger.info(f"  エポック: {epochs}")
    logger.info(f"  画像サイズ: {imgsz}")
    logger.info(f"  バッチサイズ: {batch}")

    _results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        # 学習パラメータ
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # データ拡張
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # その他
        close_mosaic=10,  # 最後10エポックはMosaicオフ
        patience=20,  # Early stopping
        save=True,
        save_period=10,
        cache=True,
        device=None,  # 自動選択
        workers=4,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  # 単一クラス（person）
        plots=True,
    )

    best_model_path = Path(project) / name / "weights" / "best.pt"
    logger.info(f"学習完了! 最良モデル: {best_model_path}")

    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Fine-tuning")
    parser.add_argument("--data", required=True, help="data.yaml path")
    parser.add_argument("--model", default="yolov8x.pt", help="事前学習モデル")
    parser.add_argument("--epochs", type=int, default=100, help="エポック数")
    parser.add_argument("--imgsz", type=int, default=640, help="画像サイズ")
    parser.add_argument("--batch", type=int, default=8, help="バッチサイズ")
    parser.add_argument("--project", default="runs/detect", help="出力ディレクトリ")
    parser.add_argument("--name", default="person_detection", help="実験名")
    args = parser.parse_args()

    train_yolov8(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
