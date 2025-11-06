"""Image processing and saving utilities."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.models import Detection


def save_detection_image(
    frame: np.ndarray,
    detections: List[Detection],
    timestamp: str,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """検出結果を画像として保存

    Args:
        frame: 入力フレーム
        detections: 検出結果のリスト
        timestamp: タイムスタンプ
        output_dir: 出力ディレクトリ
        logger: ロガー
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"save_detection_image: output_dir={output_dir}, timestamp={timestamp}, detections={len(detections)}"
        )

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
                    result_image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # 足元座標を描画
                foot_x, foot_y = detection.camera_coords
                cv2.circle(result_image, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)

            # ファイル名を生成（タイムスタンプの特殊文字を置換）
            # / と : と スペースを _ に置換（ファイル名として無効な文字を除去）
            timestamp_clean = (
                timestamp.replace("/", "_").replace(":", "").replace(" ", "_")
            )
            filename = f"detection_{timestamp_clean}.jpg"
            output_path = output_dir / filename

            logger.debug(f"保存先パス: {output_path}, ファイル名: {filename}")

            # 保存
            success = cv2.imwrite(str(output_path), result_image)
            if success:
                logger.info(f"検出画像を保存しました: {output_path}")
            else:
                logger.error(f"検出画像の保存に失敗しました: {output_path}")
                logger.error(f"  - 出力ディレクトリ: {output_dir}")
                logger.error(f"  - ファイル名: {filename}")
                logger.error(f"  - 画像サイズ: {result_image.shape}")
        finally:
            # 描画用画像の参照を削除（メモリ節約）
            del result_image

    except Exception as e:
        logger.error(f"検出画像の保存に失敗しました: {e}", exc_info=True)
        logger.error(f"  - output_dir: {output_dir}")
        logger.error(f"  - timestamp: {timestamp}")
        logger.error(f"  - detections count: {len(detections) if detections else 0}")


def create_timestamp_overlay(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    timestamp: Optional[str],
    confidence: float,
    frame_number: int,
) -> np.ndarray:
    """ROI矩形と抽出結果をオーバーレイした画像を作成

    Args:
        frame: 元のフレーム画像
        roi: ROI座標 (x, y, width, height)
        timestamp: 抽出されたタイムスタンプ
        confidence: OCR信頼度
        frame_number: フレーム番号

    Returns:
        オーバーレイ画像
    """
    overlay = frame.copy()
    x, y, w, h = roi

    # ROI矩形を描画
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ROI座標をテキストで表示
    roi_text = f"ROI: ({x}, {y}, {w}, {h})"
    cv2.putText(
        overlay,
        roi_text,
        (x, max(y - 30, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # フレーム番号を表示
    frame_text = f"Frame: {frame_number:06d}"
    cv2.putText(
        overlay,
        frame_text,
        (x, max(y - 60, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # 抽出結果を表示
    if timestamp:
        result_text = f"Timestamp: {timestamp}"
        conf_text = f"Confidence: {confidence:.2f}"
        color = (0, 255, 0)  # 成功時は緑
    else:
        result_text = "Timestamp: FAILED"
        conf_text = f"Confidence: {confidence:.2f}"
        color = (0, 0, 255)  # 失敗時は赤

    cv2.putText(
        overlay, result_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )
    cv2.putText(
        overlay, conf_text, (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

    return overlay
