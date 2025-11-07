"""Image processing and saving utilities."""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.models import Detection


def save_detection_image(
    frame: np.ndarray,
    detections: list[Detection],
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
            # ファイル名として無効な文字を全て除去
            # / と : と スペースを _ に置換し、Pathオブジェクトで安全に処理
            timestamp_clean = timestamp.replace("/", "_").replace(":", "").replace(" ", "_")
            # 念のため、残っている可能性のある特殊文字も除去
            # Windows/Mac/Linuxで無効な文字: / \ : * ? " < > |
            timestamp_clean = "".join(c for c in timestamp_clean if c.isalnum() or c in "_-.")
            filename = f"detection_{timestamp_clean}.jpg"

            # Pathオブジェクトで安全に結合（ファイル名に/が含まれていても正しく処理）
            # Pathオブジェクトは自動的にパスセパレータを処理するため、安全
            output_path = output_dir / filename

            # さらに安全のため、親ディレクトリがoutput_dirであることを確認
            if not str(output_path.resolve()).startswith(str(output_dir.resolve())):
                raise ValueError(f"安全上の理由により、出力パスが出力ディレクトリ外を指しています: {output_path}")

            logger.debug(f"保存先パス: {output_path}, ファイル名: {filename}, 元のタイムスタンプ: {timestamp}")

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
