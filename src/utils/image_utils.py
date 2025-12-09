"""Image processing and saving utilities."""

from datetime import datetime
import logging
from pathlib import Path

import cv2
import numpy as np

from src.models import Detection


def get_track_id_color(track_id: int) -> tuple[int, int, int]:
    """track_idに基づいて色を生成（phase5のfloormap可視化と同じ計算式）

    Args:
        track_id: トラックID

    Returns:
        BGR色空間の色タプル (B, G, R)
    """
    # 黄金角を使用して色を分散（phase5のFloormapVisualizerと同じロジック）
    hue = (track_id * 137) % 180
    color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))


def _sanitize_timestamp(timestamp: str) -> str:
    """タイムスタンプ文字列をファイル名安全な形式に正規化する。

    期待形式: YYYY/MM/DD HH:MM:SS など。どのセパレータでも、数字を抽出して
    `YYYY_MM_DD_HHMMSS` に整形する。
    """
    digits = "".join(ch for ch in timestamp if ch.isdigit())
    if len(digits) >= 14:
        try:
            dt = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
            return dt.strftime("%Y_%m_%d_%H%M%S")
        except ValueError:
            pass
    # フォールバック: 従来の置換ロジック（必ずアンダースコアを入れる）
    ts = timestamp.replace("/", "_").replace("-", "_").replace(":", "").replace(" ", "_")
    return "".join(c for c in ts if c.isalnum() or c in "_-.")


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
            timestamp_clean = _sanitize_timestamp(timestamp)
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


def save_tracked_detection_image(
    frame: np.ndarray,
    detections: list[Detection],
    timestamp: str,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """ID付き検出結果を画像として保存（IDごとに色分け）

    Args:
        frame: 入力フレーム
        detections: 検出結果のリスト（track_idが含まれる）
        timestamp: タイムスタンプ
        output_dir: 出力ディレクトリ
        logger: ロガー
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"save_tracked_detection_image: output_dir={output_dir}, timestamp={timestamp}, detections={len(detections)}"
        )

        # フレームをコピー（描画用）
        result_image = frame.copy()

        try:
            # ラベルの重なりを避けるための既存ラベル位置を記録
            used_label_rects: list[tuple[int, int, int, int]] = []  # (x, y, width, height)

            def get_text_size(text: str, font_scale: float, thickness: int) -> tuple[int, int]:
                """テキストのサイズを取得"""
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                return text_width, text_height + baseline

            def is_overlapping(rect1: tuple[int, int, int, int], rect2: tuple[int, int, int, int]) -> bool:
                """2つの矩形が重なっているか判定"""
                x1, y1, w1, h1 = rect1
                x2, y2, w2, h2 = rect2
                return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

            def find_label_position(
                bbox_x: int,
                bbox_y: int,
                bbox_w: int,
                bbox_h: int,
                text_width: int,
                text_height: int,
                image_width: int,
                image_height: int,
            ) -> tuple[int, int]:
                """ラベルの配置位置を決定（重なりを避ける）"""
                padding = 5

                # 候補位置のリスト（優先順位順）
                # 注意: cv2.putTextのy座標はベースライン位置（テキストの下端）
                candidates = [
                    # バウンディングボックスの上端中央
                    (bbox_x + bbox_w // 2 - text_width // 2, bbox_y - padding),
                    # バウンディングボックスの下端中央
                    (bbox_x + bbox_w // 2 - text_width // 2, bbox_y + bbox_h + text_height + padding),
                    # バウンディングボックスの左端中央（ベースラインを中央に合わせる）
                    (bbox_x - text_width - padding, bbox_y + bbox_h // 2 + text_height // 2),
                    # バウンディングボックスの右端中央（ベースラインを中央に合わせる）
                    (bbox_x + bbox_w + padding, bbox_y + bbox_h // 2 + text_height // 2),
                    # バウンディングボックスの左上
                    (bbox_x + padding, bbox_y - padding),
                    # バウンディングボックスの右上
                    (bbox_x + bbox_w - text_width - padding, bbox_y - padding),
                    # バウンディングボックスの左下
                    (bbox_x + padding, bbox_y + bbox_h + text_height + padding),
                    # バウンディングボックスの右下
                    (bbox_x + bbox_w - text_width - padding, bbox_y + bbox_h + text_height + padding),
                ]

                # 各候補位置をチェック
                for label_x, label_y in candidates:
                    # 画像範囲内かチェック
                    # label_yはベースライン位置（テキストの下端）なので、
                    # テキスト全体が画像内に収まるようにチェック
                    if (
                        label_x < 0
                        or label_x + text_width > image_width
                        or label_y < text_height  # 上端が画像範囲内
                        or label_y > image_height  # 下端が画像範囲内
                    ):
                        continue

                    # ラベルの矩形領域（y座標はベースライン位置なので、上端を計算）
                    label_rect = (label_x, label_y - text_height, text_width, text_height)

                    # 既存のラベルと重なっているかチェック
                    overlaps = False
                    for used_rect in used_label_rects:
                        if is_overlapping(label_rect, used_rect):
                            overlaps = True
                            break

                    if not overlaps:
                        # 重なっていない位置が見つかった
                        used_label_rects.append(label_rect)
                        return label_x, label_y

                # すべての候補が重なっている場合は、最初の候補を使用（強制的に配置）
                if candidates:
                    label_x, label_y = candidates[0]
                    # 画像範囲内に調整
                    label_x = max(0, min(label_x, image_width - text_width))
                    label_y = max(text_height, min(label_y, image_height))
                    # ラベルの矩形領域（y座標はベースライン位置なので、上端を計算）
                    used_label_rects.append((label_x, label_y - text_height, text_width, text_height))
                    return label_x, label_y

                # フォールバック: バウンディングボックスの上端
                return bbox_x, bbox_y - padding

            # バウンディングボックスを描画
            for detection in detections:
                x, y, w, h = detection.bbox
                x, y, w, h = int(x), int(y), int(w), int(h)

                # track_idに基づいて色を決定
                color = (
                    get_track_id_color(detection.track_id) if detection.track_id is not None else (0, 255, 0)
                )  # デフォルトは緑

                # ボックスを描画（ID色で描画、線幅3px）
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)

                # ラベルを表示（IDのみ）
                label = f"ID:{detection.track_id}" if detection.track_id is not None else "Person"

                # ラベルのフォント設定
                font_scale = 0.6
                thickness = 2
                text_width, text_height = get_text_size(label, font_scale, thickness)

                # 重ならない位置を決定
                label_x, label_y = find_label_position(
                    x, y, w, h, text_width, text_height, result_image.shape[1], result_image.shape[0]
                )

                # ラベルを描画（白のアウトライン付き）
                cv2.putText(
                    result_image,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # 白のアウトライン
                    thickness + 1,
                )
                cv2.putText(
                    result_image,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,  # ID色
                    thickness,
                )

                # 足元座標を描画（ID色の円）
                foot_x, foot_y = detection.camera_coords
                cv2.circle(result_image, (int(foot_x), int(foot_y)), 5, color, -1)

            # ファイル名を生成（タイムスタンプの特殊文字を置換）
            timestamp_clean = _sanitize_timestamp(timestamp)
            filename = f"tracking_{timestamp_clean}.jpg"

            # Pathオブジェクトで安全に結合
            output_path = output_dir / filename

            # 安全チェック
            if not str(output_path.resolve()).startswith(str(output_dir.resolve())):
                raise ValueError(f"安全上の理由により、出力パスが出力ディレクトリ外を指しています: {output_path}")

            logger.debug(f"保存先パス: {output_path}, ファイル名: {filename}, 元のタイムスタンプ: {timestamp}")

            # 保存
            success = cv2.imwrite(str(output_path), result_image)
            if success:
                logger.info(f"ID付き検出画像を保存しました: {output_path}")
            else:
                logger.error(f"ID付き検出画像の保存に失敗しました: {output_path}")
                logger.error(f"  - 出力ディレクトリ: {output_dir}")
                logger.error(f"  - ファイル名: {filename}")
                logger.error(f"  - 画像サイズ: {result_image.shape}")
        finally:
            # 描画用画像の参照を削除（メモリ節約）
            del result_image

    except Exception as e:
        logger.error(f"ID付き検出画像の保存に失敗しました: {e}", exc_info=True)
        logger.error(f"  - output_dir: {output_dir}")
        logger.error(f"  - timestamp: {timestamp}")
        logger.error(f"  - detections count: {len(detections) if detections else 0}")
