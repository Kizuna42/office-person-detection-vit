"""Timestamp OCR-only mode."""

import csv
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from src.pipeline.base_phase import BasePhase
from src.utils import create_timestamp_overlay
from src.video import VideoProcessor


class TimestampOCRMode(BasePhase):
    """タイムスタンプOCRのみを実行するモード"""

    def __init__(self, config, logger):
        """初期化

        Args:
            config: ConfigManagerインスタンス
            logger: ロガー
        """
        super().__init__(config, logger)
        self.video_processor: Optional[VideoProcessor] = None

    def execute(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> int:
        """タイムスタンプOCRのみを実行

        Args:
            start_time: 開始時刻（HH:MM形式、オプション）
            end_time: 終了時刻（HH:MM形式、オプション）

        Returns:
            終了コード（0=成功、1=失敗）
        """
        self.logger.info("=" * 80)
        self.logger.info("タイムスタンプOCRモード")
        self.logger.info("=" * 80)

        output_dir = Path(self.config.get("output.directory", "output")) / "timestamps"
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"出力ディレクトリ: {output_dir.absolute()}")

        try:
            # 共通の初期化処理を使用
            (
                self.video_processor,
                timestamp_extractor,
                frame_sampler,
            ) = self._setup_frame_sampling_components()

            roi = timestamp_extractor.roi
            interval_minutes = self.config.get("video.frame_interval_minutes", 5)
            tolerance_seconds = self.config.get("video.tolerance_seconds", 10)

            self.logger.info(f"サンプリング間隔: {interval_minutes}分")
            self.logger.info(f"許容誤差: ±{tolerance_seconds}秒")

            # フレームサンプリング実行
            self.logger.info("フレームサンプリングを開始します...")
            margin_minutes = self.config.get("video.scan_margin_minutes", 10)
            sample_frames = frame_sampler.extract_sample_frames(
                self.video_processor,
                timestamp_extractor,
                start_time=start_time,
                end_time=end_time,
                margin_minutes=margin_minutes,
            )

            self.logger.info(f"サンプルフレーム数: {len(sample_frames)}個")

            if not sample_frames:
                self.logger.error("サンプルフレームが抽出できませんでした")
                return 1

            # 各サンプルフレームに対してOCR実行
            results = []
            recognized_count = 0
            total_count = len(sample_frames)

            self.logger.info("タイムスタンプOCRを実行します...")
            for frame_num, ts_str, frame in tqdm(sample_frames, desc="OCR実行中"):
                # OCR実行
                timestamp, confidence = timestamp_extractor.extract_with_confidence(
                    frame
                )

                recognized = timestamp is not None
                if recognized:
                    recognized_count += 1

                # オーバーレイ画像作成
                overlay = create_timestamp_overlay(
                    frame, roi, timestamp, confidence, frame_num
                )
                overlay_filename = f"frame_{frame_num:06d}_overlay.png"
                overlay_path = overlays_dir / overlay_filename
                cv2.imwrite(str(overlay_path), overlay)

                # 結果を記録
                result = {
                    "frame_number": frame_num,
                    "timestamp": timestamp if timestamp else "",
                    "confidence": confidence,
                    "recognized": recognized,
                    "overlay_path": f"overlays/{overlay_filename}",
                    "roi_x": roi[0],
                    "roi_y": roi[1],
                    "roi_width": roi[2],
                    "roi_height": roi[3],
                }
                results.append(result)

            # CSV出力
            csv_path = output_dir / "frames_ocr.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_number",
                        "timestamp",
                        "confidence",
                        "recognized",
                        "overlay_path",
                        "roi_x",
                        "roi_y",
                        "roi_width",
                        "roi_height",
                    ],
                )
                writer.writeheader()
                for result in results:
                    writer.writerow(result)

            self.logger.info(f"CSV出力完了: {csv_path}")

            # 最終結果表示
            recognition_rate = (
                recognized_count / total_count if total_count > 0 else 0.0
            )
            self.logger.info("=" * 80)
            self.logger.info("タイムスタンプOCR結果")
            self.logger.info("=" * 80)
            self.logger.info(f"総フレーム数: {total_count}")
            self.logger.info(f"認識成功: {recognized_count}")
            self.logger.info(f"認識失敗: {total_count - recognized_count}")
            self.logger.info(f"認識率: {recognition_rate:.2%}")
            self.logger.info(f"ROI座標: ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]})")
            self.logger.info("=" * 80)

            # 代表サンプルを表示
            success_samples = [r for r in results if r["recognized"]][:3]
            failed_samples = [r for r in results if not r["recognized"]][:3]

            if success_samples:
                self.logger.info("\n代表成功サンプル（最大3件）:")
                for sample in success_samples:
                    self.logger.info(
                        f"  フレーム {sample['frame_number']}: {sample['timestamp']} "
                        f"(信頼度={sample['confidence']:.2f}) "
                        f"- {sample['overlay_path']}"
                    )

            if failed_samples:
                self.logger.info("\n代表失敗サンプル（最大3件）:")
                for sample in failed_samples:
                    self.logger.info(
                        f"  フレーム {sample['frame_number']}: 失敗 "
                        f"(信頼度={sample['confidence']:.2f}) "
                        f"- {sample['overlay_path']}"
                    )

            self.logger.info("=" * 80)
            self.logger.info("処理が正常に完了しました")
            self.logger.info("=" * 80)

            return 0

        except FileNotFoundError as e:
            self.logger.error(f"ファイルが見つかりません: {e}")
            return 1
        except ValueError as e:
            self.logger.error(f"設定エラー: {e}")
            return 1
        except KeyboardInterrupt:
            self.logger.warning("処理が中断されました")
            return 130
        except Exception as e:
            self.logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
            return 1
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """リソースをクリーンアップ"""
        if self.video_processor is not None:
            try:
                self.video_processor.release()
            except Exception as e:
                self.logger.error(f"リソース解放中にエラーが発生しました: {e}")
            finally:
                self.video_processor = None
