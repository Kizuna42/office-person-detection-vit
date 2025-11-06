"""動画先頭100フレームでフレーム抽出パイプラインをテストするスクリプト

動画の先頭100フレームのみを処理対象として、タイムスタンプ抽出と
フレーム抽出パイプラインを実行し、期待通り3枚のフレームが保存されるか
動作確認を行います。
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.config import ConfigManager
from src.pipeline.frame_extraction_pipeline import FrameExtractionPipeline
from src.utils import setup_logging, setup_output_directories
from src.video.frame_sampler import CoarseSampler

logger = logging.getLogger(__name__)


class LimitedCoarseSampler(CoarseSampler):
    """最大フレーム数を制限したCoarseSampler"""

    def __init__(
        self, video_path: str, interval_seconds: float = 10.0, max_frames: int = None
    ):
        """LimitedCoarseSamplerを初期化

        Args:
            video_path: 動画ファイルのパス
            interval_seconds: サンプリング間隔（秒）
            max_frames: 最大処理フレーム数（Noneの場合は制限なし）
        """
        super().__init__(video_path, interval_seconds)
        self.max_frames = max_frames

    def sample(self):
        """フレームをサンプリング（最大フレーム数制限付き）

        Yields:
            (フレーム番号, フレーム画像) のタプル
        """
        self._ensure_opened()
        frame_idx = 0
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # 最大フレーム数を適用
        if self.max_frames is not None:
            total_frames = min(total_frames, self.max_frames)

        while frame_idx < total_frames:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video.read()

            if not ret or frame is None:
                break

            yield frame_idx, frame
            frame_idx += self.interval_frames


def test_frame_extraction_100frames(
    config_path: str = "config.yaml", max_frames: int = 100
):
    """動画先頭100フレームでフレーム抽出パイプラインをテスト

    Args:
        config_path: 設定ファイルのパス
        max_frames: 最大処理フレーム数（デフォルト: 100）
    """
    # 設定の読み込み
    config = ConfigManager(config_path)
    output_dir = Path(config.get("output.directory", "output")).resolve()
    test_output_dir = output_dir / "extracted_frames_test"
    setup_output_directories(test_output_dir)

    # ロギング設定
    setup_logging(debug_mode=True, output_dir=str(output_dir))
    logger.info("=" * 80)
    logger.info(f"フレーム抽出パイプラインテスト開始（先頭{max_frames}フレーム）")
    logger.info("=" * 80)

    video_path = config.get("video.input_path")
    if not Path(video_path).exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return False

    # 動画の情報を取得
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"動画ファイルを開けません: {video_path}")
        return False

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    logger.info(f"動画情報: 総フレーム数={total_video_frames}, FPS={fps:.2f}")
    logger.info(f"処理対象: 先頭{max_frames}フレーム（{max_frames/fps:.2f}秒分）")

    # 設定からパラメータを取得
    timestamp_config = config.get("timestamp", {})
    extraction_config = timestamp_config.get("extraction", {})
    sampling_config = timestamp_config.get("sampling", {})
    target_config = timestamp_config.get("target", {})
    ocr_config = config.get("ocr", {})

    # 開始・終了日時の取得（動画先頭100フレームに対応する時間範囲を設定）
    # 100フレーム / 30fps ≈ 3.33秒
    # タイムラプス動画なので、実際の時間はもっと長い可能性がある
    # とりあえず、動画の最初のタイムスタンプを取得するために
    # 先頭フレームからタイムスタンプを読み取る必要がある

    # 先頭フレームのタイムスタンプを取得
    from src.timestamp.timestamp_extractor_v2 import TimestampExtractorV2
    from src.video import VideoProcessor

    video_processor = VideoProcessor(video_path)
    video_processor.open()

    try:
        first_frame = video_processor.get_frame(0)
        if first_frame is None:
            logger.error("先頭フレームを取得できませんでした")
            return False

        # テスト用: 信頼度閾値を下げる（0.5に設定）
        test_confidence_threshold = 0.5
        extractor = TimestampExtractorV2(
            confidence_threshold=test_confidence_threshold,
            roi_config=extraction_config.get("roi"),
            fps=fps,
            enabled_ocr_engines=ocr_config.get("engines"),
            use_improved_validator=True,  # テスト用に改善版バリデーターを使用
            base_tolerance_seconds=300.0,  # タイムラプス動画用に許容範囲を広げる（5分）
        )

        first_result = extractor.extract(first_frame, 0)
        if not first_result or not first_result.get("timestamp"):
            logger.error("先頭フレームからタイムスタンプを読み取れませんでした")
            logger.info("デフォルトの開始時刻を使用します: 2025-08-26 16:05:00")
            start_datetime = datetime(2025, 8, 26, 16, 5, 0)
        else:
            start_datetime = first_result["timestamp"]
            logger.info(f"先頭フレームのタイムスタンプ: {start_datetime}")

        # 100フレーム分の時間を計算（タイムラプス動画の時間圧縮率を考慮）
        # 100フレーム / 30fps ≈ 3.33秒（動画時間）
        # タイムラプス動画なので、実際の時間はもっと長い
        # とりあえず、5分刻みで3枚のフレームを抽出するため、
        # 開始時刻から15分後までを範囲とする
        end_datetime = start_datetime + timedelta(minutes=15)
        logger.info(f"抽出範囲: {start_datetime} ～ {end_datetime}")

    finally:
        video_processor.release()

    # パイプラインを初期化（LimitedCoarseSamplerを使用するため、カスタム実装が必要）
    # 既存のFrameExtractionPipelineを拡張するか、直接実装する

    # 簡易版: FrameExtractionPipelineを直接使用し、
    # CoarseSamplerをLimitedCoarseSamplerに置き換える
    pipeline = FrameExtractionPipeline(
        video_path=video_path,
        output_dir=str(test_output_dir),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        interval_minutes=config.get("video.frame_interval_minutes", 5),
        tolerance_seconds=300.0,  # タイムラプス動画用に許容範囲を広げる（5分）
        confidence_threshold=0.5,  # テスト用に信頼度閾値を下げる
        coarse_interval_seconds=sampling_config.get("coarse_interval_seconds", 2.0),
        fine_search_window_seconds=sampling_config.get("search_window_seconds", 60.0),
        fine_interval_seconds=sampling_config.get("fine_interval_seconds", 0.1),
        fps=config.get("video.fps", 30.0),
        roi_config=extraction_config.get("roi"),
        enabled_ocr_engines=ocr_config.get("engines"),
        use_improved_validator=True,  # テスト用に改善版バリデーターを使用
        base_tolerance_seconds=300.0,  # タイムラプス動画用に許容範囲を広げる（5分）
        history_size=timestamp_config.get("validator", {}).get("history_size", 10),
        z_score_threshold=timestamp_config.get("validator", {}).get(
            "z_score_threshold", 2.0
        ),
        use_weighted_consensus=extraction_config.get("use_weighted_consensus", False),
        use_voting_consensus=extraction_config.get("use_voting_consensus", False),
    )

    # CoarseSamplerをLimitedCoarseSamplerに置き換え
    pipeline.coarse_sampler = LimitedCoarseSampler(
        video_path,
        interval_seconds=sampling_config.get("coarse_interval_seconds", 2.0),
        max_frames=max_frames,
    )

    # パイプライン実行
    try:
        extraction_results = pipeline.run()

        # 結果を確認
        logger.info("=" * 80)
        logger.info("テスト結果")
        logger.info("=" * 80)
        logger.info(f"抽出成功フレーム数: {len(extraction_results)}")

        # 保存されたフレーム画像を確認
        saved_frames = list(test_output_dir.glob("frame_*.jpg"))
        logger.info(f"保存されたフレーム画像数: {len(saved_frames)}")

        for frame_path in saved_frames:
            logger.info(f"  - {frame_path.name}")

        # CSVファイルの確認
        csv_path = test_output_dir / "extraction_results.csv"
        if csv_path.exists():
            logger.info(f"抽出結果CSV: {csv_path}")
            import csv

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                logger.info(f"CSVレコード数: {len(rows)}")
                for i, row in enumerate(rows[:5], 1):  # 最初の5件を表示
                    logger.info(
                        f"  [{i}] {row.get('target_timestamp')} -> {row.get('extracted_timestamp')} "
                        f"(diff={row.get('time_diff_seconds')}s, conf={row.get('confidence')})"
                    )

        # 期待値との比較
        expected_frames = 3  # 5分刻みで15分間 = 3枚（開始時刻、+5分、+10分）
        if len(extraction_results) == expected_frames:
            logger.info(f"✅ 期待通り{expected_frames}枚のフレームが抽出されました")
            return True
        else:
            logger.warning(
                f"⚠️  期待値: {expected_frames}枚、実際: {len(extraction_results)}枚"
            )
            return False

    except Exception as e:
        logger.error(f"パイプライン実行中にエラーが発生しました: {e}", exc_info=True)
        return False
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="動画先頭100フレームでフレーム抽出パイプラインをテスト")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )
    parser.add_argument(
        "--max-frames", type=int, default=100, help="最大処理フレーム数（デフォルト: 100）"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    success = test_frame_extraction_100frames(args.config, args.max_frames)
    sys.exit(0 if success else 1)
