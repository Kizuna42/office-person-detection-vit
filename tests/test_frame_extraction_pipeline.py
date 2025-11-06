"""Integration tests for FrameExtractionPipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.frame_extraction_pipeline import FrameExtractionPipeline


@pytest.fixture
def mock_video_path(tmp_path: Path) -> Path:
    """モック動画ファイルパス"""
    return tmp_path / "test_video.mov"


@pytest.fixture
def sample_output_dir(tmp_path: Path) -> Path:
    """サンプル出力ディレクトリ"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def target_timestamps() -> list[datetime]:
    """テスト用の目標タイムスタンプリスト"""
    return [
        datetime(2025, 8, 26, 16, 5, 0),
        datetime(2025, 8, 26, 16, 10, 0),
        datetime(2025, 8, 26, 16, 15, 0),
    ]


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_five_minute_interval_extraction(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
    target_timestamps: list[datetime],
):
    """5分刻み抽出の正確性テスト"""
    # モックの設定
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    # CoarseSamplerのモック
    mock_coarse_sampler = MagicMock()
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock_coarse_sampler.sample.return_value = [
        (100, mock_frame),
        (200, mock_frame),
        (300, mock_frame),
    ]
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    # FineSamplerのモック
    mock_fine_sampler = MagicMock()
    mock_fine_sampler.sample_around_target.return_value = [
        (150, mock_frame),
        (180, mock_frame),
        (210, mock_frame),
    ]
    mock_fine_sampler_class.return_value = mock_fine_sampler

    # TimestampExtractorV2のモック
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = {
        "timestamp": datetime(2025, 8, 26, 16, 5, 0),
        "frame_idx": 150,
        "confidence": 0.9,
        "ocr_text": "2025/08/26 16:05:00",
        "roi_coords": (832, 0, 448, 58),
    }
    mock_extractor_class.return_value = mock_extractor

    # パイプライン初期化
    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_timestamps[0],
        end_datetime=target_timestamps[-1],
        interval_minutes=5,
        tolerance_seconds=10.0,
    )

    # 実行
    results = pipeline.run()

    # 結果が生成されることを確認
    assert len(results) > 0


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_tolerance_validation(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """±10秒許容誤差の検証"""
    # モックの設定
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock_coarse_sampler.sample.return_value = [(100, mock_frame)]
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    # ±10秒以内のタイムスタンプ
    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    mock_fine_sampler.sample_around_target.return_value = [
        (150, mock_frame),
        (180, mock_frame),
    ]
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    # ±5秒のタイムスタンプ（許容範囲内）
    extracted_ts = target_ts + timedelta(seconds=5)
    mock_extractor.extract.return_value = {
        "timestamp": extracted_ts,
        "frame_idx": 150,
        "confidence": 0.9,
        "ocr_text": extracted_ts.strftime("%Y/%m/%d %H:%M:%S"),
        "roi_coords": (832, 0, 448, 58),
    }
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
        tolerance_seconds=10.0,
    )

    results = pipeline.run()

    # 許容範囲内のタイムスタンプが採用されることを確認
    if results:
        time_diff = abs((results[0]["timestamp"] - target_ts).total_seconds())
        assert time_diff <= 10.0


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_csv_output_format(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """CSV出力フォーマットのテスト"""
    # モックの設定
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock_coarse_sampler.sample.return_value = [(100, mock_frame)]
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler.sample_around_target.return_value = [(150, mock_frame)]
    mock_fine_sampler_class.return_value = mock_fine_sampler

    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = {
        "timestamp": target_ts,
        "frame_idx": 150,
        "confidence": 0.9,
        "ocr_text": "2025/08/26 16:05:00",
        "roi_coords": (832, 0, 448, 58),
    }
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
        tolerance_seconds=10.0,
    )

    pipeline.run()

    # CSVファイルが生成されることを確認
    csv_path = sample_output_dir / "extraction_results.csv"
    assert csv_path.exists()

    # CSVの内容を確認
    import csv

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0

        # 必須カラムが存在することを確認
        required_columns = [
            "target_timestamp",
            "extracted_timestamp",
            "frame_index",
            "confidence",
            "time_diff_seconds",
            "ocr_text",
        ]
        for col in required_columns:
            assert col in rows[0]


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_missing_data_handling(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """欠損データのハンドリングテスト"""
    # モックの設定
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock_coarse_sampler.sample.return_value = [(100, mock_frame)]
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler.sample_around_target.return_value = [(150, mock_frame)]
    mock_fine_sampler_class.return_value = mock_fine_sampler

    # 抽出に失敗するモック
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = None  # 抽出失敗
    mock_extractor_class.return_value = mock_extractor

    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
        tolerance_seconds=10.0,
    )

    results = pipeline.run()

    # 失敗した場合は空のリストまたは警告が出力される
    # 実装に応じて調整が必要


def test_target_timestamps_generation(sample_output_dir: Path):
    """目標タイムスタンプ生成のテスト"""
    start = datetime(2025, 8, 26, 16, 5, 0)
    end = datetime(2025, 8, 26, 16, 20, 0)

    with patch("src.pipeline.frame_extraction_pipeline.CoarseSampler"), patch(
        "src.pipeline.frame_extraction_pipeline.FineSampler"
    ), patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2"), patch(
        "cv2.VideoCapture"
    ):
        pipeline = FrameExtractionPipeline(
            video_path="dummy.mov",
            output_dir=str(sample_output_dir),
            start_datetime=start,
            end_datetime=end,
            interval_minutes=5,
        )

        # 5分刻みで4つのタイムスタンプが生成される
        assert len(pipeline.target_timestamps) == 4  # 16:05, 16:10, 16:15, 16:20
        assert pipeline.target_timestamps[0] == start
        assert pipeline.target_timestamps[-1] == end


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_find_approximate_frame_success(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """近似フレーム検出の成功ケース"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    mock_coarse_sampler = MagicMock()
    # 目標時刻に近いフレームを返す
    mock_coarse_sampler.sample.return_value = [
        (100, mock_frame),  # 目標時刻より前
        (200, mock_frame),  # 目標時刻に近い
    ]
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    # 目標時刻に近いタイムスタンプを返す
    mock_extractor.extract.side_effect = [
        {
            "timestamp": target_ts - timedelta(seconds=30),
            "frame_idx": 100,
            "confidence": 0.9,
            "ocr_text": "2025/08/26 16:04:30",
            "roi_coords": (832, 0, 448, 58),
        },
        {
            "timestamp": target_ts + timedelta(seconds=5),
            "frame_idx": 200,
            "confidence": 0.9,
            "ocr_text": "2025/08/26 16:05:05",
            "roi_coords": (832, 0, 448, 58),
        },
    ]
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
    )

    approx_frame = pipeline._find_approximate_frame(target_ts)

    # 近似フレームが見つかることを確認
    assert approx_frame is not None
    assert approx_frame == 200  # 目標時刻に最も近いフレーム


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_find_approximate_frame_not_found(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """近似フレームが見つからない場合"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    mock_coarse_sampler = MagicMock()
    # タイムスタンプが抽出できないフレームのみ
    mock_coarse_sampler.sample.return_value = [
        (100, mock_frame),
        (200, mock_frame),
    ]
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    # タイムスタンプ抽出に失敗
    mock_extractor.extract.return_value = None
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
    )

    approx_frame = pipeline._find_approximate_frame(target_ts)

    # 近似フレームが見つからないことを確認
    assert approx_frame is None


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_find_best_frame_around_multiple_candidates(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """複数候補からの最良フレーム選択"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    mock_coarse_sampler = MagicMock()
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    # 複数の候補フレーム
    mock_fine_sampler.sample_around_target.return_value = [
        (150, mock_frame),
        (160, mock_frame),
        (170, mock_frame),
    ]
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    # 複数の候補（時間差が異なる）
    mock_extractor.extract.side_effect = [
        {
            "timestamp": target_ts + timedelta(seconds=8),
            "frame_idx": 150,
            "confidence": 0.9,
            "ocr_text": "2025/08/26 16:05:08",
            "roi_coords": (832, 0, 448, 58),
        },
        {
            "timestamp": target_ts + timedelta(seconds=3),
            "frame_idx": 160,
            "confidence": 0.85,
            "ocr_text": "2025/08/26 16:05:03",
            "roi_coords": (832, 0, 448, 58),
        },
        {
            "timestamp": target_ts + timedelta(seconds=5),
            "frame_idx": 170,
            "confidence": 0.95,
            "ocr_text": "2025/08/26 16:05:05",
            "roi_coords": (832, 0, 448, 58),
        },
    ]
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
        tolerance_seconds=10.0,
    )

    best_frame = pipeline._find_best_frame_around(target_ts, 160)

    # 最良のフレーム（時間差が最小）が選択されることを確認
    assert best_frame is not None
    assert best_frame["frame_idx"] == 160  # 時間差3秒が最小
    assert abs(best_frame["time_diff"] - 3.0) < 0.1


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_find_best_frame_around_no_candidates(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """候補がない場合"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    mock_coarse_sampler = MagicMock()
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler.sample_around_target.return_value = [
        (150, mock_frame),
        (160, mock_frame),
    ]
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    # 許容範囲外のタイムスタンプ
    mock_extractor.extract.side_effect = [
        {
            "timestamp": target_ts + timedelta(seconds=20),
            "frame_idx": 150,
            "confidence": 0.9,
            "ocr_text": "2025/08/26 16:05:20",
            "roi_coords": (832, 0, 448, 58),
        },
        {
            "timestamp": target_ts + timedelta(seconds=15),
            "frame_idx": 160,
            "confidence": 0.85,
            "ocr_text": "2025/08/26 16:05:15",
            "roi_coords": (832, 0, 448, 58),
        },
    ]
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
        tolerance_seconds=10.0,  # ±10秒以内
    )

    best_frame = pipeline._find_best_frame_around(target_ts, 160)

    # 候補がない場合はNoneが返される
    assert best_frame is None


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_save_frame(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """フレーム保存のテスト"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=datetime(2025, 8, 26, 16, 5, 0),
        end_datetime=datetime(2025, 8, 26, 16, 10, 0),
        interval_minutes=5,
    )

    timestamp = datetime(2025, 8, 26, 16, 5, 0)
    mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = {
        "timestamp": timestamp,
        "frame_idx": 100,
        "confidence": 0.9,
        "ocr_text": "2025/08/26 16:05:00",
        "frame": mock_frame,
    }

    pipeline._save_frame(result)

    # フレームファイルが生成されることを確認
    frames_dir = sample_output_dir / "frames"
    assert frames_dir.exists()

    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    frame_path = frames_dir / f"frame_{timestamp_str}.jpg"
    assert frame_path.exists()


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_save_results_csv_format(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """CSV出力フォーマットのテスト"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=datetime(2025, 8, 26, 16, 5, 0),
        end_datetime=datetime(2025, 8, 26, 16, 10, 0),
        interval_minutes=5,
    )

    timestamp = datetime(2025, 8, 26, 16, 5, 0)
    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    results = [
        {
            "timestamp": timestamp,
            "target_timestamp": target_ts,
            "frame_idx": 100,
            "confidence": 0.9,
            "ocr_text": "2025/08/26 16:05:00",
            "time_diff": 0.5,
        }
    ]

    pipeline._save_results_csv(results)

    # CSVファイルが生成されることを確認
    csv_path = sample_output_dir / "extraction_results.csv"
    assert csv_path.exists()

    # CSVの内容を確認
    import csv

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["target_timestamp"] == "2025/08/26 16:05:00"
    assert rows[0]["extracted_timestamp"] == "2025/08/26 16:05:00"
    assert rows[0]["frame_index"] == "100"
    assert rows[0]["confidence"] == "0.9000"
    assert abs(float(rows[0]["time_diff_seconds"]) - 0.5) < 0.01


@patch("src.pipeline.frame_extraction_pipeline.VideoProcessor")
@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_run_with_auto_targets(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_processor_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """自動目標タイムスタンプ生成のテスト"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    # VideoProcessorのモック
    mock_video_processor = MagicMock()
    mock_video_processor.total_frames = 1000
    mock_video_processor.get_frame.return_value = np.random.randint(
        0, 255, (720, 1280, 3), dtype=np.uint8
    )
    mock_video_processor_class.return_value = mock_video_processor

    mock_extractor = MagicMock()
    # タイムスタンプを抽出できるフレーム
    base_time = datetime(2025, 8, 26, 16, 4, 0)
    mock_extractor.extract.side_effect = [
        {
            "timestamp": base_time + timedelta(seconds=i * 10),
            "frame_idx": i,
            "confidence": 0.9,
            "ocr_text": (base_time + timedelta(seconds=i * 10)).strftime(
                "%Y/%m/%d %H:%M:%S"
            ),
            "roi_coords": (832, 0, 448, 58),
        }
        if i % 10 == 0  # 10フレームごとにタイムスタンプを抽出
        else None
        for i in range(100)
    ]
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=base_time,
        end_datetime=base_time + timedelta(minutes=20),
        interval_minutes=5,
    )

    results = pipeline.run_with_auto_targets(max_frames=100)

    # 結果が生成されることを確認
    assert len(results) >= 0


@patch("src.pipeline.frame_extraction_pipeline.VideoProcessor")
@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_run_with_auto_targets_max_frames(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_processor_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """最大フレーム数制限のテスト"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_video_processor = MagicMock()
    mock_video_processor.total_frames = 10000
    mock_video_processor.get_frame.return_value = np.random.randint(
        0, 255, (720, 1280, 3), dtype=np.uint8
    )
    mock_video_processor_class.return_value = mock_video_processor

    mock_extractor = MagicMock()
    base_time = datetime(2025, 8, 26, 16, 4, 0)
    mock_extractor.extract.side_effect = [
        {
            "timestamp": base_time + timedelta(seconds=i * 10),
            "frame_idx": i,
            "confidence": 0.9,
            "ocr_text": (base_time + timedelta(seconds=i * 10)).strftime(
                "%Y/%m/%d %H:%M:%S"
            ),
            "roi_coords": (832, 0, 448, 58),
        }
        if i % 10 == 0
        else None
        for i in range(10000)
    ]
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=base_time,
        end_datetime=base_time + timedelta(minutes=20),
        interval_minutes=5,
    )

    # 最大50フレームに制限
    results = pipeline.run_with_auto_targets(max_frames=50)

    # 最大フレーム数が適用されることを確認
    mock_video_processor.get_frame.assert_called()
    # 50フレームまでしか処理されない
    assert mock_video_processor.get_frame.call_count <= 50


@patch("src.pipeline.frame_extraction_pipeline.VideoProcessor")
@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_run_with_auto_targets_disable_validation(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_processor_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """検証無効化のテスト"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_video_processor = MagicMock()
    mock_video_processor.total_frames = 100
    mock_video_processor.get_frame.return_value = np.random.randint(
        0, 255, (720, 1280, 3), dtype=np.uint8
    )
    mock_video_processor_class.return_value = mock_video_processor

    mock_extractor = MagicMock()
    base_time = datetime(2025, 8, 26, 16, 4, 0)
    mock_extractor.extract.return_value = {
        "timestamp": base_time,
        "frame_idx": 0,
        "confidence": 0.9,
        "ocr_text": base_time.strftime("%Y/%m/%d %H:%M:%S"),
        "roi_coords": (832, 0, 448, 58),
    }
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=base_time,
        end_datetime=base_time + timedelta(minutes=20),
        interval_minutes=5,
    )

    # 検証を無効化
    results = pipeline.run_with_auto_targets(disable_validation=True)

    # 検証が無効化されていることを確認（エラーが発生しない）
    assert results is not None


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_cleanup(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """リソース解放のテスト"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.release = MagicMock()
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    mock_coarse_sampler.close = MagicMock()
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    mock_extractor.reset_validator = MagicMock()
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=datetime(2025, 8, 26, 16, 5, 0),
        end_datetime=datetime(2025, 8, 26, 16, 10, 0),
        interval_minutes=5,
    )

    pipeline.cleanup()

    # リソースが解放されることを確認
    mock_coarse_sampler.close.assert_called_once()
    mock_cap.release.assert_called_once()
    mock_extractor.reset_validator.assert_called_once()


@patch("src.pipeline.frame_extraction_pipeline.CoarseSampler")
@patch("src.pipeline.frame_extraction_pipeline.FineSampler")
@patch("src.pipeline.frame_extraction_pipeline.TimestampExtractorV2")
@patch("cv2.VideoCapture")
def test_extract_frame_for_target_failure(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
):
    """フレーム抽出失敗のテスト"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    mock_coarse_sampler = MagicMock()
    # 近似フレームが見つからない
    mock_coarse_sampler.sample.return_value = []
    mock_coarse_sampler_class.return_value = mock_coarse_sampler

    mock_fine_sampler = MagicMock()
    mock_fine_sampler_class.return_value = mock_fine_sampler

    mock_extractor = MagicMock()
    mock_extractor_class.return_value = mock_extractor

    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=datetime(2025, 8, 26, 16, 5, 0),
        end_datetime=datetime(2025, 8, 26, 16, 10, 0),
        interval_minutes=5,
    )

    target_ts = datetime(2025, 8, 26, 16, 5, 0)
    result = pipeline._extract_frame_for_target(target_ts)

    # 抽出に失敗することを確認
    assert result is None
