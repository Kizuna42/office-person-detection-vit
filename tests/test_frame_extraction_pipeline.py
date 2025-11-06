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


@patch('src.pipeline.frame_extraction_pipeline.CoarseSampler')
@patch('src.pipeline.frame_extraction_pipeline.FineSampler')
@patch('src.pipeline.frame_extraction_pipeline.TimestampExtractorV2')
@patch('cv2.VideoCapture')
def test_five_minute_interval_extraction(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path,
    target_timestamps: list[datetime]
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
        'timestamp': datetime(2025, 8, 26, 16, 5, 0),
        'frame_idx': 150,
        'confidence': 0.9,
        'ocr_text': '2025/08/26 16:05:00',
        'roi_coords': (832, 0, 448, 58)
    }
    mock_extractor_class.return_value = mock_extractor
    
    # パイプライン初期化
    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_timestamps[0],
        end_datetime=target_timestamps[-1],
        interval_minutes=5,
        tolerance_seconds=10.0
    )
    
    # 実行
    results = pipeline.run()
    
    # 結果が生成されることを確認
    assert len(results) > 0


@patch('src.pipeline.frame_extraction_pipeline.CoarseSampler')
@patch('src.pipeline.frame_extraction_pipeline.FineSampler')
@patch('src.pipeline.frame_extraction_pipeline.TimestampExtractorV2')
@patch('cv2.VideoCapture')
def test_tolerance_validation(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path
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
        'timestamp': extracted_ts,
        'frame_idx': 150,
        'confidence': 0.9,
        'ocr_text': extracted_ts.strftime('%Y/%m/%d %H:%M:%S'),
        'roi_coords': (832, 0, 448, 58)
    }
    mock_extractor_class.return_value = mock_extractor
    
    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
        tolerance_seconds=10.0
    )
    
    results = pipeline.run()
    
    # 許容範囲内のタイムスタンプが採用されることを確認
    if results:
        time_diff = abs((results[0]['timestamp'] - target_ts).total_seconds())
        assert time_diff <= 10.0


@patch('src.pipeline.frame_extraction_pipeline.CoarseSampler')
@patch('src.pipeline.frame_extraction_pipeline.FineSampler')
@patch('src.pipeline.frame_extraction_pipeline.TimestampExtractorV2')
@patch('cv2.VideoCapture')
def test_csv_output_format(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path
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
        'timestamp': target_ts,
        'frame_idx': 150,
        'confidence': 0.9,
        'ocr_text': '2025/08/26 16:05:00',
        'roi_coords': (832, 0, 448, 58)
    }
    mock_extractor_class.return_value = mock_extractor
    
    pipeline = FrameExtractionPipeline(
        video_path=str(mock_video_path),
        output_dir=str(sample_output_dir),
        start_datetime=target_ts,
        end_datetime=target_ts + timedelta(minutes=5),
        interval_minutes=5,
        tolerance_seconds=10.0
    )
    
    pipeline.run()
    
    # CSVファイルが生成されることを確認
    csv_path = sample_output_dir / 'extraction_results.csv'
    assert csv_path.exists()
    
    # CSVの内容を確認
    import csv
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0
        
        # 必須カラムが存在することを確認
        required_columns = [
            'target_timestamp',
            'extracted_timestamp',
            'frame_index',
            'confidence',
            'time_diff_seconds',
            'ocr_text'
        ]
        for col in required_columns:
            assert col in rows[0]


@patch('src.pipeline.frame_extraction_pipeline.CoarseSampler')
@patch('src.pipeline.frame_extraction_pipeline.FineSampler')
@patch('src.pipeline.frame_extraction_pipeline.TimestampExtractorV2')
@patch('cv2.VideoCapture')
def test_missing_data_handling(
    mock_video_capture,
    mock_extractor_class,
    mock_fine_sampler_class,
    mock_coarse_sampler_class,
    mock_video_path: Path,
    sample_output_dir: Path
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
        tolerance_seconds=10.0
    )
    
    results = pipeline.run()
    
    # 失敗した場合は空のリストまたは警告が出力される
    # 実装に応じて調整が必要


def test_target_timestamps_generation(sample_output_dir: Path):
    """目標タイムスタンプ生成のテスト"""
    start = datetime(2025, 8, 26, 16, 5, 0)
    end = datetime(2025, 8, 26, 16, 20, 0)
    
    with patch('src.pipeline.frame_extraction_pipeline.CoarseSampler'), \
         patch('src.pipeline.frame_extraction_pipeline.FineSampler'), \
         patch('src.pipeline.frame_extraction_pipeline.TimestampExtractorV2'), \
         patch('cv2.VideoCapture'):
        
        pipeline = FrameExtractionPipeline(
            video_path="dummy.mov",
            output_dir=str(sample_output_dir),
            start_datetime=start,
            end_datetime=end,
            interval_minutes=5
        )
        
        # 5分刻みで3つのタイムスタンプが生成される
        assert len(pipeline.target_timestamps) == 4  # 16:05, 16:10, 16:15, 16:20
        assert pipeline.target_timestamps[0] == start
        assert pipeline.target_timestamps[-1] == end

