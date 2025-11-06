"""Unit tests for Visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.aggregation import Aggregator
from src.models import Detection
from src.visualization import Visualizer


@pytest.fixture
def sample_detections() -> list[Detection]:
    """テスト用の検出結果リスト"""

    return [
        Detection(
            bbox=(100.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 300.0),
            floor_coords=(150.0, 350.0),
            zone_ids=["zone_a"],
        ),
        Detection(
            bbox=(300.0, 400.0, 80.0, 120.0),
            confidence=0.85,
            class_id=1,
            class_name="person",
            camera_coords=(340.0, 520.0),
            floor_coords=(400.0, 500.0),
            zone_ids=["zone_b"],
        ),
    ]


def test_init():
    """Visualizer が正しく初期化される。"""

    visualizer = Visualizer(debug_mode=False)
    assert visualizer.debug_mode is False

    visualizer_debug = Visualizer(debug_mode=True)
    assert visualizer_debug.debug_mode is True


def test_draw_detections(sample_frame: np.ndarray, sample_detections: list[Detection]):
    """検出結果を正しく描画できる。"""

    visualizer = Visualizer()
    result = visualizer.draw_detections(sample_frame, sample_detections)

    assert result.shape == sample_frame.shape
    assert result.dtype == sample_frame.dtype


def test_draw_detections_with_confidence(
    sample_frame: np.ndarray, sample_detections: list[Detection]
):
    """信頼度情報を表示して描画できる。"""

    visualizer = Visualizer()
    result = visualizer.draw_detections(
        sample_frame, sample_detections, show_confidence=True
    )

    assert result.shape == sample_frame.shape


def test_draw_detections_with_coords(
    sample_frame: np.ndarray, sample_detections: list[Detection]
):
    """座標情報を表示して描画できる。"""

    visualizer = Visualizer(debug_mode=True)
    result = visualizer.draw_detections(
        sample_frame, sample_detections, show_coords=True
    )

    assert result.shape == sample_frame.shape


def test_draw_detections_empty(sample_frame: np.ndarray):
    """検出結果が空の場合でもエラーなく処理できる。"""

    visualizer = Visualizer()
    result = visualizer.draw_detections(sample_frame, [])

    assert result.shape == sample_frame.shape


def test_draw_attention_map_1d(sample_frame: np.ndarray):
    """1次元のAttention Mapを描画できる。"""

    visualizer = Visualizer()
    attention_map = np.random.rand(256)  # 16x16のパッチ

    result = visualizer.draw_attention_map(sample_frame, attention_map)

    assert result.shape == sample_frame.shape


def test_draw_attention_map_2d(sample_frame: np.ndarray):
    """2次元のAttention Mapを描画できる。"""

    visualizer = Visualizer()
    attention_map = np.random.rand(32, 32)

    result = visualizer.draw_attention_map(sample_frame, attention_map)

    assert result.shape == sample_frame.shape


def test_draw_attention_map_with_alpha(sample_frame: np.ndarray):
    """透明度を指定してAttention Mapを描画できる。"""

    visualizer = Visualizer()
    attention_map = np.random.rand(32, 32)

    result = visualizer.draw_attention_map(sample_frame, attention_map, alpha=0.5)

    assert result.shape == sample_frame.shape


def test_visualize_with_attention(
    sample_frame: np.ndarray, sample_detections: list[Detection]
):
    """検出結果とAttention Mapを同時に可視化できる。"""

    visualizer = Visualizer()
    attention_map = np.random.rand(32, 32)

    result = visualizer.visualize_with_attention(
        sample_frame, sample_detections, attention_map
    )

    assert result.shape == sample_frame.shape


def test_visualize_without_attention(
    sample_frame: np.ndarray, sample_detections: list[Detection]
):
    """Attention Mapなしで可視化できる。"""

    visualizer = Visualizer()

    result = visualizer.visualize_with_attention(
        sample_frame, sample_detections, attention_map=None
    )

    assert result.shape == sample_frame.shape


def test_save_image(tmp_path: Path, sample_frame: np.ndarray):
    """画像を保存できる。"""

    visualizer = Visualizer()
    output_path = tmp_path / "test_image.jpg"

    result = visualizer.save_image(sample_frame, str(output_path))

    assert result is True
    assert output_path.exists()


def test_save_image_creates_directory(tmp_path: Path, sample_frame: np.ndarray):
    """保存先ディレクトリが存在しない場合は作成される。"""

    visualizer = Visualizer()
    output_path = tmp_path / "subdir" / "test_image.jpg"

    result = visualizer.save_image(sample_frame, str(output_path))

    assert result is True
    assert output_path.exists()


def test_create_comparison_view(
    sample_frame: np.ndarray, sample_detections: list[Detection]
):
    """比較ビューを作成できる。"""

    visualizer = Visualizer()
    with_detections = visualizer.draw_detections(sample_frame, sample_detections)

    result = visualizer.create_comparison_view(sample_frame, with_detections)

    assert result.shape[1] == sample_frame.shape[1] * 2  # 2つの画像を横に連結


def test_create_comparison_view_with_attention(
    sample_frame: np.ndarray, sample_detections: list[Detection]
):
    """Attention Map付きの比較ビューを作成できる。"""

    visualizer = Visualizer()
    with_detections = visualizer.draw_detections(sample_frame, sample_detections)
    attention_map = np.random.rand(32, 32)
    with_attention = visualizer.draw_attention_map(sample_frame, attention_map)

    result = visualizer.create_comparison_view(
        sample_frame, with_detections, with_attention
    )

    assert result.shape[1] == sample_frame.shape[1] * 3  # 3つの画像を横に連結


def test_plot_time_series(tmp_path: Path):
    """時系列グラフを生成できる。"""

    visualizer = Visualizer()
    aggregator = Aggregator()

    # データを追加
    from src.models import Detection as Det

    detections_1 = [
        Det(
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5, 10),
            zone_ids=["zone_a"],
        )
    ]
    detections_2 = [
        Det(
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5, 10),
            zone_ids=["zone_a"],
        ),
        Det(
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5, 10),
            zone_ids=["zone_a"],
        ),
    ]

    aggregator.aggregate_frame("12:00", detections_1)
    aggregator.aggregate_frame("12:05", detections_2)

    output_path = tmp_path / "time_series.png"
    result = visualizer.plot_time_series(aggregator, str(output_path))

    assert result is True
    assert output_path.exists()


def test_plot_time_series_empty(tmp_path: Path):
    """データがない場合は False が返される。"""

    visualizer = Visualizer()
    aggregator = Aggregator()

    output_path = tmp_path / "time_series.png"
    result = visualizer.plot_time_series(aggregator, str(output_path))

    assert result is False


def test_plot_zone_statistics(tmp_path: Path):
    """ゾーン統計グラフを生成できる。"""

    visualizer = Visualizer()
    aggregator = Aggregator()

    from src.models import Detection as Det

    detections = [
        Det(
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5, 10),
            zone_ids=["zone_a"],
        )
    ]

    aggregator.aggregate_frame("12:00", detections)
    aggregator.aggregate_frame("12:05", detections)

    output_path = tmp_path / "zone_statistics.png"
    result = visualizer.plot_zone_statistics(aggregator, str(output_path))

    assert result is True
    assert output_path.exists()


def test_plot_zone_statistics_empty(tmp_path: Path):
    """データがない場合は False が返される。"""

    visualizer = Visualizer()
    aggregator = Aggregator()

    output_path = tmp_path / "zone_statistics.png"
    result = visualizer.plot_zone_statistics(aggregator, str(output_path))

    assert result is False


def test_plot_heatmap(tmp_path: Path):
    """ヒートマップを生成できる。"""

    visualizer = Visualizer()
    aggregator = Aggregator()

    from src.models import Detection as Det

    detections = [
        Det(
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(5, 10),
            zone_ids=["zone_a"],
        )
    ]

    aggregator.aggregate_frame("12:00", detections)
    aggregator.aggregate_frame("12:05", detections)

    output_path = tmp_path / "heatmap.png"
    result = visualizer.plot_heatmap(aggregator, str(output_path))

    assert result is True
    assert output_path.exists()


def test_plot_heatmap_empty(tmp_path: Path):
    """データがない場合は False が返される。"""

    visualizer = Visualizer()
    aggregator = Aggregator()

    output_path = tmp_path / "heatmap.png"
    result = visualizer.plot_heatmap(aggregator, str(output_path))

    assert result is False
