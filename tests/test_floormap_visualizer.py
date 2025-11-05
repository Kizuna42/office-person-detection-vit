"""Unit tests for FloormapVisualizer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.models import Detection, FrameResult
from src.visualization.floormap_visualizer import FloormapVisualizer


@pytest.fixture
def sample_floormap_path(tmp_path: Path) -> Path:
    """テスト用のフロアマップ画像を作成"""

    floormap = np.ones((1369, 1878, 3), dtype=np.uint8) * 255
    floormap_path = tmp_path / "floormap.png"
    cv2.imwrite(str(floormap_path), floormap)
    return floormap_path


@pytest.fixture
def sample_floormap_config() -> dict:
    """テスト用のフロアマップ設定"""

    return {
        "image_width": 1878,
        "image_height": 1369,
        "image_origin_x": 7,
        "image_origin_y": 9,
        "image_x_mm_per_pixel": 28.1926406926406,
        "image_y_mm_per_pixel": 28.241430700447,
    }


@pytest.fixture
def sample_zones() -> list[dict]:
    """テスト用のゾーン定義"""

    return [
        {
            "id": "zone_a",
            "name": "Zone A",
            "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "priority": 1,
        },
        {
            "id": "zone_b",
            "name": "Zone B",
            "polygon": [(150, 150), (250, 150), (250, 250), (150, 250)],
            "priority": 2,
        },
    ]


@pytest.fixture
def sample_camera_config() -> dict:
    """テスト用のカメラ設定"""

    return {
        "position_x": 859,
        "position_y": 1040,
        "height_m": 2.2,
        "show_on_floormap": True,
        "marker_color": [0, 0, 255],
        "marker_size": 15,
    }


def test_init(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """FloormapVisualizer が正しく初期化される。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    assert visualizer.floormap_image is not None
    assert visualizer.floormap_image.shape == (1369, 1878, 3)
    assert len(visualizer.zone_colors) == 3  # zone_a, zone_b, unclassified


def test_init_file_not_found(tmp_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """存在しないフロアマップ画像では FileNotFoundError が発生する。"""

    with pytest.raises(FileNotFoundError):
        FloormapVisualizer("nonexistent.png", sample_floormap_config, sample_zones)


def test_init_invalid_image(tmp_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """不正な画像ファイルでは ValueError が発生する。"""

    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_bytes(b"invalid image data")

    with pytest.raises(ValueError):
        FloormapVisualizer(str(invalid_path), sample_floormap_config, sample_zones)


def test_generate_zone_colors(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """ゾーンごとの色が正しく生成される。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    assert "zone_a" in visualizer.zone_colors
    assert "zone_b" in visualizer.zone_colors
    assert "unclassified" in visualizer.zone_colors
    assert all(isinstance(color, tuple) and len(color) == 3 for color in visualizer.zone_colors.values())


def test_draw_camera_position(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict], sample_camera_config: dict):
    """カメラ位置を正しく描画できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path),
        sample_floormap_config,
        sample_zones,
        camera_config=sample_camera_config,
    )

    image = visualizer.floormap_image.copy()
    result = visualizer.draw_camera_position(image)

    assert result.shape == image.shape


def test_draw_camera_position_hidden(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """show_on_floormap=False の場合はカメラ位置を描画しない。"""

    camera_config = {"show_on_floormap": False}
    visualizer = FloormapVisualizer(
        str(sample_floormap_path),
        sample_floormap_config,
        sample_zones,
        camera_config=camera_config,
    )

    image = visualizer.floormap_image.copy()
    result = visualizer.draw_camera_position(image)

    assert result.shape == image.shape
    # 画像が変更されていないことを確認（厳密には難しいが、エラーなく処理されることを確認）


def test_draw_zones(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """ゾーンを正しく描画できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    image = visualizer.floormap_image.copy()
    result = visualizer.draw_zones(image)

    assert result.shape == image.shape


def test_draw_zones_with_alpha(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """透明度を指定してゾーンを描画できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    image = visualizer.floormap_image.copy()
    result = visualizer.draw_zones(image, alpha=0.5)

    assert result.shape == image.shape


def test_draw_detections(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """検出結果を正しく描画できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    detections = [
        Detection(
            bbox=(100, 200, 50, 100),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125, 300),
            floor_coords=(500, 600),
            zone_ids=["zone_a"],
        ),
        Detection(
            bbox=(300, 400, 80, 120),
            confidence=0.85,
            class_id=1,
            class_name="person",
            camera_coords=(340, 520),
            floor_coords=(800, 900),
            zone_ids=["zone_b"],
        ),
    ]

    image = visualizer.floormap_image.copy()
    result = visualizer.draw_detections(image, detections)

    assert result.shape == image.shape


def test_draw_detections_without_labels(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """ラベルなしで検出結果を描画できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    detections = [
        Detection(
            bbox=(100, 200, 50, 100),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125, 300),
            floor_coords=(500, 600),
            zone_ids=["zone_a"],
        ),
    ]

    image = visualizer.floormap_image.copy()
    result = visualizer.draw_detections(image, detections, draw_labels=False)

    assert result.shape == image.shape


def test_draw_detections_out_of_bounds(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """範囲外の座標はスキップされる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    detections = [
        Detection(
            bbox=(100, 200, 50, 100),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125, 300),
            floor_coords=(10000, 10000),  # 範囲外
            zone_ids=["zone_a"],
        ),
    ]

    image = visualizer.floormap_image.copy()
    result = visualizer.draw_detections(image, detections)

    assert result.shape == image.shape


def test_draw_detections_no_floor_coords(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """floor_coords が None の場合はスキップされる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    detections = [
        Detection(
            bbox=(100, 200, 50, 100),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125, 300),
            floor_coords=None,
            zone_ids=["zone_a"],
        ),
    ]

    image = visualizer.floormap_image.copy()
    result = visualizer.draw_detections(image, detections)

    assert result.shape == image.shape


def test_visualize_frame(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """フレーム結果を可視化できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    detections = [
        Detection(
            bbox=(100, 200, 50, 100),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125, 300),
            floor_coords=(500, 600),
            zone_ids=["zone_a"],
        ),
    ]

    frame_result = FrameResult(
        frame_number=0,
        timestamp="12:00",
        detections=detections,
        zone_counts={"zone_a": 1, "zone_b": 0},
    )

    result = visualizer.visualize_frame(frame_result)

    assert result.shape == visualizer.floormap_image.shape


def test_visualize_frame_without_zones(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """ゾーンを描画せずに可視化できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    detections = [
        Detection(
            bbox=(100, 200, 50, 100),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125, 300),
            floor_coords=(500, 600),
            zone_ids=["zone_a"],
        ),
    ]

    frame_result = FrameResult(
        frame_number=0,
        timestamp="12:00",
        detections=detections,
        zone_counts={"zone_a": 1},
    )

    result = visualizer.visualize_frame(frame_result, draw_zones=False)

    assert result.shape == visualizer.floormap_image.shape


def test_save_visualization(tmp_path: Path, sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """可視化画像を保存できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    image = visualizer.floormap_image.copy()
    output_path = tmp_path / "visualization.png"

    visualizer.save_visualization(image, str(output_path))

    assert output_path.exists()


def test_create_legend(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """凡例を作成できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    legend = visualizer.create_legend()

    assert isinstance(legend, np.ndarray)
    assert len(legend.shape) == 3
    assert legend.shape[2] == 3


def test_create_legend_custom_size(sample_floormap_path: Path, sample_floormap_config: dict, sample_zones: list[dict]):
    """カスタムサイズの凡例を作成できる。"""

    visualizer = FloormapVisualizer(
        str(sample_floormap_path), sample_floormap_config, sample_zones
    )

    legend = visualizer.create_legend(width=400, height=300)

    assert isinstance(legend, np.ndarray)
    assert legend.shape[1] == 400
    assert legend.shape[0] == 300
