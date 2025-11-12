"""Unit tests for ConfigManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config import ConfigManager

if TYPE_CHECKING:
    from pathlib import Path


def test_init_with_default_config():
    """存在しない設定ファイルパスで初期化するとデフォルト設定が使用される。"""

    config = ConfigManager("nonexistent_config.yaml")
    assert config.config is not None
    assert "video" in config.config
    assert "detection" in config.config


def test_load_yaml_config(tmp_path: Path):
    """YAML設定ファイルを正しく読み込める。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    assert config.get("video.input_path") == "test_video.mov"
    assert config.get("detection.model_name") == "test_model"
    assert config.get("detection.confidence_threshold") == 0.7


def test_load_json_config(tmp_path: Path):
    """JSON設定ファイルを正しく読み込める。"""

    import json

    json_content = {
        "video": {"input_path": "test_video.mov"},
        "detection": {
            "model_name": "test_model",
            "confidence_threshold": 0.7,
            "device": "cpu",
        },
        "floormap": {
            "image_path": "test_floormap.png",
            "image_width": 100,
            "image_height": 50,
            "image_origin_x": 0,
            "image_origin_y": 0,
            "image_x_mm_per_pixel": 1.0,
            "image_y_mm_per_pixel": 1.0,
        },
        "homography": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "zones": [],
        "output": {"directory": "output"},
    }
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(json_content), encoding="utf-8")

    config = ConfigManager(str(config_path))
    assert config.get("video.input_path") == "test_video.mov"
    assert config.get("detection.model_name") == "test_model"


def test_load_empty_config_file(tmp_path: Path):
    """空の設定ファイルではデフォルト設定が使用される。"""

    config_path = tmp_path / "empty_config.yaml"
    config_path.write_text("", encoding="utf-8")

    config = ConfigManager(str(config_path))
    assert config.config is not None
    assert "video" in config.config


def test_invalid_yaml_format(tmp_path: Path):
    """不正なYAML形式では ValueError が発生する。"""

    config_path = tmp_path / "invalid_config.yaml"
    config_path.write_text("invalid: yaml: content: [", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML解析エラー"):
        ConfigManager(str(config_path))


def test_validate_success(tmp_path: Path):
    """正常な設定では validate が True を返す。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    assert config.validate() is True


def test_validate_missing_section(tmp_path: Path):
    """必須セクションがない場合は ValueError が発生する。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="必須セクション"):
        config.validate()


def test_validate_missing_key(tmp_path: Path):
    """必須キーがない場合は ValueError が発生する。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match=r"必須項目.*confidence_threshold"):
        config.validate()


def test_validate_invalid_confidence_threshold(tmp_path: Path):
    """confidence_threshold が範囲外の場合は ValueError が発生する。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 1.5
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="confidence_threshold"):
        config.validate()


def test_validate_invalid_device(tmp_path: Path):
    """device が不正な値の場合は ValueError が発生する。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "invalid_device"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="device"):
        config.validate()


def test_validate_invalid_homography_matrix(tmp_path: Path):
    """ホモグラフィ行列が不正な場合は ValueError が発生する。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0], [0, 1]]  # 3x3でない
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match=r"homography\.matrix"):
        config.validate()


def test_validate_zones(tmp_path: Path):
    """zones セクションの検証が正しく動作する。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones:
  - id: "zone_a"
    polygon: [[0, 0], [10, 0], [10, 10], [0, 10]]
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    assert config.validate() is True


def test_validate_zones_invalid_polygon(tmp_path: Path):
    """zones の polygon が不正な場合は ValueError が発生する。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones:
  - id: "zone_a"
    polygon: [[0, 0], [10, 0]]  # 3点未満
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="polygon"):
        config.validate()


def test_get_dot_notation():
    """ドット記法で設定値を取得できる。"""

    config = ConfigManager("nonexistent.yaml")
    assert config.get("video.input_path") is not None
    assert config.get("detection.model_name") is not None


def test_get_with_default():
    """存在しないキーではデフォルト値が返される。"""

    config = ConfigManager("nonexistent.yaml")
    assert config.get("nonexistent.key", "default_value") == "default_value"


def test_get_section():
    """セクション全体を取得できる。"""

    config = ConfigManager("nonexistent.yaml")
    video_section = config.get_section("video")
    assert isinstance(video_section, dict)
    assert "input_path" in video_section


def test_set_dot_notation():
    """ドット記法で設定値を変更できる。"""

    config = ConfigManager("nonexistent.yaml")
    config.set("video.input_path", "new_path.mov")
    assert config.get("video.input_path") == "new_path.mov"


def test_set_nested():
    """ネストされた設定値を変更できる。"""

    config = ConfigManager("nonexistent.yaml")
    config.set("detection.confidence_threshold", 0.9)
    assert config.get("detection.confidence_threshold") == 0.9


def test_save_yaml(tmp_path: Path):
    """設定をYAMLファイルに保存できる。"""

    config = ConfigManager("nonexistent.yaml")
    config.set("video.input_path", "saved_path.mov")

    output_path = tmp_path / "saved_config.yaml"
    config.save(str(output_path))

    assert output_path.exists()
    saved_config = ConfigManager(str(output_path))
    assert saved_config.get("video.input_path") == "saved_path.mov"


def test_save_json(tmp_path: Path):
    """設定をJSONファイルに保存できる。"""

    config = ConfigManager("nonexistent.yaml")
    config.set("video.input_path", "saved_path.mov")

    output_path = tmp_path / "saved_config.json"
    config.save(str(output_path))

    assert output_path.exists()
    saved_config = ConfigManager(str(output_path))
    assert saved_config.get("video.input_path") == "saved_path.mov"


def test_save_invalid_format(tmp_path: Path):
    """サポートされていない形式では ValueError が発生する。"""

    config = ConfigManager("nonexistent.yaml")
    output_path = tmp_path / "saved_config.txt"

    with pytest.raises(ValueError, match="サポートされていないファイル形式"):
        config.save(str(output_path))


def test_validate_camera_config(tmp_path: Path):
    """camera セクションの検証が正しく動作する。"""

    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
camera:
  position_x: 100
  position_y: 200
  height_m: 2.5
  show_on_floormap: true
  marker_color: [0, 0, 255]
  marker_size: 15
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    assert config.validate() is True


def test_validate_video_config_edge_cases(tmp_path: Path):
    """video設定のエッジケース"""
    # 無効なinput_path（None）
    yaml_content = """
video:
  input_path: null
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match=r"video\.input_path は文字列"):
        config.validate()

    # 無効なframe_interval_minutes（負の値）
    yaml_content = """
video:
  input_path: "test_video.mov"
  frame_interval_minutes: -5
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path.write_text(yaml_content, encoding="utf-8")
    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="frame_interval_minutes"):
        config.validate()


def test_validate_detection_config_edge_cases(tmp_path: Path):
    """detection設定のエッジケース"""
    # 無効なconfidence_threshold（範囲外）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 1.5
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="confidence_threshold"):
        config.validate()

    # 無効なdevice
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "invalid_device"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path.write_text(yaml_content, encoding="utf-8")
    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="device"):
        config.validate()


def test_validate_floormap_config_edge_cases(tmp_path: Path):
    """floormap設定のエッジケース"""
    # 無効なimage_width（負の値）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: -100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="image_width"):
        config.validate()

    # 無効なimage_x_mm_per_pixel（ゼロ）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
"""
    config_path.write_text(yaml_content, encoding="utf-8")
    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="image_x_mm_per_pixel"):
        config.validate()


def test_validate_homography_config_edge_cases(tmp_path: Path):
    """homography設定のエッジケース"""
    # 無効なmatrix（行数が不足）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0]]
zones: []
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="3x3 行列"):
        config.validate()

    # 無効なmatrix（行の長さが不足）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0], [0, 1], [0, 0]]
zones: []
output:
  directory: "output"
"""
    config_path.write_text(yaml_content, encoding="utf-8")
    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="長さ3のリスト"):
        config.validate()


def test_validate_zones_config_edge_cases(tmp_path: Path):
    """zones設定のエッジケース"""
    # 無効なpolygon（頂点数が不足）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones:
  - id: "zone1"
    polygon: [[0, 0], [10, 10]]
output:
  directory: "output"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="少なくとも3つの頂点"):
        config.validate()

    # 無効なpolygon（座標が数値でない）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones:
  - id: "zone1"
    polygon: [[0, "invalid"], [10, 10], [20, 20]]
output:
  directory: "output"
"""
    config_path.write_text(yaml_content, encoding="utf-8")
    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="座標は数値"):
        config.validate()


def test_validate_camera_config_edge_cases(tmp_path: Path):
    """camera設定のエッジケース"""
    # 無効なposition_x（負の値）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
camera:
  position_x: -10
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="position_x"):
        config.validate()

    # 無効なmarker_color（要素数が不足）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: "output"
camera:
  marker_color: [0, 0]
"""
    config_path.write_text(yaml_content, encoding="utf-8")
    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="3要素リスト"):
        config.validate()


def test_validate_output_config_edge_cases(tmp_path: Path):
    """output設定のエッジケース"""
    # 無効なdirectory（None）
    yaml_content = """
video:
  input_path: "test_video.mov"
detection:
  model_name: "test_model"
  confidence_threshold: 0.7
  device: "cpu"
floormap:
  image_path: "test_floormap.png"
  image_width: 100
  image_height: 50
  image_origin_x: 0
  image_origin_y: 0
  image_x_mm_per_pixel: 1.0
  image_y_mm_per_pixel: 1.0
homography:
  matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
zones: []
output:
  directory: null
  save_detection_images: "not_bool"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    config = ConfigManager(str(config_path))
    with pytest.raises(ValueError, match="directory"):
        config.validate()


def test_save_file_permissions_error(tmp_path: Path):
    """ファイル権限エラーのテスト"""
    import os

    config = ConfigManager("nonexistent.yaml")

    # 読み取り専用ディレクトリを作成
    read_only_dir = tmp_path / "readonly"
    read_only_dir.mkdir()
    read_only_dir.chmod(0o555)

    output_path = read_only_dir / "config.yaml"

    # 権限エラーが発生することを確認（Windowsでは異なる動作）
    if os.name != "nt":  # Windows以外
        with pytest.raises((PermissionError, OSError, ValueError)):
            config.save(str(output_path))

    # クリーンアップ
    read_only_dir.chmod(0o755)
