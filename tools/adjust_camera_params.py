#!/usr/bin/env python3
"""
Interactive Camera Parameter Adjustment Tool.

This tool allows users to adjust camera physical parameters (height, pitch, yaw, roll, FOV)
AND camera position (x, y) via a GUI and visualize the projected floor grid on the camera image in real-time.

Features:
- Dual-view interface (Camera + Floormap)
- Camera position adjustment (Trackbars + Click on Map)
- Click-to-correspondence (click on camera image to see projected point on floormap)
- Keyboard fine-tuning
- Negative value support

Yaw符号規約:
- 正の値: 時計回り（右方向に回転）
- 負の値: 反時計回り（左方向に回転）
- yaw=0: 右下方向を向く
"""

import argparse
import logging
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.transform.coordinate_transformer import CoordinateTransformer
from src.utils import setup_logging

logger = logging.getLogger(__name__)


class CameraAdjuster:
    def __init__(self, config_path: str, reference_image_path: str):
        self.config_path = Path(config_path)
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Load images
        self.ref_image = cv2.imread(reference_image_path)
        if self.ref_image is None:
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

        # Use config_manager.get to support dot notation
        floormap_path = self.config_manager.get("floormap.image_path")
        if floormap_path:
            self.floormap_image = cv2.imread(floormap_path)
        else:
            self.floormap_image = None

        if self.floormap_image is None:
            logger.warning(f"Floormap image not found: {floormap_path}")
            self.floormap_image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        self.floormap_config = self.config.get("floormap", {})

        # Initial parameters from config
        self.initial_params = self._load_initial_params()

        # Current parameters (will be modified)
        self.current_params = self.initial_params.copy()

        # UI State
        self.window_name = "Camera Parameter Adjustment"
        self.selected_param_index = 0
        self.clicked_point = None
        self.param_names = [
            "Height (m)",
            "Pitch (deg)",
            "Yaw (deg)",
            "Roll (deg)",
            "FOV (deg)",
            "Pos X (px)",
            "Pos Y (px)",
        ]

        # Trackbar ranges and offsets
        # Name: (min, max, offset, scale)
        # Value = (Trackbar - offset) / scale
        # Trackbar = Value * scale + offset

        map_w = self.floormap_image.shape[1]
        map_h = self.floormap_image.shape[0]

        self.controls = [
            {"name": "Height", "min": 0, "max": 500, "offset": 0, "scale": 100.0},  # 0.00 - 5.00m
            {"name": "Pitch", "min": 0, "max": 3600, "offset": 1800, "scale": 10.0},  # -180.0 - 180.0 deg
            {"name": "Yaw", "min": 0, "max": 3600, "offset": 1800, "scale": 10.0},  # -180.0 - 180.0 deg
            {"name": "Roll", "min": 0, "max": 3600, "offset": 1800, "scale": 10.0},  # -180.0 - 180.0 deg
            {"name": "FOV", "min": 1, "max": 1790, "offset": 0, "scale": 10.0},  # 0.1 - 179.0 deg
            {"name": "Pos X", "min": 0, "max": map_w, "offset": 0, "scale": 1.0},  # 0 - MapWidth
            {"name": "Pos Y", "min": 0, "max": map_h, "offset": 0, "scale": 1.0},  # 0 - MapHeight
        ]

        self._setup_window()

    def _load_initial_params(self):
        params = self.config.get("camera_params", {})
        camera_config = self.config.get("camera", {})

        # Calculate FOV from focal length if available
        focal_length_x = float(params.get("focal_length_x", 1000.0))
        image_width = self.ref_image.shape[1]
        fov_deg = math.degrees(2 * math.atan(image_width / 2 / focal_length_x)) if focal_length_x > 0 else 60.0

        yaw_deg = float(params.get("yaw_deg", 0.0))

        # Position priority: camera_params > camera > default(0)
        # camera_params が計算用、camera が可視化用だが、位置は同期させる
        pos_x = float(params.get("position_x") or camera_config.get("position_x", 0))
        pos_y = float(params.get("position_y") or camera_config.get("position_y", 0))

        return {
            "height_m": float(params.get("height_m", 2.2)),
            "pitch_deg": float(params.get("pitch_deg", 45.0)),
            "yaw_deg": yaw_deg,
            "roll_deg": float(params.get("roll_deg", 0.0)),
            "fov_deg": float(fov_deg),
            "position_x": pos_x,
            "position_y": pos_y,
        }

    def _setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 1000)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # Create Trackbars
        self._create_trackbar(0, self.current_params["height_m"])
        self._create_trackbar(1, self.current_params["pitch_deg"])
        self._create_trackbar(2, self.current_params["yaw_deg"])
        self._create_trackbar(3, self.current_params["roll_deg"])
        self._create_trackbar(4, self.current_params["fov_deg"])
        self._create_trackbar(5, self.current_params["position_x"])
        self._create_trackbar(6, self.current_params["position_y"])

    def _create_trackbar(self, index, initial_value):
        ctrl = self.controls[index]
        pos = int(initial_value * ctrl["scale"] + ctrl["offset"])
        pos = max(ctrl["min"], min(pos, ctrl["max"]))

        def callback(val):
            real_val = (val - ctrl["offset"]) / ctrl["scale"]

            if index == 0:
                self.current_params["height_m"] = real_val
            elif index == 1:
                self.current_params["pitch_deg"] = real_val
            elif index == 2:
                self.current_params["yaw_deg"] = real_val
            elif index == 3:
                self.current_params["roll_deg"] = real_val
            elif index == 4:
                self.current_params["fov_deg"] = real_val
            elif index == 5:
                self.current_params["position_x"] = real_val
            elif index == 6:
                self.current_params["position_y"] = real_val

            self.update_visualization()

        cv2.createTrackbar(self.param_names[index], self.window_name, pos, ctrl["max"], callback)

    def _update_trackbar_pos(self, index, value):
        ctrl = self.controls[index]
        pos = int(value * ctrl["scale"] + ctrl["offset"])
        pos = max(ctrl["min"], min(pos, ctrl["max"]))
        cv2.setTrackbarPos(self.param_names[index], self.window_name, pos)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determine which view was clicked
            # Layout: [Camera View] [Floormap View]

            # We need to know the scaling to determine boundaries
            # Logic copied from update_visualization:
            vis_cam = self.ref_image
            vis_map = self.floormap_image
            scale = vis_cam.shape[0] / vis_map.shape[0]
            int(vis_map.shape[1] * scale)

            cam_w = vis_cam.shape[1]

            if x < cam_w:
                # Clicked on Camera Image
                if 0 <= x < cam_w and 0 <= y < vis_cam.shape[0]:
                    self.clicked_point = (x, y)
                    self.update_visualization()
            else:
                # Clicked on Floormap Image
                # Map x coordinate relative to map start
                map_click_x = x - cam_w
                map_click_y = y

                # Convert back to original map coordinates (undo resize)
                orig_map_x = int(map_click_x / scale)
                orig_map_y = int(map_click_y / scale)

                if 0 <= orig_map_x < vis_map.shape[1] and 0 <= orig_map_y < vis_map.shape[0]:
                    self.current_params["position_x"] = float(orig_map_x)
                    self.current_params["position_y"] = float(orig_map_y)
                    self._update_trackbar_pos(5, self.current_params["position_x"])
                    self._update_trackbar_pos(6, self.current_params["position_y"])
                    self.update_visualization()

    def update_visualization(self):
        # Calculate Focal Length from FOV
        image_width = self.ref_image.shape[1]
        image_height = self.ref_image.shape[0]
        fov = max(0.1, self.current_params["fov_deg"])
        focal_length = (image_width / 2) / math.tan(math.radians(fov / 2))

        calc_params = {
            "height_m": self.current_params["height_m"],
            "pitch_deg": self.current_params["pitch_deg"],
            "yaw_deg": self.current_params["yaw_deg"],
            "roll_deg": self.current_params["roll_deg"],
            "focal_length_x": focal_length,
            "focal_length_y": focal_length,
            "center_x": image_width / 2,
            "center_y": image_height / 2,
            "position_x": self.current_params["position_x"],
            "position_y": self.current_params["position_y"],
        }

        # Compute Homography
        H = None
        error_msg = None
        try:
            H = CoordinateTransformer.compute_homography_from_params(calc_params, self.floormap_config)
            det = np.linalg.det(H)
            if abs(det) < 1e-10:
                error_msg = f"Singular Matrix (det={det:.2e})"
                H = None
        except Exception as e:
            error_msg = f"Error: {type(e).__name__}"
            H = None

        # 1. Camera View
        vis_cam = self.ref_image.copy()
        if H is not None:
            self.draw_projected_grid(vis_cam, H)

        if self.clicked_point:
            cv2.circle(vis_cam, self.clicked_point, 5, (0, 0, 255), -1)
            cv2.circle(vis_cam, self.clicked_point, 7, (255, 255, 255), 2)

        # 2. Floormap View
        vis_map = self.floormap_image.copy()

        # Draw Camera Icon
        pos_x = int(self.current_params["position_x"])
        pos_y = int(self.current_params["position_y"])
        cv2.circle(vis_map, (pos_x, pos_y), 10, (0, 255, 255), -1)
        cv2.circle(vis_map, (pos_x, pos_y), 12, (0, 0, 0), 2)

        if H is not None:
            self.draw_camera_on_map(vis_map, H)

            if self.clicked_point:
                cam_pt = np.array([self.clicked_point[0], self.clicked_point[1], 1.0])
                floor_pt_h = H @ cam_pt
                if abs(floor_pt_h[2]) > 1e-10:
                    floor_x = int(floor_pt_h[0] / floor_pt_h[2])
                    floor_y = int(floor_pt_h[1] / floor_pt_h[2])
                    cv2.circle(vis_map, (floor_x, floor_y), 5, (0, 0, 255), -1)
                    cv2.circle(vis_map, (floor_x, floor_y), 7, (255, 255, 255), 2)
                    cv2.line(vis_map, (floor_x, floor_y - 10), (floor_x, floor_y + 10), (0, 0, 255), 2)
                    cv2.line(vis_map, (floor_x - 10, floor_y), (floor_x + 10, floor_y), (0, 0, 255), 2)

        # 3. Combine
        scale = vis_cam.shape[0] / vis_map.shape[0]
        vis_map_resized = cv2.resize(vis_map, (int(vis_map.shape[1] * scale), vis_cam.shape[0]))
        combined = np.hstack((vis_cam, vis_map_resized))

        # 4. UI Overlay
        self._draw_ui_overlay(combined, error_msg)

        cv2.imshow(self.window_name, combined)

    def _draw_ui_overlay(self, image, error_msg=None):
        _h, _w = image.shape[:2]
        overlay = image.copy()
        overlay_height = 320 if error_msg else 300
        cv2.rectangle(overlay, (0, 0), (400, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv2.putText(image, "Controls:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, "UP/DOWN: Select Param", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, "LEFT/RIGHT: Fine Tune", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, "R: Reset  S: Save  Q: Quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, "Click Map: Set Camera Position", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if error_msg:
            cv2.putText(image, f"ERROR: {error_msg}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            start_y = 160
        else:
            start_y = 140

        for i, name in enumerate(self.param_names):
            val = 0.0
            if i == 0:
                val = self.current_params["height_m"]
            elif i == 1:
                val = self.current_params["pitch_deg"]
            elif i == 2:
                val = self.current_params["yaw_deg"]
            elif i == 3:
                val = self.current_params["roll_deg"]
            elif i == 4:
                val = self.current_params["fov_deg"]
            elif i == 5:
                val = self.current_params["position_x"]
            elif i == 6:
                val = self.current_params["position_y"]

            text = f"{name}: {val:.2f}"

            color = (255, 255, 255)
            if i == self.selected_param_index:
                color = (0, 255, 0)
                text = "> " + text
            else:
                text = "  " + text

            cv2.putText(image, text, (10, start_y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    def draw_projected_grid(self, image, H):
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return

        h, w = self.floormap_image.shape[:2]
        grid_spacing = max(50, min(w, h) // 20)

        # Draw grid lines
        for x in range(0, w, grid_spacing):
            pt1 = H_inv @ np.array([x, 0, 1.0])
            pt2 = H_inv @ np.array([x, h, 1.0])
            if pt1[2] != 0 and pt2[2] != 0:
                p1 = (int(pt1[0] / pt1[2]), int(pt1[1] / pt1[2]))
                p2 = (int(pt2[0] / pt2[2]), int(pt2[1] / pt2[2]))
                cv2.line(image, p1, p2, (255, 100, 100), 1)

        for y in range(0, h, grid_spacing):
            pt1 = H_inv @ np.array([0, y, 1.0])
            pt2 = H_inv @ np.array([w, y, 1.0])
            if pt1[2] != 0 and pt2[2] != 0:
                p1 = (int(pt1[0] / pt1[2]), int(pt1[1] / pt1[2]))
                p2 = (int(pt2[0] / pt2[2]), int(pt2[1] / pt2[2]))
                cv2.line(image, p1, p2, (255, 100, 100), 1)

    def draw_camera_on_map(self, image, H):
        h, w = self.ref_image.shape[:2]
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        floor_corners = []
        for cx, cy in corners:
            pt = H @ np.array([cx, cy, 1.0])
            if pt[2] != 0:
                floor_corners.append((int(pt[0] / pt[2]), int(pt[1] / pt[2])))
            else:
                floor_corners.append(None)

        valid = [p for p in floor_corners if p]
        if len(valid) == 4:
            pts = np.array(valid, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 255), 2)

    def run(self):
        print("=" * 60)
        print("Camera Parameter Adjustment Tool")
        print("=" * 60)
        print("Controls:")
        print("  Arrow Up/Down / W/S / K/J: Select Parameter")
        print("  Arrow Left/Right / A/D / H/L: Fine Tune")
        print("  Click on Map: Set Camera Position")
        print("  R: Reset  S: Save  Q: Quit")
        print("=" * 60)

        while True:
            key = cv2.waitKey(10)
            if key == -1:
                continue

            key = key & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                self.save_config()
            elif key == ord("r"):
                self.current_params = self.initial_params.copy()
                for i in range(7):
                    val = 0
                    if i == 0:
                        val = self.current_params["height_m"]
                    elif i == 1:
                        val = self.current_params["pitch_deg"]
                    elif i == 2:
                        val = self.current_params["yaw_deg"]
                    elif i == 3:
                        val = self.current_params["roll_deg"]
                    elif i == 4:
                        val = self.current_params["fov_deg"]
                    elif i == 5:
                        val = self.current_params["position_x"]
                    elif i == 6:
                        val = self.current_params["position_y"]
                    self._update_trackbar_pos(i, val)

            # Navigation
            if key in [0, 82, ord("w"), ord("k")]:  # Up
                self.selected_param_index = (self.selected_param_index - 1) % 7
                self.update_visualization()
            elif key in [1, 84, ord("j")]:  # Down ('s' is save)
                self.selected_param_index = (self.selected_param_index + 1) % 7
                self.update_visualization()

            # Tuning
            elif key in [2, 81, ord("a"), ord("h")]:  # Left
                self.adjust_value(-1)
            elif key in [3, 83, ord("d"), ord("l")]:  # Right
                self.adjust_value(1)

    def adjust_value(self, direction):
        idx = self.selected_param_index
        # Determine step size
        step = 0.1
        if idx == 0:
            step = 0.01  # Height
        elif idx >= 5:
            step = 1.0  # Position

        val = 0.0
        if idx == 0:
            val = self.current_params["height_m"]
        elif idx == 1:
            val = self.current_params["pitch_deg"]
        elif idx == 2:
            val = self.current_params["yaw_deg"]
        elif idx == 3:
            val = self.current_params["roll_deg"]
        elif idx == 4:
            val = self.current_params["fov_deg"]
        elif idx == 5:
            val = self.current_params["position_x"]
        elif idx == 6:
            val = self.current_params["position_y"]

        new_val = val + (step * direction)

        if idx == 0:
            self.current_params["height_m"] = new_val
        elif idx == 1:
            self.current_params["pitch_deg"] = new_val
        elif idx == 2:
            self.current_params["yaw_deg"] = new_val
        elif idx == 3:
            self.current_params["roll_deg"] = new_val
        elif idx == 4:
            self.current_params["fov_deg"] = new_val
        elif idx == 5:
            self.current_params["position_x"] = new_val
        elif idx == 6:
            self.current_params["position_y"] = new_val

        self._update_trackbar_pos(idx, new_val)

    def save_config(self):
        """設定を保存する

        camera_params（計算用）とcamera（可視化用）の両方を更新します。
        """
        image_width = self.ref_image.shape[1]
        fov = max(0.1, self.current_params["fov_deg"])
        focal_length = (image_width / 2) / math.tan(math.radians(fov / 2))

        # camera_params: 計算用パラメータ
        new_params = {
            "height_m": float(self.current_params["height_m"]),
            "pitch_deg": float(self.current_params["pitch_deg"]),
            "yaw_deg": float(self.current_params["yaw_deg"]),
            "roll_deg": float(self.current_params["roll_deg"]),
            "focal_length_x": float(focal_length),
            "focal_length_y": float(focal_length),
            "center_x": float(self.ref_image.shape[1] / 2),
            "center_y": float(self.ref_image.shape[0] / 2),
            "position_x": float(self.current_params["position_x"]),
            "position_y": float(self.current_params["position_y"]),
        }

        # camera: 可視化用パラメータ（position_x/y と height_m を同期）
        camera_config = {
            "position_x": int(new_params["position_x"]),
            "position_y": int(new_params["position_y"]),
            "height_m": new_params["height_m"],
        }

        print("\n" + "=" * 50)
        print("Recommended Configuration (Copy to config.yaml):")
        print("=" * 50)
        print("# カメラ設定（可視化用）")
        print("camera:")
        print(f"  position_x: {camera_config['position_x']}")
        print(f"  position_y: {camera_config['position_y']}")
        print(f"  height_m: {camera_config['height_m']}")
        print("")
        print("# カメラ物理パラメータ（ホモグラフィ計算用）")
        print("camera_params:")
        for k, v in new_params.items():
            print(f"  {k}: {v}")
        print("=" * 50 + "\n")

        # 実際のconfig.yamlファイルを更新
        try:
            with open(self.config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # camera_paramsを更新
            if "camera_params" not in config:
                config["camera_params"] = {}
            config["camera_params"].update(new_params)

            # cameraを更新（position_x/y と height_m のみ）
            if "camera" not in config:
                config["camera"] = {}
            config["camera"]["position_x"] = camera_config["position_x"]
            config["camera"]["position_y"] = camera_config["position_y"]
            config["camera"]["height_m"] = camera_config["height_m"]

            # バックアップを作成
            backup_path = str(self.config_path) + ".backup"
            import shutil

            shutil.copy2(self.config_path, backup_path)
            logger.info(f"Backup created: {backup_path}")

            # 更新した設定を保存
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            print(f"✓ Updated {self.config_path}")
            print(f"  Backup saved to: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to update config.yaml: {e}")
            logger.info("Please manually update config.yaml with the values shown above")

        # 最適化されたパラメータを別ファイルにも保存
        output_path = "camera_params_optimized.yaml"
        with open(output_path, "w") as f:
            yaml.dump({"camera": camera_config, "camera_params": new_params}, f, allow_unicode=True)
        print(f"✓ Saved parameters to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Camera Parameter Adjustment")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--image", required=True, help="Path to reference camera image")
    args = parser.parse_args()

    setup_logging()

    try:
        adjuster = CameraAdjuster(args.config, args.image)
        adjuster.run()
    except Exception as e:
        logger.error(f"Failed to start adjuster: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
