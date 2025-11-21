#!/usr/bin/env python3
"""
Estimate Camera Parameters from Annotated Image.

This script analyzes a floor map image with red annotations (camera dot and FOV lines)
to estimate the camera's position (x, y), yaw angle, and field of view (FOV).

Usage:
    python3 tools/estimate_from_image.py --image path/to/image.png
"""

import argparse
import math

import cv2
import numpy as np
import yaml


def estimate_parameters(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # 2. Preprocess & Color Segmentation (Red)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red has two ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No red annotations found.")
        return None

    # 4. Analyze Features
    # We expect:
    # - One "blob" for the camera position (dot/circle)
    # - Two "lines" or elongated blobs for the FOV rays
    # Or sometimes they are all connected.

    # Let's try to find the camera point first.
    # It's likely the point where lines converge, or a distinct circle.
    # If connected, we might need skeletonization or corner detection.

    # Approach: Find the "Camera" point.
    # It is usually the vertex of the "V" shape formed by the FOV lines.

    # Let's use HoughLinesP to find lines.
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

    debug_img = img.copy()

    fov_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            fov_lines.append(((x1, y1), (x2, y2)))

    # If we have many lines, we need to filter/merge them into 2 main directions.
    # But first, let's find the camera position.
    # It should be the intersection of the two main lines.

    # Alternative: Find the centroid of the "blob" that looks like a dot?
    # In the user image, there is a "camera" text and a dot.
    # And two long lines.

    # Let's look for the intersection point of the longest lines.
    def line_intersection(line1, line2):
        p1, p2 = line1
        p3, p4 = line2

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return None  # Parallel

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom

        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)

    # Filter lines: keep only long ones
    long_lines = [line for line in fov_lines if math.hypot(line[0][0] - line[1][0], line[0][1] - line[1][1]) > 100]

    if len(long_lines) < 2:
        print("Warning: Could not detect at least 2 long FOV lines. Trying all lines.")
        long_lines = fov_lines

    # Find intersection of all pairs and cluster them?
    intersections = []
    for i in range(len(long_lines)):
        for j in range(i + 1, len(long_lines)):
            pt = line_intersection(long_lines[i], long_lines[j])
            # Check if intersection is within or near image bounds
            if pt and -1000 < pt[0] < img.shape[1] + 1000 and -1000 < pt[1] < img.shape[0] + 1000:
                intersections.append(pt)

    if not intersections:
        print("Error: Could not find line intersections.")
        return None

    # The camera position is likely the most common intersection point
    # Simple clustering
    intersections = np.array(intersections)
    mean_pt = np.mean(intersections, axis=0)

    # Refined: Remove outliers
    distances = np.linalg.norm(intersections - mean_pt, axis=1)
    valid_pts = intersections[distances < 50]  # 50px radius

    camera_pos = mean_pt if len(valid_pts) == 0 else np.mean(valid_pts, axis=0)

    cx, cy = camera_pos
    cv2.circle(debug_img, (int(cx), int(cy)), 10, (255, 0, 0), -1)

    # Now determine angles of the lines radiating from this point
    angles = []
    for line in long_lines:
        p1, p2 = line
        # Determine which end is further from camera
        d1 = math.hypot(p1[0] - cx, p1[1] - cy)
        d2 = math.hypot(p2[0] - cx, p2[1] - cy)

        far_pt = p2 if d2 > d1 else p1

        angle = math.degrees(math.atan2(far_pt[1] - cy, far_pt[0] - cx))
        angles.append(angle)

    # We expect 2 main clusters of angles (the two sides of the FOV)
    # Histogram or clustering
    angles = np.array(angles)
    # Normalize to -180..180

    # Simple K-Means for k=2?
    # Or just sort and find largest gap?
    angles.sort()

    # Group angles that are close (e.g., within 10 degrees)
    clusters = []
    if len(angles) > 0:
        current_cluster = [angles[0]]
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i - 1]
            if diff > 300:  # Wrap around case (e.g. -179 and 179)
                # Handle wrap around later, for now assume simple case
                pass

            if diff < 10:
                current_cluster.append(angles[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [angles[i]]
        clusters.append(current_cluster)

    # We need exactly 2 main directions
    main_angles = []
    for c in clusters:
        if len(c) >= 1:  # Threshold for noise
            avg = sum(c) / len(c)
            main_angles.append(avg)

    if len(main_angles) < 2:
        print(f"Error: Could not identify 2 distinct FOV edges. Found angles: {main_angles}")
        # Fallback: Use min and max angle?
        if len(angles) >= 2:
            main_angles = [angles[0], angles[-1]]
        else:
            return None

    # Sort main angles
    main_angles.sort()

    # Calculate Yaw and FOV
    # Yaw is the bisector
    # FOV is the difference

    a1, a2 = main_angles[0], main_angles[1]

    # Check for wrap-around (e.g. 170 and -170)
    if a2 - a1 > 180:
        # The angle crosses the -180/180 boundary
        # e.g. a1 = -170, a2 = 170. Diff = 340.
        # Real diff is 20.
        fov = 360 - (a2 - a1)
        yaw = (a1 + a2 + 360) / 2
        if yaw > 180:
            yaw -= 360
    else:
        fov = a2 - a1
        yaw = (a1 + a2) / 2

    # Save debug image
    cv2.imwrite("debug_estimation.png", debug_img)
    print("Saved debug image to debug_estimation.png")

    return {
        "position_x": int(cx),
        "position_y": int(cy),
        "yaw_deg": float(round(yaw, 1)),
        "fov_deg": float(round(fov, 1)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    result = estimate_parameters(args.image)

    if result:
        print("\nEstimated Parameters:")
        print(yaml.dump(result, sort_keys=False))
    else:
        print("Estimation failed.")
