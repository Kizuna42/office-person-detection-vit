"""Example usage of the Visualizer module."""

import cv2
import numpy as np
from src.visualization import Visualizer
from src.aggregation import Aggregator
from src.models import Detection, AggregationResult

# Initialize visualizer
visualizer = Visualizer(debug_mode=True)

# Example 1: Visualize detections on a frame
print("Example 1: Visualizing detections")
frame = cv2.imread("input/sample_frame.jpg")  # Replace with actual frame
if frame is not None:
    detections = [
        Detection(
            bbox=(100, 100, 50, 100),
            confidence=0.95,
            class_id=1,
            class_name="person",
            camera_coords=(125, 200),
            floor_coords=(300, 400),
            zone_ids=["zone_a"]
        )
    ]
    
    result = visualizer.draw_detections(frame, detections)
    visualizer.save_image(result, "output/detections/example_detections.png")
    print("✓ Saved detection visualization")

# Example 2: Visualize with attention map
print("\nExample 2: Visualizing with attention map")
if frame is not None:
    # Simulate attention map (in real usage, get from ViTDetector)
    attention_map = np.random.rand(196)  # 14x14 patches
    
    result = visualizer.visualize_with_attention(
        frame,
        detections,
        attention_map,
        alpha=0.4
    )
    visualizer.save_image(result, "output/detections/example_attention.png")
    print("✓ Saved attention map visualization")

# Example 3: Generate time series graphs
print("\nExample 3: Generating time series graphs")
aggregator = Aggregator()

# Simulate aggregation data
for i in range(12):
    timestamp = f"12:{10 + i * 5:02d}"
    aggregator.results.append(AggregationResult(timestamp, "zone_a", 5 + i % 3))
    aggregator.results.append(AggregationResult(timestamp, "zone_b", 3 + i % 4))
    aggregator._zone_data["zone_a"].append(5 + i % 3)
    aggregator._zone_data["zone_b"].append(3 + i % 4)

# Generate time series graph
visualizer.plot_time_series(aggregator, "output/graphs/time_series.png")
print("✓ Saved time series graph")

# Generate statistics graph
visualizer.plot_zone_statistics(aggregator, "output/graphs/statistics.png")
print("✓ Saved statistics graph")

# Generate heatmap
visualizer.plot_heatmap(aggregator, "output/graphs/heatmap.png")
print("✓ Saved heatmap")

print("\n" + "="*50)
print("All examples completed!")
print("Check the output/ directory for results.")
print("="*50)
