#!/usr/bin/env python3
"""実装機能の動作確認スクリプト"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_tracking():
    """追跡機能のテスト"""
    print("=" * 60)
    print("追跡機能のテスト")
    print("=" * 60)

    import numpy as np

    from src.models.data_models import Detection
    from src.tracking import Tracker

    tracker = Tracker(min_hits=1)
    detection = Detection(
        bbox=(100.0, 100.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 200.0),
    )
    detection.features = np.random.rand(256).astype(np.float32)
    detection.features = detection.features / np.linalg.norm(detection.features)

    result = tracker.update([detection])
    print(f"✓ 追跡成功: Track ID {result[0].track_id}")
    print(f"  検出座標: {result[0].camera_coords}")
    print(f"  信頼度: {result[0].confidence:.3f}")


def test_calibration():
    """カメラキャリブレーションのテスト"""
    print("\n" + "=" * 60)
    print("カメラキャリブレーションのテスト")
    print("=" * 60)

    from src.calibration import CameraCalibrator

    _ = CameraCalibrator(chessboard_size=(9, 6))
    print("✓ キャリブレーター初期化成功")
    print("  注意: 実際のキャリブレーションにはチェスボード画像が必要です")
    print("  使用方法:")
    print("    image_paths = list(Path('calibration_images').glob('*.jpg'))")
    print("    camera_matrix, dist_coeffs = calibrator.calibrate_from_images(image_paths)")


def test_coordinate_transformation():
    """座標変換のテスト"""
    print("\n" + "=" * 60)
    print("座標変換のテスト")
    print("=" * 60)

    from src.transform.coordinate_transformer import CoordinateTransformer

    homography_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    transformer = CoordinateTransformer(
        homography_matrix=homography_matrix,
        floormap_config={
            "image_width": 1000,
            "image_height": 1000,
            "image_origin_x": 0,
            "image_origin_y": 0,
        },
    )

    result = transformer.transform((100.0, 200.0))
    print(f"✓ 座標変換成功: (100.0, 200.0) -> {result}")


def test_reprojection_error():
    """再投影誤差評価のテスト"""
    print("\n" + "=" * 60)
    print("再投影誤差評価のテスト")
    print("=" * 60)

    import numpy as np

    from src.calibration import ReprojectionErrorEvaluator

    evaluator = ReprojectionErrorEvaluator()
    result = evaluator.evaluate_homography(
        src_points=[(100.0, 100.0), (200.0, 150.0)],
        dst_points=[(50.0, 50.0), (150.0, 100.0)],
        homography_matrix=np.eye(3),
    )
    print("✓ 再投影誤差評価成功")
    print(f"  平均誤差: {result['mean_error']:.2f}px")
    print(f"  最大誤差: {result['max_error']:.2f}px")
    print(f"  標準偏差: {result['std_error']:.2f}px")


def test_export():
    """エクスポート機能のテスト"""
    print("\n" + "=" * 60)
    print("エクスポート機能のテスト")
    print("=" * 60)

    import numpy as np

    from src.models.data_models import Detection
    from src.tracking.kalman_filter import KalmanFilter
    from src.tracking.track import Track
    from src.utils.export_utils import TrajectoryExporter

    output_dir = Path("output/test_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = TrajectoryExporter(output_dir=output_dir)
    print(f"✓ エクスポーター初期化成功: {output_dir}")

    # サンプルトラックを作成
    detection = Detection(
        bbox=(100.0, 100.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 200.0),
    )
    kf = KalmanFilter()
    track = Track(track_id=1, detection=detection, kalman_filter=kf)
    track.trajectory = [(100.0, 100.0), (105.0, 105.0), (110.0, 110.0)]

    # CSVエクスポート
    csv_path = exporter.export_csv([track], filename="test_trajectories.csv")
    print(f"✓ CSVエクスポート成功: {csv_path}")

    # JSONエクスポート
    json_path = exporter.export_json([track], filename="test_trajectories.json")
    print(f"✓ JSONエクスポート成功: {json_path}")


def test_mot_metrics():
    """MOTメトリクス評価のテスト"""
    print("\n" + "=" * 60)
    print("MOTメトリクス評価のテスト")
    print("=" * 60)

    from src.evaluation.mot_metrics import MOTMetrics
    from src.models.data_models import Detection
    from src.tracking.kalman_filter import KalmanFilter
    from src.tracking.track import Track

    mot_metrics = MOTMetrics()

    ground_truth_tracks = [
        {
            "track_id": 1,
            "trajectory": [{"x": 100.0, "y": 100.0}, {"x": 105.0, "y": 105.0}],
        }
    ]

    detection = Detection(
        bbox=(100.0, 100.0, 50.0, 100.0),
        confidence=0.9,
        class_id=1,
        class_name="person",
        camera_coords=(125.0, 200.0),
    )
    kf = KalmanFilter()
    predicted_tracks = [Track(track_id=1, detection=detection, kalman_filter=kf)]

    metrics = mot_metrics.calculate_tracking_metrics(
        ground_truth_tracks=ground_truth_tracks,
        predicted_tracks=predicted_tracks,
        frame_count=10,
    )

    print("✓ MOTメトリクス評価成功")
    print(f"  MOTA: {metrics['MOTA']:.3f}")
    print(f"  IDF1: {metrics['IDF1']:.3f}")
    print(f"  ID Switches: {metrics['ID_Switches']}")


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("実装機能の動作確認")
    print("=" * 60 + "\n")

    try:
        test_tracking()
        test_calibration()
        test_coordinate_transformation()
        test_reprojection_error()
        test_export()
        test_mot_metrics()

        print("\n" + "=" * 60)
        print("✓ すべてのテストが成功しました！")
        print("=" * 60)
        print("\n詳細な使い方は docs/implementation_verification_guide.md を参照してください。")

    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
