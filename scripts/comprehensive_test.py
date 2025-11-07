#!/usr/bin/env python3
"""実装機能の包括的な動作確認テスト（解説付き）"""

import json
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ColorPrint:
    """カラー出力用のヘルパークラス"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

    @classmethod
    def header(cls, text):
        print(f"\n{cls.HEADER}{cls.BOLD}{'='*70}{cls.ENDC}")
        print(f"{cls.HEADER}{cls.BOLD}{text}{cls.ENDC}")
        print(f"{cls.HEADER}{cls.BOLD}{'='*70}{cls.ENDC}\n")

    @classmethod
    def section(cls, text):
        print(f"\n{cls.OKCYAN}{cls.BOLD}▶ {text}{cls.ENDC}")

    @classmethod
    def success(cls, text):
        print(f"{cls.OKGREEN}✓ {text}{cls.ENDC}")

    @classmethod
    def info(cls, text):
        print(f"{cls.OKBLUE}ℹ {text}{cls.ENDC}")

    @classmethod
    def warning(cls, text):
        print(f"{cls.WARNING}⚠ {text}{cls.ENDC}")

    @classmethod
    def error(cls, text):
        print(f"{cls.FAIL}✗ {text}{cls.ENDC}")


def test_tracking():
    """1. 追跡機能のテスト"""
    ColorPrint.header("1. 追跡機能（Tracker）の動作確認")

    ColorPrint.section("機能説明")
    print(
        """
    追跡機能は、フレーム間で同じ人物を一貫して追跡するための機能です。
    - Kalman Filter: 位置と速度を予測
    - Hungarian Algorithm: 検出結果とトラックを最適にマッチング
    - Similarity Calculator: 外観特徴量と位置情報で類似度を計算
    - Track管理: トラックの作成、更新、削除を管理
    """
    )

    try:
        import numpy as np

        from src.models.data_models import Detection
        from src.tracking import Tracker

        ColorPrint.section("テスト1: トラッカーの初期化")
        tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
        ColorPrint.success(f"トラッカー初期化成功 (max_age={tracker.max_age}, min_hits={tracker.min_hits})")

        ColorPrint.section("テスト2: 単一検出結果での追跡")
        detection1 = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )
        # 特徴量を追加（必須）
        detection1.features = np.random.rand(256).astype(np.float32)
        detection1.features = detection1.features / np.linalg.norm(detection1.features)

        tracked1 = tracker.update([detection1])
        assert len(tracked1) == 1, "追跡結果が1件であること"
        assert tracked1[0].track_id == 1, "Track IDが1であること"
        ColorPrint.success(f"追跡成功: Track ID {tracked1[0].track_id}, 座標 {tracked1[0].camera_coords}")

        ColorPrint.section("テスト3: 複数フレームでの追跡（IDの維持）")
        # フレーム2: 同じ人物が少し移動
        detection2 = Detection(
            bbox=(105.0, 105.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(130.0, 205.0),
        )
        # 同じ特徴量を使用（同じ人物として認識される）
        detection2.features = detection1.features.copy()

        tracked2 = tracker.update([detection2])
        assert tracked2[0].track_id == 1, "同じTrack IDが維持されること"
        ColorPrint.success(f"ID維持確認: Track ID {tracked2[0].track_id} (フレーム1と同じ)")

        ColorPrint.section("テスト4: 複数オブジェクトの追跡")
        # フレーム3: 2人の検出
        detection3a = Detection(
            bbox=(100.0, 100.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(125.0, 200.0),
        )
        detection3a.features = detection1.features.copy()

        detection3b = Detection(
            bbox=(200.0, 200.0, 50.0, 100.0),
            confidence=0.9,
            class_id=1,
            class_name="person",
            camera_coords=(225.0, 300.0),
        )
        detection3b.features = np.random.rand(256).astype(np.float32)
        detection3b.features = detection3b.features / np.linalg.norm(detection3b.features)

        tracked3 = tracker.update([detection3a, detection3b])
        assert len(tracked3) == 2, "2つの検出結果が追跡されること"
        assert tracked3[0].track_id != tracked3[1].track_id, "異なるTrack IDが割り当てられること"
        ColorPrint.success(f"複数追跡成功: Track ID {tracked3[0].track_id} と {tracked3[1].track_id}")

        ColorPrint.section("テスト5: トラック情報の取得")
        all_tracks = tracker.get_tracks()
        confirmed_tracks = tracker.get_confirmed_tracks()
        ColorPrint.success(f"総トラック数: {len(all_tracks)}, 確立されたトラック数: {len(confirmed_tracks)}")

        for track in confirmed_tracks:
            ColorPrint.info(
                f"  Track ID {track.track_id}: 軌跡点数={len(track.trajectory)}, ヒット数={track.hits}, 年齢={track.age}"
            )

        ColorPrint.success("✓ 追跡機能のすべてのテストが成功しました")
        return True

    except Exception as e:
        ColorPrint.error(f"追跡機能のテストに失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_calibration():
    """2. カメラキャリブレーションのテスト"""
    ColorPrint.header("2. カメラキャリブレーション機能の動作確認")

    ColorPrint.section("機能説明")
    print(
        """
    カメラキャリブレーションは、カメラの内部パラメータと歪み係数を推定する機能です。
    - チェスボード画像からコーナーを検出
    - 3D-2D対応点からカメラ行列と歪み係数を計算
    - 画像の歪み補正が可能になる
    """
    )

    try:
        from src.calibration import CameraCalibrator

        ColorPrint.section("テスト1: キャリブレーターの初期化")
        calibrator = CameraCalibrator(chessboard_size=(9, 6))
        ColorPrint.success(f"キャリブレーター初期化成功 (チェスボードサイズ: {calibrator.chessboard_size})")

        ColorPrint.section("テスト2: パラメータ取得メソッドの確認")
        assert calibrator.get_camera_matrix() is None, "初期状態ではカメラ行列がNone"
        assert calibrator.get_distortion_coefficients() is None, "初期状態では歪み係数がNone"
        ColorPrint.success("パラメータ取得メソッドが正常に動作")

        ColorPrint.warning("実際のキャリブレーションにはチェスボード画像が必要です")
        ColorPrint.info("使用方法:")
        print(
            """
        from pathlib import Path
        image_paths = list(Path('calibration_images').glob('*.jpg'))
        camera_matrix, dist_coeffs = calibrator.calibrate_from_images(image_paths)
        """
        )

        ColorPrint.success("✓ カメラキャリブレーション機能の基本テストが成功しました")
        return True

    except Exception as e:
        ColorPrint.error(f"カメラキャリブレーションのテストに失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_coordinate_transformation():
    """3. 座標変換（歪み補正付き）のテスト"""
    ColorPrint.header("3. 座標変換機能（歪み補正付き）の動作確認")

    ColorPrint.section("機能説明")
    print(
        """
    座標変換機能は、カメラ座標をフロアマップ座標に変換する機能です。
    - ホモグラフィ変換: 射影変換による座標変換
    - 歪み補正: カメラのレンズ歪みを補正
    - 原点オフセット: フロアマップの原点位置を考慮
    - スケール変換: ピクセル単位からmm単位への変換
    """
    )

    try:
        import numpy as np

        from src.transform.coordinate_transformer import CoordinateTransformer

        ColorPrint.section("テスト1: 基本的な座標変換（歪み補正なし）")
        homography_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        transformer = CoordinateTransformer(
            homography_matrix=homography_matrix,
            floormap_config={
                "image_width": 1000,
                "image_height": 1000,
                "image_origin_x": 7,
                "image_origin_y": 9,
            },
        )

        result = transformer.transform((100.0, 200.0))
        ColorPrint.success(f"座標変換成功: (100.0, 200.0) -> {result}")
        assert abs(result[0] - 107.0) < 0.01, "原点オフセットが適用されること"
        assert abs(result[1] - 209.0) < 0.01, "原点オフセットが適用されること"

        ColorPrint.section("テスト2: バッチ変換")
        camera_points = [(100.0, 200.0), (150.0, 250.0), (200.0, 300.0)]
        floor_points = transformer.transform_batch(camera_points)
        assert len(floor_points) == 3, "3つの座標が変換されること"
        ColorPrint.success(f"バッチ変換成功: {len(floor_points)}個の座標を変換")

        ColorPrint.section("テスト3: 歪み補正付き座標変換の設定")
        # ダミーのカメラ行列と歪み係数
        camera_matrix = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])

        transformer_with_correction = CoordinateTransformer(
            homography_matrix=homography_matrix,
            floormap_config={
                "image_width": 1000,
                "image_height": 1000,
                "image_origin_x": 0,
                "image_origin_y": 0,
            },
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            use_distortion_correction=True,
        )
        ColorPrint.success("歪み補正付き座標変換器の初期化成功")

        ColorPrint.section("テスト4: 歪み補正の効果確認")
        test_point = (500.0, 500.0)
        point_with = transformer_with_correction.transform(test_point)
        point_without = transformer.transform(test_point)
        ColorPrint.info(f"歪み補正あり: {point_with}")
        ColorPrint.info(f"歪み補正なし: {point_without}")

        ColorPrint.section("テスト5: 画像の歪み補正")
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        undistorted = transformer_with_correction.undistort_image(test_image)
        assert undistorted.shape == test_image.shape, "画像サイズが維持されること"
        ColorPrint.success("画像の歪み補正が正常に動作")

        ColorPrint.success("✓ 座標変換機能のすべてのテストが成功しました")
        return True

    except Exception as e:
        ColorPrint.error(f"座標変換のテストに失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_reprojection_error():
    """4. 再投影誤差評価のテスト"""
    ColorPrint.header("4. 再投影誤差評価機能の動作確認")

    ColorPrint.section("機能説明")
    print(
        """
    再投影誤差評価は、座標変換の精度を定量評価する機能です。
    - ホモグラフィ変換の精度評価
    - カメラキャリブレーションの精度評価
    - 誤差マップの生成
    - 平均誤差、最大誤差、標準偏差の計算
    """
    )

    try:
        import numpy as np

        from src.calibration import ReprojectionErrorEvaluator

        ColorPrint.section("テスト1: 評価器の初期化")
        evaluator = ReprojectionErrorEvaluator()
        ColorPrint.success("再投影誤差評価器の初期化成功")

        ColorPrint.section("テスト2: ホモグラフィ変換の精度評価")
        src_points = [(100.0, 100.0), (200.0, 150.0), (300.0, 200.0)]
        dst_points = [(50.0, 50.0), (150.0, 100.0), (250.0, 150.0)]
        homography_matrix = np.eye(3)

        result = evaluator.evaluate_homography(
            src_points=src_points,
            dst_points=dst_points,
            homography_matrix=homography_matrix,
        )

        assert "mean_error" in result, "平均誤差が含まれること"
        assert "max_error" in result, "最大誤差が含まれること"
        assert "min_error" in result, "最小誤差が含まれること"
        assert "std_error" in result, "標準偏差が含まれること"
        assert "errors" in result, "各点の誤差が含まれること"

        ColorPrint.success("再投影誤差評価成功")
        ColorPrint.info(f"  平均誤差: {result['mean_error']:.2f}px")
        ColorPrint.info(f"  最大誤差: {result['max_error']:.2f}px")
        ColorPrint.info(f"  最小誤差: {result['min_error']:.2f}px")
        ColorPrint.info(f"  標準偏差: {result['std_error']:.2f}px")
        ColorPrint.info(f"  各点の誤差: {[f'{e:.2f}' for e in result['errors']]}")

        ColorPrint.section("テスト3: 誤差マップの生成")
        error_map = evaluator.create_error_map(
            src_points=src_points,
            dst_points=dst_points,
            homography_matrix=homography_matrix,
            image_shape=(100, 200),  # (height, width)
        )
        assert error_map.shape == (100, 200), "誤差マップの形状が正しいこと"
        ColorPrint.success(f"誤差マップ生成成功: 形状 {error_map.shape}")

        ColorPrint.warning("誤差マップの可視化は目視確認が必要です")
        ColorPrint.info("可視化方法:")
        print(
            """
        import cv2
        error_map_normalized = (error_map / (error_map.max() + 1e-8) * 255).astype(np.uint8)
        error_colored = cv2.applyColorMap(error_map_normalized, cv2.COLORMAP_JET)
        cv2.imwrite("reprojection_error_map.jpg", error_colored)
        """
        )

        ColorPrint.success("✓ 再投影誤差評価機能のすべてのテストが成功しました")
        return True

    except Exception as e:
        ColorPrint.error(f"再投影誤差評価のテストに失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_export():
    """5. 軌跡データエクスポートのテスト"""
    ColorPrint.header("5. 軌跡データエクスポート機能の動作確認")

    ColorPrint.section("機能説明")
    print(
        """
    軌跡データエクスポート機能は、追跡結果を様々な形式で出力する機能です。
    - CSV形式: スプレッドシートで開ける形式
    - JSON形式: プログラムで読み込みやすい形式
    - 画像シーケンス: フレームごとの画像
    - 動画形式: 軌跡をアニメーション化
    """
    )

    try:
        import cv2
        import numpy as np

        from src.models.data_models import Detection
        from src.tracking.kalman_filter import KalmanFilter
        from src.tracking.track import Track
        from src.utils.export_utils import TrajectoryExporter

        ColorPrint.section("テスト1: エクスポーターの初期化")
        output_dir = Path("output/test_export_comprehensive")
        output_dir.mkdir(parents=True, exist_ok=True)

        exporter = TrajectoryExporter(output_dir=output_dir)
        ColorPrint.success(f"エクスポーター初期化成功: {output_dir}")

        ColorPrint.section("テスト2: サンプルトラックデータの作成")
        tracks = []
        for i in range(3):
            detection = Detection(
                bbox=(100.0 + i * 10, 100.0 + i * 10, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(125.0 + i * 10, 200.0 + i * 10),
            )
            kf = KalmanFilter()
            track = Track(track_id=i + 1, detection=detection, kalman_filter=kf)
            # 軌跡を追加
            track.trajectory = [
                (100.0 + i * 10, 100.0 + i * 10),
                (105.0 + i * 10, 105.0 + i * 10),
                (110.0 + i * 10, 110.0 + i * 10),
            ]
            tracks.append(track)

        ColorPrint.success(f"サンプルトラックデータ作成成功: {len(tracks)}個のトラック")

        ColorPrint.section("テスト3: CSV形式でエクスポート")
        csv_path = exporter.export_csv(tracks, filename="test_trajectories.csv")
        assert csv_path.exists(), "CSVファイルが生成されること"
        ColorPrint.success(f"CSVエクスポート成功: {csv_path}")

        # CSVファイルの内容を確認
        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
            ColorPrint.info(f"  CSVファイル: {len(lines)}行（ヘッダー含む）")
            ColorPrint.info("  最初の3行:")
            for i, line in enumerate(lines[:3]):
                print(f"    {line.strip()}")

        ColorPrint.section("テスト4: JSON形式でエクスポート")
        json_path = exporter.export_json(tracks, filename="test_trajectories.json")
        assert json_path.exists(), "JSONファイルが生成されること"
        ColorPrint.success(f"JSONエクスポート成功: {json_path}")

        # JSONファイルの内容を確認
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            ColorPrint.info(f"  トラック数: {data['metadata']['num_tracks']}")
            ColorPrint.info(f"  総軌跡点数: {data['metadata']['total_points']}")

        ColorPrint.section("テスト5: 画像シーケンスでエクスポート")
        # ダミーのフロアマップ画像を作成
        floormap = np.zeros((100, 100, 3), dtype=np.uint8)
        floormap.fill(255)  # 白背景

        image_paths = exporter.export_image_sequence(
            tracks=tracks,
            floormap_image=floormap,
            output_prefix="trajectory_frame",
            draw_trajectories=True,
            draw_ids=True,
        )
        assert len(image_paths) > 0, "画像ファイルが生成されること"
        ColorPrint.success(f"画像シーケンスエクスポート成功: {len(image_paths)}枚")
        ColorPrint.info(f"  出力ファイル例: {image_paths[0].name}")

        ColorPrint.section("テスト6: 動画形式でエクスポート")
        video_path = exporter.export_video(
            tracks=tracks,
            floormap_image=floormap,
            filename="test_trajectories.mp4",
            fps=2.0,
            draw_trajectories=True,
            draw_ids=True,
        )
        assert video_path.exists(), "動画ファイルが生成されること"
        ColorPrint.success(f"動画エクスポート成功: {video_path}")
        ColorPrint.info(f"  ファイルサイズ: {video_path.stat().st_size / 1024:.2f} KB")

        ColorPrint.warning("画像と動画の内容は目視確認が必要です")
        ColorPrint.info(f"確認方法: {output_dir} ディレクトリ内のファイルを確認してください")

        ColorPrint.success("✓ 軌跡データエクスポート機能のすべてのテストが成功しました")
        return True

    except Exception as e:
        ColorPrint.error(f"エクスポート機能のテストに失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mot_metrics():
    """6. MOTメトリクス評価のテスト"""
    ColorPrint.header("6. MOTメトリクス評価機能の動作確認")

    ColorPrint.section("機能説明")
    print(
        """
    MOT（Multiple Object Tracking）メトリクスは、追跡精度を評価する標準的な指標です。
    - MOTA: Multiple Object Tracking Accuracy（0.0-1.0、高いほど良い）
    - IDF1: ID F1 Score（0.0-1.0、高いほど良い）
    - ID Switches: IDスイッチの回数（少ないほど良い）
    """
    )

    try:
        from src.evaluation.mot_metrics import MOTMetrics
        from src.models.data_models import Detection
        from src.tracking.kalman_filter import KalmanFilter
        from src.tracking.track import Track

        ColorPrint.section("テスト1: MOTメトリクス評価器の初期化")
        mot_metrics = MOTMetrics()
        ColorPrint.success("MOTメトリクス評価器の初期化成功")

        ColorPrint.section("テスト2: Ground Truthと予測トラックの準備")
        ground_truth_tracks = [
            {
                "track_id": 1,
                "trajectory": [
                    {"x": 100.0, "y": 100.0},
                    {"x": 105.0, "y": 105.0},
                    {"x": 110.0, "y": 110.0},
                ],
            },
            {
                "track_id": 2,
                "trajectory": [
                    {"x": 200.0, "y": 200.0},
                    {"x": 205.0, "y": 205.0},
                ],
            },
        ]

        predicted_tracks = []
        for i in range(2):
            detection = Detection(
                bbox=(100.0 + i * 100, 100.0 + i * 100, 50.0, 100.0),
                confidence=0.9,
                class_id=1,
                class_name="person",
                camera_coords=(125.0 + i * 100, 200.0 + i * 100),
            )
            kf = KalmanFilter()
            track = Track(track_id=i + 1, detection=detection, kalman_filter=kf)
            track.trajectory = [
                (100.0 + i * 100, 100.0 + i * 100),
                (105.0 + i * 100, 105.0 + i * 100),
            ]
            predicted_tracks.append(track)

        ColorPrint.success(f"Ground Truth: {len(ground_truth_tracks)}トラック, 予測: {len(predicted_tracks)}トラック")

        ColorPrint.section("テスト3: MOTメトリクスの計算")
        metrics = mot_metrics.calculate_tracking_metrics(
            ground_truth_tracks=ground_truth_tracks,
            predicted_tracks=predicted_tracks,
            frame_count=10,
        )

        assert "MOTA" in metrics, "MOTAが含まれること"
        assert "IDF1" in metrics, "IDF1が含まれること"
        assert "ID_Switches" in metrics, "ID_Switchesが含まれること"

        assert 0.0 <= metrics["MOTA"] <= 1.0, "MOTAが0.0-1.0の範囲内であること"
        assert 0.0 <= metrics["IDF1"] <= 1.0, "IDF1が0.0-1.0の範囲内であること"

        ColorPrint.success("MOTメトリクス計算成功")
        ColorPrint.info(f"  MOTA: {metrics['MOTA']:.3f} (0.0-1.0, 高いほど良い)")
        ColorPrint.info(f"  IDF1: {metrics['IDF1']:.3f} (0.0-1.0, 高いほど良い)")
        ColorPrint.info(f"  ID Switches: {metrics['ID_Switches']} (少ないほど良い)")

        ColorPrint.section("テスト4: 個別メトリクスの計算")
        mota = mot_metrics.calculate_mota(
            ground_truth_tracks=ground_truth_tracks,
            predicted_tracks=predicted_tracks,
            frame_count=10,
        )
        idf1 = mot_metrics.calculate_idf1(
            ground_truth_tracks=ground_truth_tracks,
            predicted_tracks=predicted_tracks,
        )

        ColorPrint.success(f"個別メトリクス計算成功: MOTA={mota:.3f}, IDF1={idf1:.3f}")

        ColorPrint.success("✓ MOTメトリクス評価機能のすべてのテストが成功しました")
        return True

    except Exception as e:
        ColorPrint.error(f"MOTメトリクス評価のテストに失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """7. 統合テスト"""
    ColorPrint.header("7. 統合テスト（複数機能の連携）")

    ColorPrint.section("機能説明")
    print(
        """
    統合テストでは、複数の機能が連携して動作することを確認します。
    - 追跡 → 座標変換 → エクスポートの流れ
    - 実際の使用シーンに近い動作確認
    """
    )

    try:
        import numpy as np

        from src.models.data_models import Detection
        from src.tracking import Tracker
        from src.transform.coordinate_transformer import CoordinateTransformer
        from src.utils.export_utils import TrajectoryExporter

        ColorPrint.section("テスト1: 追跡 → 座標変換の連携")
        # 追跡
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

        tracked = tracker.update([detection])
        assert len(tracked) == 1, "追跡が成功すること"

        # 座標変換
        homography_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        transformer = CoordinateTransformer(
            homography_matrix=homography_matrix,
            floormap_config={"image_width": 1000, "image_height": 1000},
        )

        floor_coords = transformer.transform(tracked[0].camera_coords)
        tracked[0].floor_coords = floor_coords

        ColorPrint.success(f"追跡 → 座標変換成功: フロアマップ座標 {floor_coords}")

        ColorPrint.section("テスト2: 追跡 → エクスポートの連携")
        tracks = tracker.get_tracks()
        assert len(tracks) > 0, "トラックが存在すること"

        output_dir = Path("output/test_integration")
        output_dir.mkdir(parents=True, exist_ok=True)
        exporter = TrajectoryExporter(output_dir=output_dir)

        csv_path = exporter.export_csv(tracks, filename="integration_test.csv")
        assert csv_path.exists(), "CSVファイルが生成されること"

        ColorPrint.success(f"追跡 → エクスポート成功: {csv_path}")

        ColorPrint.success("✓ 統合テストが成功しました")
        return True

    except Exception as e:
        ColorPrint.error(f"統合テストに失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    ColorPrint.header("実装機能の包括的な動作確認テスト")

    print(
        """
    このスクリプトは、実装したすべての機能を自動的にテストします。
    各機能について、解説とテスト結果を表示します。

    目視確認が必要な項目:
    - 画像ファイル（軌跡の可視化）
    - 動画ファイル（軌跡のアニメーション）
    - Streamlitアプリ（インタラクティブ可視化）
    """
    )

    results = {}

    # 各機能をテスト
    results["追跡機能"] = test_tracking()
    results["カメラキャリブレーション"] = test_calibration()
    results["座標変換"] = test_coordinate_transformation()
    results["再投影誤差評価"] = test_reprojection_error()
    results["エクスポート機能"] = test_export()
    results["MOTメトリクス評価"] = test_mot_metrics()
    results["統合テスト"] = test_integration()

    # 結果サマリー
    ColorPrint.header("テスト結果サマリー")

    for name, result in results.items():
        if result:
            ColorPrint.success(f"{name}: 成功")
        else:
            ColorPrint.error(f"{name}: 失敗")

    success_count = sum(results.values())
    total_count = len(results)

    print(f"\n成功: {success_count}/{total_count}")

    if success_count == total_count:
        ColorPrint.success("\n✓ すべてのテストが成功しました！")
        print("\n目視確認が必要な項目:")
        print("  1. output/test_export_comprehensive/ 内の画像と動画")
        print("  2. Streamlitアプリ: streamlit run tools/interactive_visualizer.py")
        print("  3. 目視確認ツール: python tools/visual_inspection.py --mode tracking --session <session_id>")
        return 0
    else:
        ColorPrint.error(f"\n✗ {total_count - success_count}個のテストが失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
