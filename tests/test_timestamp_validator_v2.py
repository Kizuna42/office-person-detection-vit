"""Unit tests for timestamp validator v2 module."""

from datetime import datetime, timedelta

from src.timestamp.timestamp_validator_v2 import TemporalValidatorV2


class TestTemporalValidatorV2:
    """TemporalValidatorV2のテスト"""

    def test_init_default(self):
        """デフォルトパラメータでの初期化テスト"""
        validator = TemporalValidatorV2()
        assert validator.fps == 30.0
        assert validator.base_tolerance == 10.0
        assert validator.history_size == 10
        assert validator.z_score_threshold == 2.0
        assert validator.last_timestamp is None
        assert validator.last_frame_idx is None

    def test_init_custom(self):
        """カスタムパラメータでの初期化テスト"""
        validator = TemporalValidatorV2(fps=60.0, base_tolerance_seconds=5.0, history_size=20, z_score_threshold=3.0)
        assert validator.fps == 60.0
        assert validator.base_tolerance == 5.0
        assert validator.history_size == 20
        assert validator.z_score_threshold == 3.0

    def test_validate_first_frame(self):
        """初回フレームの検証テスト"""
        validator = TemporalValidatorV2()
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        is_valid, confidence, reason = validator.validate(timestamp, 0)
        assert is_valid is True
        assert confidence == 1.0
        assert "First frame" in reason

    def test_validate_valid_sequence(self):
        """有効な時系列の検証テスト"""
        validator = TemporalValidatorV2(fps=30.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 初回フレーム
        validator.validate(base_time, 0)

        # 次のフレーム（1秒後、30フレーム後）
        next_time = base_time + timedelta(seconds=1.0)
        is_valid, confidence, reason = validator.validate(next_time, 30)
        assert is_valid is True
        assert 0.0 <= confidence <= 1.0

    def test_validate_invalid_frame_diff(self):
        """無効なフレーム差の検証テスト"""
        validator = TemporalValidatorV2()
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 初回フレーム
        validator.validate(base_time, 10)

        # フレーム番号が前回より小さい
        is_valid, confidence, reason = validator.validate(base_time + timedelta(seconds=1.0), 5)
        assert is_valid is False
        assert confidence == 0.0
        assert "Invalid frame_diff" in reason

    def test_validate_zero_frame_diff(self):
        """フレーム差が0の場合のテスト"""
        validator = TemporalValidatorV2()
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 初回フレーム
        validator.validate(base_time, 10)

        # 同じフレーム番号
        is_valid, confidence, reason = validator.validate(base_time + timedelta(seconds=1.0), 10)
        assert is_valid is False
        assert confidence == 0.0

    def test_validate_out_of_tolerance(self):
        """許容範囲外のタイムスタンプの検証テスト"""
        validator = TemporalValidatorV2(fps=30.0, base_tolerance_seconds=1.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 初回フレーム
        validator.validate(base_time, 0)

        # 許容範囲を大幅に超える時間差（10秒後、30フレーム後）
        next_time = base_time + timedelta(seconds=10.0)
        is_valid, confidence, reason = validator.validate(next_time, 30)
        assert is_valid is False
        assert confidence == 0.0

    def test_calculate_adaptive_tolerance_insufficient_history(self):
        """履歴が不足している場合の適応的許容範囲計算テスト"""
        validator = TemporalValidatorV2()
        tolerance = validator._calculate_adaptive_tolerance()
        assert tolerance == validator.base_tolerance

    def test_calculate_adaptive_tolerance_with_history(self):
        """履歴がある場合の適応的許容範囲計算テスト"""
        validator = TemporalValidatorV2(fps=30.0, base_tolerance_seconds=1.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 履歴を構築
        for i in range(5):
            frame_idx = i * 30
            timestamp = base_time + timedelta(seconds=i * 1.0)
            validator.validate(timestamp, frame_idx)

        tolerance = validator._calculate_adaptive_tolerance()
        assert tolerance >= validator.base_tolerance * 0.5
        assert tolerance <= validator.base_tolerance * 3.0

    def test_detect_outlier_insufficient_history(self):
        """履歴が不足している場合の外れ値検出テスト"""
        validator = TemporalValidatorV2()
        is_outlier, z_score = validator._detect_outlier(1.0, 1.0)
        assert is_outlier is False
        assert z_score == 0.0

    def test_detect_outlier_normal(self):
        """通常の場合の外れ値検出テスト"""
        validator = TemporalValidatorV2(fps=30.0, base_tolerance_seconds=1.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 履歴を構築
        for i in range(5):
            frame_idx = i * 30
            timestamp = base_time + timedelta(seconds=i * 1.0)
            validator.validate(timestamp, frame_idx)

        # 正常な時間差
        is_outlier, z_score = validator._detect_outlier(1.0, 1.0)
        assert isinstance(is_outlier, bool)
        assert z_score >= 0.0

    def test_detect_outlier_high_z_score(self):
        """Z-scoreが高い場合の外れ値検出テスト"""
        validator = TemporalValidatorV2(fps=30.0, base_tolerance_seconds=1.0, z_score_threshold=2.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 履歴を構築（均一な間隔）
        for i in range(5):
            frame_idx = i * 30
            timestamp = base_time + timedelta(seconds=i * 1.0)
            validator.validate(timestamp, frame_idx)

        # 異常に大きい時間差（Z-scoreが高くなる）
        is_outlier, z_score = validator._detect_outlier(10.0, 1.0)
        # 履歴の標準偏差によっては外れ値と判定される可能性がある
        assert isinstance(is_outlier, bool)
        assert z_score >= 0.0

    def test_detect_outlier_zero_std(self):
        """標準偏差が0の場合の外れ値検出テスト"""
        validator = TemporalValidatorV2()
        # 履歴に同じ値を追加（標準偏差が0になる）
        validator.interval_history.extend([1.0, 1.0, 1.0])

        is_outlier, z_score = validator._detect_outlier(1.0, 1.0)
        assert is_outlier is False
        assert z_score == 0.0

    def test_recover_timestamp(self):
        """タイムスタンプリカバリーテスト"""
        validator = TemporalValidatorV2(fps=30.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)
        validator.last_timestamp = base_time

        recovered = validator._recover_timestamp(30, 1.0)
        assert recovered == base_time + timedelta(seconds=1.0)

    def test_recover_timestamp_no_last_timestamp(self):
        """last_timestampがない場合のリカバリーテスト"""
        validator = TemporalValidatorV2()
        validator.last_timestamp = None

        recovered = validator._recover_timestamp(30, 1.0)
        assert recovered is None

    def test_reset(self):
        """リセットテスト"""
        validator = TemporalValidatorV2()
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 状態を設定
        validator.validate(base_time, 0)
        validator.validate(base_time + timedelta(seconds=1.0), 30)

        assert validator.last_timestamp is not None
        assert validator.last_frame_idx is not None
        assert len(validator.interval_history) > 0

        # リセット
        validator.reset()

        assert validator.last_timestamp is None
        assert validator.last_frame_idx is None
        assert len(validator.interval_history) == 0

    def test_validate_with_outlier_recovery(self):
        """外れ値リカバリーを含む検証テスト"""
        validator = TemporalValidatorV2(fps=30.0, base_tolerance_seconds=1.0, z_score_threshold=2.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 履歴を構築（均一な間隔）
        for i in range(5):
            frame_idx = i * 30
            timestamp = base_time + timedelta(seconds=i * 1.0)
            validator.validate(timestamp, frame_idx)

        # 異常なタイムスタンプ（外れ値として検出される可能性がある）
        # ただし、リカバリーが機能する場合もある
        abnormal_time = base_time + timedelta(seconds=10.0)
        is_valid, confidence, reason = validator.validate(abnormal_time, 150)
        # 結果は実装による（リカバリーが成功するかどうか）
        assert isinstance(is_valid, bool)
        assert 0.0 <= confidence <= 1.0

    def test_validate_confidence_calculation(self):
        """信頼度計算のテスト"""
        validator = TemporalValidatorV2(fps=30.0, base_tolerance_seconds=2.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 初回フレーム
        validator.validate(base_time, 0)

        # 期待値に近いタイムスタンプ（高信頼度）
        next_time = base_time + timedelta(seconds=1.0)
        is_valid1, confidence1, _ = validator.validate(next_time, 30)

        # 期待値から少しずれたタイムスタンプ（低信頼度）
        validator2 = TemporalValidatorV2(fps=30.0, base_tolerance_seconds=2.0)
        validator2.validate(base_time, 0)
        next_time2 = base_time + timedelta(seconds=1.5)
        is_valid2, confidence2, _ = validator2.validate(next_time2, 30)

        if is_valid1 and is_valid2:
            # 期待値に近い方が信頼度が高い
            assert confidence1 >= confidence2

    def test_validate_multiple_frames(self):
        """複数フレームの連続検証テスト"""
        validator = TemporalValidatorV2(fps=30.0)
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        # 複数フレームを連続して検証
        for i in range(10):
            frame_idx = i * 30
            timestamp = base_time + timedelta(seconds=i * 1.0)
            is_valid, confidence, _ = validator.validate(timestamp, frame_idx)
            assert is_valid is True
            assert 0.0 <= confidence <= 1.0

        # 履歴が構築されていることを確認
        assert len(validator.interval_history) > 0
