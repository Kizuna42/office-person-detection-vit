# 改善機能の統合とテストガイド

## 📋 概要

Phase 3で実装された改善機能を統合しました。**デフォルトでは既存実装を使用**し、精度テスト後に改善機能を有効化できるようになっています。

## 🔧 統合した機能

### 1. TemporalValidatorV2（適応的許容範囲と異常値リカバリー）

- **適応的許容範囲**: 過去N個のフレーム間隔から動的に許容範囲を計算
- **外れ値検出**: Z-score法による異常値検出
- **異常値リカバリー**: 前後フレームからの線形補間によるタイムスタンプ補正

### 2. 重み付けスキーム（コンセンサスアルゴリズム）

- エンジン別の重み付け（Tesseractを優先）
- エンジン間の一致度を評価指標に追加
- 一致度が高い場合は信頼度を10%向上

### 3. 投票ロジック（コンセンサスアルゴリズム）

- 2/3以上のエンジンが一致したテキストを採用
- 複数エンジン使用時に有効

## ⚙️ 設定方法

### config.yamlでの設定

```yaml
timestamp:
  extraction:
    # 改善機能の有効化（デフォルト: false）
    use_improved_validator: false  # TemporalValidatorV2を使用
    use_weighted_consensus: false  # 重み付けスキームを使用
    use_voting_consensus: false    # 投票ロジックを使用

    # TemporalValidatorV2の設定
    validator:
      base_tolerance_seconds: 10.0  # ベース許容範囲（秒）
      history_size: 10              # 履歴サイズ
      z_score_threshold: 2.0        # Z-score閾値
```

### デフォルト設定

- **use_improved_validator**: `false`（既存のTemporalValidatorを使用）
- **use_weighted_consensus**: `false`（ベースラインのコンセンサスアルゴリズムを使用）
- **use_voting_consensus**: `false`（ベースラインのコンセンサスアルゴリズムを使用）

**重要**: デフォルトでは既存実装を使用するため、精度低下のリスクはありません。

## 🧪 精度テストの実行

改善機能の精度をテストするには、以下のコマンドを実行します：

```bash
# 基本的なテスト（最初の10フレーム）
python tools/test_improved_features.py --video input/merged_moviefiles.mov

# 特定のフレームをテスト
python tools/test_improved_features.py --video input/merged_moviefiles.mov --frames 0 100 200 300 400

# 出力ディレクトリを指定
python tools/test_improved_features.py --video input/merged_moviefiles.mov --output output/improved_features_test
```

### テスト結果の確認

テスト実行後、以下の結果が出力されます：

1. **各設定の成功率と平均信頼度**
2. **改善判定**（ベースラインと比較）
3. **推奨設定**（最も精度が高い設定）

結果は `output/improved_features_test/improved_features_comparison.json` に保存されます。

### 改善判定の基準

- **成功率**: ベースライン以上
- **信頼度**: ベースラインから5%以内の低下（許容範囲）

両方を満たす場合、改善機能は有効化を推奨されます。

## 📊 改善機能の有効化

テスト結果で改善が確認された場合、`config.yaml`で有効化します：

```yaml
timestamp:
  extraction:
    # テスト結果に基づいて有効化
    use_improved_validator: true   # 改善が確認された場合
    use_weighted_consensus: true   # 改善が確認された場合
    use_voting_consensus: false    # 複数エンジン使用時のみ推奨
```

## ⚠️ 注意事項

1. **デフォルトは既存実装**: 精度低下のリスクを避けるため、デフォルトでは既存実装を使用します。

2. **テスト後の有効化**: 改善機能を有効化する前に、必ず精度テストを実行してください。

3. **段階的な有効化**: 複数の改善機能がある場合、1つずつ有効化してテストすることを推奨します。

4. **複数エンジン使用時**: 重み付けスキームと投票ロジックは、複数のOCRエンジンを使用する場合に効果的です。現在Tesseractのみ使用の場合は、効果が限定的です。

## 🔄 元に戻す方法

改善機能を有効化した後、精度が低下した場合は、`config.yaml`で無効化します：

```yaml
timestamp:
  extraction:
    use_improved_validator: false
    use_weighted_consensus: false
    use_voting_consensus: false
```

設定を変更後、再度実行すると既存実装に戻ります。

## 📝 実装状況

- ✅ TemporalValidatorV2: 統合済み（デフォルト: 無効）
- ✅ 重み付けスキーム: 統合済み（デフォルト: 無効）
- ✅ 投票ロジック: 統合済み（デフォルト: 無効）
- ✅ 設定ファイル対応: 完了
- ✅ テストツール: 作成済み

## 🎯 次のステップ

1. **精度テストの実行**: `tools/test_improved_features.py`を実行
2. **結果の確認**: 改善が確認された機能を特定
3. **段階的な有効化**: 改善が確認された機能から順に有効化
4. **本番データでの検証**: 実際の動画データで精度を確認

