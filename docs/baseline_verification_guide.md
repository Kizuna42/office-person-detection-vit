# ベースライン精度定量化システム - 動作確認ガイド

このドキュメントでは、実装したベースライン精度定量化システムの各セクションについて、ユーザーが何をすべきか、どのように動作確認するか、どのようなアウトプットが出れば完了かを説明します。

## 全体の流れ

```
1. データ準備（オプション）
   ↓
2. パイプライン実行
   ↓
3. 評価実行
   ↓
4. レポート生成
   ↓
5. 結果確認
```

---

## セクション 1: ベースライン実行スクリプト (`scripts/run_baseline.py`)

### ユーザーがすること

1. **設定ファイルの確認**

   - `config.yaml`の`output.use_session_management`が`true`になっていることを確認
   - 動画ファイルパス（`video.input_path`）が正しいことを確認

2. **スクリプトの実行**
   ```bash
   python scripts/run_baseline.py --config config.yaml --tag baseline-20250115
   ```

### 動作確認

#### ✅ 正常な動作

1. **ログ出力の確認**

   ```
   ================================================================================
   ベースライン実行を開始
   ================================================================================
   パイプラインを実行中: python main.py --config config.yaml
   パイプライン実行が完了しました
   セッションIDを取得しました: 20250115_143022
   ================================================================================
   ベースライン実行が完了しました
   セッションID: 20250115_143022
   ================================================================================
   ```

2. **出力ファイルの確認**

   ```bash
   # セッションディレクトリが作成されている
   ls output/sessions/20250115_143022/

   # 以下のファイルが存在する
   - metadata.json
   - baseline_info.json
   - phase1_extraction/
   - phase2_detection/
   - phase2.5_tracking/
   - phase3_transform/
   - phase4_aggregation/
   - phase5_visualization/
   - summary.json
   ```

3. **baseline_info.json の内容確認**

   ```bash
   cat output/sessions/20250115_143022/baseline_info.json
   ```

   期待される内容：

   ```json
   {
     "session_id": "20250115_143022",
     "timestamp": "2025-01-15T14:30:22.123456",
     "config_path": "config.yaml",
     "tag": "baseline-20250115",
     "pipeline_executed": true,
     "session_management_enabled": true
   }
   ```

#### ❌ エラーケース

- **セッション管理が無効**: 警告メッセージが表示されるが、処理は続行される
- **動画ファイルが見つからない**: エラーメッセージが表示され、処理が中断される

### 完了基準

- ✅ スクリプトが正常終了（終了コード 0）
- ✅ セッションディレクトリが作成される
- ✅ `baseline_info.json`が存在し、正しい内容が記録されている
- ✅ パイプラインの各フェーズが実行され、出力ファイルが生成されている

---

## セクション 2: 評価統合スクリプト (`scripts/evaluate_baseline.py`)

### ユーザーがすること

1. **前ステップで取得したセッション ID を確認**

   ```bash
   # 最新セッションを確認
   ls -lt output/sessions/ | head -2
   ```

2. **データファイルの準備（オプション）**

   - Ground Truth トラックファイル（`data/gt_tracks.json`）がある場合
   - 対応点ファイル（`data/correspondence_points_cam01.json`）がある場合

3. **スクリプトの実行**

   ```bash
   # パフォーマンスのみ評価（データファイルなし）
   python scripts/evaluate_baseline.py --session 20250115_143022 --config config.yaml

   # GTトラックと対応点ありの場合
   python scripts/evaluate_baseline.py \
       --session 20250115_143022 \
       --config config.yaml \
       --gt data/gt_tracks.json \
       --points data/correspondence_points_cam01.json
   ```

### 動作確認

#### ✅ 正常な動作

1. **ログ出力の確認**

   ```
   ================================================================================
   ベースライン評価を開始
   セッションID: 20250115_143022
   ================================================================================
   パフォーマンスを評価中...
   パフォーマンス評価が完了しました
   ================================================================================
   ベースライン評価が完了しました
   ================================================================================
   評価結果サマリー:
     処理時間/フレーム: 1.85 秒 (目標: 2.0 秒)
     メモリ増加: 8192 MB (目標: 12288 MB)
   ```

2. **出力ファイルの確認**

   ```bash
   # baseline_metrics.jsonが作成されている
   ls output/sessions/20250115_143022/baseline_metrics.json

   # データがある場合、個別の評価結果も確認
   ls output/sessions/20250115_143022/mot_metrics.json          # GTがある場合
   ls output/sessions/20250115_143022/reprojection_error.json    # 対応点がある場合
   ls output/sessions/20250115_143022/performance_metrics.json  # 常に生成
   ```

3. **baseline_metrics.json の内容確認**

   ```bash
   cat output/sessions/20250115_143022/baseline_metrics.json | jq .
   ```

   期待される内容（パフォーマンスのみの場合）：

   ```json
   {
     "session_id": "20250115_143022",
     "timestamp": "2025-01-15T14:35:00.123456",
     "pipeline": {
       "num_frames": 100,
       "total_time_seconds": 185.0
     },
     "mot_metrics": {
       "available": false
     },
     "reprojection_error": {
       "available": false
     },
     "performance": {
       "time_per_frame_seconds": 1.85,
       "total_time_seconds": 185.0,
       "memory_peak_mb": 8192.0,
       "memory_increase_mb": 4096.0,
       "num_frames": 100,
       "available": true
     },
     "targets": {
       "MOTA": 0.7,
       "IDF1": 0.8,
       "mean_error": 2.0,
       "max_error": 4.0,
       "time_per_frame": 2.0,
       "memory_mb": 12288
     },
     "achieved": {
       "MOTA": false,
       "IDF1": false,
       "mean_error": false,
       "max_error": false,
       "time_per_frame": true,
       "memory": true
     }
   }
   ```

#### ❌ エラーケース

- **セッション ID が存在しない**: `FileNotFoundError`が発生
- **GT ファイルが存在しない**: MOT 評価をスキップし、警告メッセージを表示
- **対応点ファイルが存在しない**: 再投影誤差評価をスキップし、警告メッセージを表示

### 完了基準

- ✅ スクリプトが正常終了（終了コード 0）
- ✅ `baseline_metrics.json`が生成される
- ✅ パフォーマンス評価が実行され、`performance_metrics.json`が生成される
- ✅ データファイルがある場合、MOT 評価または再投影誤差評価が実行される
- ✅ データファイルがない場合、適切な警告メッセージが表示される

---

## セクション 3: レポート生成スクリプト (`scripts/generate_baseline_report.py`)

### ユーザーがすること

1. **前ステップで評価が完了していることを確認**

   ```bash
   ls output/sessions/20250115_143022/baseline_metrics.json
   ```

2. **スクリプトの実行**
   ```bash
   python scripts/generate_baseline_report.py \
       --session 20250115_143022 \
       --config config.yaml
   ```

### 動作確認

#### ✅ 正常な動作

1. **ログ出力の確認**

   ```
   ================================================================================
   ベースラインレポートを生成中
   セッションID: 20250115_143022
   ================================================================================
   ベースラインレポートを保存しました: output/sessions/20250115_143022/baseline_report.md
   ================================================================================
   ベースラインレポート生成が完了しました
   ================================================================================
   ```

2. **出力ファイルの確認**

   ```bash
   # baseline_report.mdが作成されている
   ls output/sessions/20250115_143022/baseline_report.md

   # レポートの内容を確認
   cat output/sessions/20250115_143022/baseline_report.md
   ```

3. **レポートの内容確認**

   レポートには以下のセクションが含まれる：

   - **実行概要**: 処理フレーム数、総処理時間
   - **MOT メトリクス**: MOTA、IDF1、ID Switches（データがある場合）
   - **再投影誤差**: 平均誤差、最大誤差（データがある場合）
   - **パフォーマンス**: 処理時間/フレーム、メモリ使用量、フェーズ別処理時間
   - **達成状況サマリー**: 各目標値の達成状況
   - **推奨アクション**: 未達成項目がある場合の推奨事項
   - **関連ファイル**: 生成されたファイルの一覧

   例（パフォーマンスのみの場合）：

   ```markdown
   # ベースライン評価レポート

   **セッション ID**: `20250115_143022`
   **生成日時**: 2025-01-15 14:40:00

   ## 実行概要

   - **処理フレーム数**: 100
   - **総処理時間**: 185.00 秒

   ## MOT メトリクス

   **データなし**: Ground Truth トラックファイルが指定されていないか、存在しません。

   ## 再投影誤差

   **データなし**: 対応点ファイルが指定されていないか、存在しません。

   ## パフォーマンス

   ### 結果

   | メトリクス        | 値      | 目標値   | 達成状況 |
   | ----------------- | ------- | -------- | -------- |
   | 処理時間/フレーム | 1.85 秒 | 2.0 秒   | ✅ 達成  |
   | メモリ増加        | 4096 MB | 12288 MB | ✅ 達成  |

   ## 達成状況サマリー

   **全体**: ❌ 一部未達成

   | 項目           | 達成状況  |
   | -------------- | --------- |
   | MOTA           | ❌ 未達成 |
   | IDF1           | ❌ 未達成 |
   | mean_error     | ❌ 未達成 |
   | max_error      | ❌ 未達成 |
   | time_per_frame | ✅ 達成   |
   | memory         | ✅ 達成   |
   ```

#### ❌ エラーケース

- **baseline_metrics.json が存在しない**: `FileNotFoundError`が発生
- **セッション ID が存在しない**: `FileNotFoundError`が発生

### 完了基準

- ✅ スクリプトが正常終了（終了コード 0）
- ✅ `baseline_report.md`が生成される
- ✅ レポートに必要なセクションがすべて含まれている
- ✅ 達成状況が正しく表示されている
- ✅ データがない場合でも、適切なメッセージが表示される

---

## セクション 4: データ準備ガイド (`docs/baseline_data_preparation.md`)

### ユーザーがすること

1. **ドキュメントの確認**

   ```bash
   cat docs/baseline_data_preparation.md
   ```

2. **必要に応じてデータを作成**
   - Ground Truth トラックファイル（MOT 評価用）
   - 対応点ファイル（再投影誤差評価用）

### 動作確認

#### ✅ 正常な動作

1. **ドキュメントが存在する**

   ```bash
   ls docs/baseline_data_preparation.md
   ```

2. **ドキュメントの内容確認**
   - Ground Truth トラックの作成手順が記載されている
   - 対応点データの作成手順が記載されている
   - データ形式の説明がある
   - テンプレートファイルへの参照がある

### 完了基準

- ✅ ドキュメントファイルが存在する
- ✅ 必要な情報がすべて記載されている
- ✅ テンプレートファイルへの参照が正しい

---

## セクション 5: データテンプレートファイル

### ユーザーがすること

1. **テンプレートファイルの確認**

   ```bash
   ls data/gt_tracks_template.json
   ls data/correspondence_points_cam01.json.template
   ```

2. **必要に応じてテンプレートをコピーして使用**

   ```bash
   # GTトラックファイルを作成する場合
   cp data/gt_tracks_template.json data/gt_tracks.json
   # 実際のデータで編集

   # 対応点ファイルを作成する場合
   cp data/correspondence_points_cam01.json.template data/correspondence_points_cam01.json
   # 実際のデータで編集
   ```

### 動作確認

#### ✅ 正常な動作

1. **テンプレートファイルが存在する**

   ```bash
   ls data/gt_tracks_template.json
   ls data/correspondence_points_cam01.json.template
   ```

2. **テンプレートファイルの内容確認**

   ```bash
   # JSON形式が正しいか確認
   cat data/gt_tracks_template.json | jq .
   cat data/correspondence_points_cam01.json.template | jq .
   ```

   期待される内容：

   - `gt_tracks_template.json`: `tracks`配列と`metadata`が含まれる
   - `correspondence_points_cam01.json.template`: `src_points`と`dst_points`が含まれる

### 完了基準

- ✅ テンプレートファイルが存在する
- ✅ JSON 形式が正しい（`jq`でパースできる）
- ✅ 必要なフィールドがすべて含まれている
- ✅ コメントや説明が適切に記載されている

---

## セクション 6: 統合実行スクリプト (`scripts/run_full_baseline.py`)

### ユーザーがすること

1. **設定ファイルの確認**

   - `config.yaml`の設定が正しいことを確認

2. **データファイルの準備（オプション）**

   - Ground Truth トラックファイル（`data/gt_tracks.json`）
   - 対応点ファイル（`data/correspondence_points_cam01.json`）

3. **スクリプトの実行**

   ```bash
   # 全ステップを一括実行
   python scripts/run_full_baseline.py \
       --config config.yaml \
       --tag baseline-20250115 \
       --gt data/gt_tracks.json \
       --points data/correspondence_points_cam01.json

   # データファイルなしの場合
   python scripts/run_full_baseline.py \
       --config config.yaml \
       --tag baseline-20250115
   ```

### 動作確認

#### ✅ 正常な動作

1. **ログ出力の確認**

   ```
   ================================================================================
   ベースライン統合実行を開始
   ================================================================================
   ================================================================================
   ステップ1: パイプライン実行
   ================================================================================
   パイプライン実行が完了しました
   セッションIDを取得しました: 20250115_143022
   ================================================================================
   ステップ2: 評価実行
   ================================================================================
   評価実行が完了しました
   ================================================================================
   ステップ3: レポート生成
   ================================================================================
   レポート生成が完了しました
   ================================================================================
   ベースライン統合実行が完了しました
   ================================================================================
   セッションID: 20250115_143022
   生成されたファイル:
     - 評価結果: output/sessions/20250115_143022/baseline_metrics.json
     - レポート: output/sessions/20250115_143022/baseline_report.md

   評価結果サマリー:
     ✅ すべての目標値を達成しました
   ```

2. **出力ファイルの確認**

   ```bash
   # すべてのファイルが生成されている
   ls output/sessions/20250115_143022/

   # 以下のファイルが存在する
   - baseline_info.json
   - baseline_metrics.json
   - baseline_report.md
   - performance_metrics.json
   - mot_metrics.json          # GTがある場合
   - reprojection_error.json   # 対応点がある場合
   ```

3. **既存セッションを使用する場合**
   ```bash
   # パイプライン実行をスキップして、既存セッションで評価のみ実行
   python scripts/run_full_baseline.py \
       --config config.yaml \
       --skip-pipeline \
       --session 20250115_143022 \
       --gt data/gt_tracks.json \
       --points data/correspondence_points_cam01.json
   ```

#### ❌ エラーケース

- **パイプライン実行に失敗**: エラーメッセージが表示され、処理が中断される
- **評価実行に失敗**: エラーメッセージが表示され、処理が中断される
- **レポート生成に失敗**: エラーメッセージが表示され、処理が中断される

### 完了基準

- ✅ スクリプトが正常終了（終了コード 0）
- ✅ 3 つのステップ（パイプライン実行、評価実行、レポート生成）がすべて実行される
- ✅ すべての出力ファイルが生成される
- ✅ 評価結果サマリーが表示される
- ✅ `--skip-pipeline`オプションが正しく動作する

---

## 全体の完了基準

すべてのセクションが完了したことを確認するには、以下を実行してください：

### 1. 統合テスト（推奨）

```bash
# データファイルなしで実行（パフォーマンスのみ評価）
python scripts/run_full_baseline.py \
    --config config.yaml \
    --tag baseline-test
```

**期待される結果**:

- ✅ パイプラインが正常に実行される
- ✅ パフォーマンス評価が実行される
- ✅ レポートが生成される
- ✅ すべてのファイルが`output/sessions/<session_id>/`に生成される

### 2. 個別テスト

```bash
# 各スクリプトを個別に実行して確認
python scripts/run_baseline.py --config config.yaml --tag test1
python scripts/evaluate_baseline.py --session <session_id> --config config.yaml
python scripts/generate_baseline_report.py --session <session_id> --config config.yaml
```

### 3. 出力ファイルの確認

```bash
SESSION_ID="20250115_143022"  # 実際のセッションIDに置き換え

# 必須ファイルの存在確認
ls output/sessions/$SESSION_ID/baseline_info.json
ls output/sessions/$SESSION_ID/baseline_metrics.json
ls output/sessions/$SESSION_ID/baseline_report.md
ls output/sessions/$SESSION_ID/performance_metrics.json

# JSONファイルの形式確認
cat output/sessions/$SESSION_ID/baseline_metrics.json | jq .
cat output/sessions/$SESSION_ID/performance_metrics.json | jq .

# レポートの内容確認
cat output/sessions/$SESSION_ID/baseline_report.md
```

### 4. 最終確認チェックリスト

- [ ] すべてのスクリプトが正常に実行できる
- [ ] エラーハンドリングが適切に動作する（存在しないセッション ID、ファイルパスなど）
- [ ] データファイルがない場合でも、パフォーマンス評価は実行される
- [ ] データファイルがある場合、MOT 評価と再投影誤差評価が実行される
- [ ] レポートに必要な情報がすべて含まれている
- [ ] 達成状況が正しく表示される
- [ ] テンプレートファイルが正しい形式である
- [ ] ドキュメントが分かりやすく記載されている

---

## トラブルシューティング

### よくあるエラーと対処法

1. **`ModuleNotFoundError: No module named 'src'`**

   - 解決策: プロジェクトルートから実行していることを確認
   - 確認: `pwd`で現在のディレクトリを確認

2. **`FileNotFoundError: セッションディレクトリが見つかりません`**

   - 解決策: セッション ID が正しいか確認、またはパイプライン実行を先に実行

3. **`subprocess.CalledProcessError`**

   - 解決策: 呼び出されるスクリプト（`main.py`、`evaluate_mot_metrics.py`など）が正常に動作するか確認

4. **評価結果が`available: false`になる**
   - 原因: データファイルが指定されていない、またはファイルが存在しない
   - 対処: データファイルを準備するか、パフォーマンス評価のみで完了とみなす

---

## 次のステップ

ベースライン評価が完了したら：

1. **結果の分析**: レポートを確認し、目標値を達成しているか確認
2. **パラメータ調整**: 未達成項目がある場合、`config.yaml`のパラメータを調整
3. **再評価**: パラメータ調整後、再度ベースライン評価を実行
4. **比較**: 複数のセッションの結果を比較して、最適なパラメータを決定
