# 出力ディレクトリ構造 - 最終設計

## 📋 概要

セッション管理が有効化された後の、合理的で無駄のない出力ディレクトリ構造を定義します。

---

## 🎯 最終的な構造（セッション管理有効時）

```
output/
├── sessions/                    # 実行セッション（main.pyの実行結果のみ）
│   └── YYYYMMDD_HHMMSS/        # 各実行セッション
│       ├── metadata.json        # 実行メタデータ
│       ├── summary.json         # 実行サマリー
│       ├── phase1_extraction/   # フェーズ1: フレーム抽出
│       │   ├── frames/          # 抽出フレーム画像
│       │   └── extraction_results.csv
│       ├── phase2_detection/    # フェーズ2: 人物検出
│       │   ├── images/          # 検出結果画像
│       │   ├── detection_results.json
│       │   └── detection_statistics.json
│       ├── phase3_transform/    # フェーズ3: 座標変換
│       │   └── coordinate_transformations.json
│       ├── phase4_aggregation/  # フェーズ4: 集計
│       │   └── zone_counts.csv
│       ├── phase5_visualization/ # フェーズ5: 可視化
│       │   ├── graphs/
│       │   │   ├── time_series.png
│       │   │   └── statistics.png
│       │   └── floormaps/
│       │       └── floormap_*.png
│
├── latest -> sessions/...        # 最新セッションへのシンボリックリンク
├── archive/                     # アーカイブ（30日以上古いセッション）
├── shared/                      # 共有リソース（全セッション共通）
│   └── labels/                  # Ground Truthデータ
│       └── result_fixed.json
└── system.log                   # システム全体のログ
```

---

## ✅ 作成されるディレクトリ（セッション管理有効時）

### ルートディレクトリ（output/）

**作成されるもの:**
- `sessions/` - 実行セッション管理
- `archive/` - アーカイブ
- `shared/` - 共有リソース
- `system.log` - システムログ

**作成されないもの（削除）:**
- `detections/` - セッション内の`phase2_detection/images/`に移動
- `floormaps/` - セッション内の`phase5_visualization/floormaps/`に移動
- `graphs/` - セッション内の`phase5_visualization/graphs/`に移動
- `labels/` - `shared/labels/`に移動（既存データがある場合）

**ルートに残らないファイル:**
- `coordinate_transformations.json` → `phase3_transform/`に移動
- `detection_statistics.json` → `phase2_detection/`に移動
- `zone_counts.csv` → `phase4_aggregation/`に移動
- `timestamp_extraction_*.csv` → `phase1_extraction/`に移動

---

## 🔧 実装の変更点

### main.py

1. **セッション管理が有効な場合**
   - `setup_output_directories()`を呼ばない
   - `OutputManager`が必要なディレクトリのみ作成
   - すべての出力はセッションディレクトリ内に保存

2. **セッション管理が無効な場合（後方互換性）**
   - `setup_output_directories()`を呼ぶ
   - 従来の構造（`detections/`, `floormaps/`, `graphs/`, `labels/`）を使用

### OutputManager

- `_setup_directories()`で最小限のディレクトリのみ作成
- セッション管理に必要なディレクトリ（`sessions/`, `archive/`, `shared/`）のみ

---

## 🧹 既存の古い出力の整理

### 整理ツールの使用

```bash
# DRY RUNモード（削除せずに確認のみ）
python tools/cleanup_old_outputs.py --output-dir output

# 実際に削除を実行
python tools/cleanup_old_outputs.py --output-dir output --execute
```

### 手動整理

```bash
# 1. バックアップ（必要に応じて）
mkdir -p output/backup_old
mv output/*.json output/backup_old/ 2>/dev/null
mv output/*.csv output/backup_old/ 2>/dev/null

# 2. 空のディレクトリを削除
rmdir output/detections output/floormaps output/graphs 2>/dev/null

# 3. 内容があるディレクトリは手動で確認
ls -la output/detections/
# 必要に応じてバックアップしてから削除
```

---

## 📊 ディレクトリ数の比較

### 改善前（セッション管理無効時）

```
output/
├── detections/          # 48ファイル
├── floormaps/           # 多数のファイル
├── graphs/              # 3ファイル
├── labels/              # 1ファイル
├── *.json               # 2ファイル（ルート）
├── *.csv                # 2ファイル（ルート）
└── system.log
```

**問題点:**
- ルートにファイルが散在
- 実行ごとの区別ができない
- 古いファイルと新しいファイルが混在

### 改善後（セッション管理有効時）

```
output/
├── sessions/
│   └── YYYYMMDD_HHMMSS/  # 各実行が独立
│       └── phase*/        # フェーズ別に整理
├── latest -> sessions/...
├── archive/
├── shared/
└── system.log
```

**利点:**
- 実行ごとに独立したセッション
- フェーズ別に整理
- ルートにファイルが散在しない
- 履歴管理が容易

---

## ✅ 確認事項

### セッション管理が有効な場合

- [x] `output/sessions/`のみが作成される
- [x] `output/detections/`, `output/floormaps/`, `output/graphs/`は作成されない
- [x] すべての出力はセッションディレクトリ内に保存される
- [x] ルートにJSON/CSVファイルが散在しない

### セッション管理が無効な場合（後方互換性）

- [x] 従来の構造（`detections/`, `floormaps/`, `graphs/`, `labels/`）が使用される
- [x] 既存のコードとの互換性が保たれる

---

## 🚀 使用方法

### 新しい実行

```bash
# セッション管理が有効な場合（config.yamlで設定）
python main.py

# 出力先: output/sessions/YYYYMMDD_HHMMSS/
```

### 古い出力の整理

```bash
# 確認のみ
python tools/cleanup_old_outputs.py

# 実際に削除
python tools/cleanup_old_outputs.py --execute
```

---

## 📝 まとめ

- **セッション管理有効時**: 最小限のディレクトリ構造、実行ごとに独立したセッション
- **セッション管理無効時**: 従来の構造を維持（後方互換性）
- **整理ツール**: 既存の古い出力を安全に整理可能

---

## 📚 参考

- **実装コード**: `src/utils/output_manager.py`
- **クリーンアップツール**: `tools/cleanup_output.py`, `tools/cleanup_old_outputs.py`
- **設定**: `config.yaml` の `output.use_session_management`

