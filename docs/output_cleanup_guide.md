# 出力ディレクトリ整理ガイド

## 📋 概要

セッション管理が有効化された後、`output/`ディレクトリを整理して、main.pyの実行結果のみが合理的に格納されるようにします。

---

## 🎯 整理後の理想的な構造

```
output/
├── sessions/                    # 実行セッション（main.pyの実行結果のみ）
│   └── YYYYMMDD_HHMMSS/        # 各実行セッション
│       ├── metadata.json
│       ├── summary.json
│       ├── phase1_extraction/
│       ├── phase2_detection/
│       ├── phase3_transform/
│       ├── phase4_aggregation/
│       ├── phase5_visualization/
│       └── logs/
├── latest -> sessions/...        # 最新セッションへのシンボリックリンク
├── archive/                     # アーカイブ（30日以上古いセッション）
├── shared/                      # 共有リソース（Ground Truthなど）
│   └── labels/
└── system.log                   # システム全体のログ
```

---

## 🧹 整理手順

### 1. 既存の余分なディレクトリを確認

```bash
# 現在の構造を確認
ls -la output/
```

### 2. 古い出力ファイルをアーカイブまたは削除

以下のファイル/ディレクトリは、セッション管理が有効化される前の古い出力です：

- `coordinate_transformations.json` → セッション内の`phase3_transform/`に移動済み
- `detection_statistics.json` → セッション内の`phase2_detection/`に移動済み
- `zone_counts.csv` → セッション内の`phase4_aggregation/`に移動済み
- `timestamp_extraction_*.csv` → セッション内の`phase1_extraction/`に移動済み
- `detections/` → セッション内の`phase2_detection/images/`に移動済み
- `floormaps/` → セッション内の`phase5_visualization/floormaps/`に移動済み
- `graphs/` → セッション内の`phase5_visualization/graphs/`に移動済み

### 3. 整理スクリプトの実行（オプション）

```bash
# 古いファイルを確認（削除は実行しない）
python tools/cleanup_output.py --list

# 30日以上古いセッションをアーカイブ
python tools/cleanup_output.py --archive --archive-days 30
```

### 4. 手動整理（推奨）

重要なデータを確認してから、以下のコマンドで整理：

```bash
# 1. 既存のセッションを確認
ls -la output/sessions/

# 2. 古い出力ファイルをバックアップ（必要に応じて）
mkdir -p output/backup_old_outputs
mv output/*.json output/backup_old_outputs/ 2>/dev/null
mv output/*.csv output/backup_old_outputs/ 2>/dev/null

# 3. 空のディレクトリを削除（セッション管理が有効な場合、不要）
# 注意: 既存のデータが重要でない場合のみ実行
rmdir output/detections output/floormaps output/graphs 2>/dev/null
```

---

## ✅ 整理後の確認

### セッション管理が有効な場合

```bash
# 出力ディレクトリの構造を確認
tree output/ -L 2 -d

# 期待される構造:
# output/
# ├── sessions/
# ├── latest -> sessions/...
# ├── archive/
# └── shared/
```

### セッション管理が無効な場合（後方互換性）

```bash
# 従来の構造が使用される
# output/
# ├── detections/
# ├── floormaps/
# ├── graphs/
# └── labels/
```

---

## 🔧 自動整理（将来実装）

定期的なクリーンアップを自動化：

```bash
# cronで定期実行（例: 毎日午前3時）
0 3 * * * cd /path/to/project && python tools/cleanup_output.py --archive
```

---

## 📝 注意事項

1. **既存データの保護**: 重要なデータは事前にバックアップ
2. **セッション管理の有効化**: `config.yaml`で`output.use_session_management: true`を確認
3. **後方互換性**: セッション管理を無効にすると、従来の構造が使用される

