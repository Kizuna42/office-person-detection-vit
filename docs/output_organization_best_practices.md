# 出力ファイル整理のベストプラクティス

## 📋 概要

本ドキュメントでは、オフィス人物検出システムにおける出力ファイルの整理方法とベストプラクティスを定義します。

---

## 🎯 設計原則

### 1. **実行セッション管理**

- 各実行を独立したセッションとして管理
- タイムスタンプベースのディレクトリ構造
- 実行履歴の追跡が可能

### 2. **階層的なディレクトリ構造**

- フェーズ別に整理
- データタイプ別に分類
- 検索・アクセスが容易

### 3. **一貫した命名規則**

- 明確で予測可能なファイル名
- タイムスタンプを含む
- バージョン管理が容易

### 4. **クリーンアップとアーカイブ**

- 古い出力の自動整理
- 重要な結果の保護
- ストレージ効率の最適化

---

## 📁 推奨ディレクトリ構造

```
output/
├── sessions/                          # 実行セッション管理
│   └── 20251107_023001/              # 実行日時 (YYYYMMDD_HHMMSS)
│       ├── metadata.json              # 実行メタデータ（設定、パラメータなど）
│       ├── summary.json               # 実行サマリー（統計情報）
│       │
│       ├── phase1_extraction/         # フェーズ1: フレーム抽出
│       │   ├── frames/                # 抽出フレーム画像
│       │   │   ├── frame_20250826_160500_idx4.jpg
│       │   │   └── ...
│       │   ├── extraction_results.csv
│       │   └── timestamp_analysis.json # タイムスタンプ分析結果
│       │
│       ├── phase2_detection/          # フェーズ2: 人物検出
│       │   ├── images/                # 検出結果画像（オプション）
│       │   │   └── detection_*.jpg
│       │   ├── detection_results.json # 検出結果（詳細）
│       │   └── detection_statistics.json
│       │
│       ├── phase3_transform/          # フェーズ3: 座標変換
│       │   └── coordinate_transformations.json
│       │
│       ├── phase4_aggregation/        # フェーズ4: 集計
│       │   ├── zone_counts.csv
│       │   └── aggregation_summary.json
│       │
│       ├── phase5_visualization/      # フェーズ5: 可視化
│       │   ├── graphs/
│       │   │   ├── time_series.png
│       │   │   ├── statistics.png
│       │   │   └── heatmap.png
│       │   └── floormaps/
│       │       ├── floormap_20250826_160456.png
│       │       └── legend.png
│       │
│       └── logs/                      # セッション固有のログ
│           └── session.log
│
├── latest/                            # 最新実行結果へのシンボリックリンク
│   └── -> sessions/20251107_023001/
│
├── archive/                           # アーカイブ（30日以上古いセッション）
│   └── 20251001_120000/
│
├── shared/                            # 共有リソース（全セッション共通）
│   ├── labels/                       # Ground Truthデータ
│   │   └── result_fixed.json
│   └── templates/                    # テンプレート（将来拡張用）
│
└── system.log                        # システム全体のログ
```

---

## 📝 ファイル命名規則

### 基本ルール

1. **タイムスタンプ形式**: `YYYYMMDD_HHMMSS` または `YYYYMMDDHHMMSS`
2. **セマンティックな名前**: 内容が分かる名前
3. **拡張子**: データタイプに応じた拡張子（`.json`, `.csv`, `.png`, `.jpg`）

### 命名例

#### フレーム画像

```
frame_20250826_160500_idx4.jpg
```

- `frame_`: プレフィックス
- `20250826_160500`: タイムスタンプ（YYYYMMDD_HHMMSS）
- `idx4`: フレームインデックス
- `.jpg`: 拡張子

#### 検出結果画像

```
detection_20250826_160500_frame4.jpg
```

#### フロアマップ可視化

```
floormap_20250826_160456.png
```

#### データファイル

```
extraction_results.csv
detection_statistics.json
zone_counts.csv
```

---

## 🔧 実装ガイドライン

### 1. セッション管理

各実行時に一意のセッション ID を生成：

```python
from datetime import datetime
from pathlib import Path

def create_session_directory(output_base: Path) -> Path:
    """実行セッション用のディレクトリを作成"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_base / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir
```

### 2. メタデータの保存

実行時の設定とパラメータを保存：

```python
def save_session_metadata(session_dir: Path, config: dict, args: dict):
    """セッションメタデータを保存"""
    metadata = {
        "session_id": session_dir.name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "arguments": args,
        "version": get_version(),  # プロジェクトバージョン
    }

    metadata_path = session_dir / "metadata.json"
    with metadata_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
```

### 3. サマリーの生成

実行結果のサマリーを自動生成：

```python
def save_session_summary(session_dir: Path, results: dict):
    """実行サマリーを保存"""
    summary = {
        "session_id": session_dir.name,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "phases": {
            "extraction": {
                "frames_extracted": results.get("extraction_count", 0),
                "success_rate": results.get("extraction_success_rate", 0.0),
            },
            "detection": {
                "total_detections": results.get("total_detections", 0),
                "avg_per_frame": results.get("avg_detections", 0.0),
            },
            # ... 他のフェーズ
        },
        "execution_time_seconds": results.get("execution_time", 0.0),
    }

    summary_path = session_dir / "summary.json"
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
```

### 4. 最新セッションへのリンク

```python
def update_latest_link(output_base: Path, session_dir: Path):
    """最新セッションへのシンボリックリンクを更新"""
    latest_dir = output_base / "latest"
    if latest_dir.exists():
        latest_dir.unlink()
    latest_dir.symlink_to(session_dir.relative_to(output_base))
```

---

## 🗂️ フェーズ別出力構造

### Phase 1: フレーム抽出

```
phase1_extraction/
├── frames/                           # 抽出フレーム画像
│   └── frame_*.jpg
├── extraction_results.csv            # 抽出結果（CSV）
└── timestamp_analysis.json          # タイムスタンプ分析（オプション）
```

### Phase 2: 人物検出

```
phase2_detection/
├── images/                           # 検出結果画像（オプション）
│   └── detection_*.jpg
├── detection_results.json           # 検出結果（詳細、全検出データ）
└── detection_statistics.json        # 検出統計（集計情報）
```

### Phase 3: 座標変換

```
phase3_transform/
└── coordinate_transformations.json  # 座標変換結果
```

### Phase 4: 集計

```
phase4_aggregation/
├── zone_counts.csv                  # ゾーン別人数集計
└── aggregation_summary.json        # 集計サマリー
```

### Phase 5: 可視化

```
phase5_visualization/
├── graphs/
│   ├── time_series.png
│   ├── statistics.png
│   └── heatmap.png
└── floormaps/
    ├── floormap_*.png
    └── legend.png
```

---

## 🧹 クリーンアップとアーカイブ

### 自動アーカイブ

30 日以上古いセッションを自動的にアーカイブ：

```python
def archive_old_sessions(output_base: Path, days: int = 30):
    """古いセッションをアーカイブ"""
    sessions_dir = output_base / "sessions"
    archive_dir = output_base / "archive"
    archive_dir.mkdir(exist_ok=True)

    cutoff_date = datetime.now() - timedelta(days=days)

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        # セッションIDから日時を抽出
        try:
            session_date = datetime.strptime(session_dir.name, "%Y%m%d_%H%M%S")
            if session_date < cutoff_date:
                # アーカイブに移動
                archive_path = archive_dir / session_dir.name
                session_dir.rename(archive_path)
        except ValueError:
            continue
```

### 手動クリーンアップ

```bash
# 30日以上古いセッションをアーカイブ
python -m src.utils.cleanup --days 30

# アーカイブを完全削除
python -m src.utils.cleanup --archive --delete

# 特定のセッションを削除
python -m src.utils.cleanup --session 20251107_023001 --delete
```

---

## 📊 データファイルの形式

### JSON ファイル

- **インデント**: 2 スペース
- **エンコーディング**: UTF-8
- **日時形式**: ISO 8601 (`YYYY-MM-DDTHH:MM:SS`)
- **数値**: 浮動小数点は適切な精度で保存

```json
{
  "session_id": "20251107_023001",
  "timestamp": "2025-11-07T02:30:01",
  "data": {
    "value": 21.25,
    "count": 85
  }
}
```

### CSV ファイル

- **エンコーディング**: UTF-8
- **ヘッダー**: 必須
- **区切り文字**: カンマ（`,`）
- **日時形式**: `YYYY/MM/DD HH:MM:SS`

```csv
timestamp,zone_id,count
2025/08/26 16:04:56,zone_2,12
2025/08/26 16:04:56,zone_1,3
```

---

## 🔍 検索とアクセス

### セッション検索

```python
def find_sessions(
    output_base: Path,
    start_date: datetime = None,
    end_date: datetime = None,
    pattern: str = None
) -> List[Path]:
    """条件に一致するセッションを検索"""
    sessions_dir = output_base / "sessions"
    sessions = []

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        # 日時フィルタ
        if start_date or end_date:
            session_date = datetime.strptime(session_dir.name, "%Y%m%d_%H%M%S")
            if start_date and session_date < start_date:
                continue
            if end_date and session_date > end_date:
                continue

        # パターンフィルタ
        if pattern and pattern not in session_dir.name:
            continue

        sessions.append(session_dir)

    return sorted(sessions, reverse=True)  # 新しい順
```

### 最新セッションの取得

```python
def get_latest_session(output_base: Path) -> Path:
    """最新のセッションを取得"""
    latest_link = output_base / "latest"
    if latest_link.is_symlink():
        return latest_link.resolve()

    # フォールバック: セッション一覧から最新を取得
    sessions = find_sessions(output_base)
    return sessions[0] if sessions else None
```

---

## 📈 パフォーマンス考慮事項

### 1. 大容量ファイルの管理

- 検出結果画像はオプション（設定で制御）
- 圧縮形式の使用（PNG → JPEG、品質調整）
- 不要な中間ファイルの削除

### 2. ディスク使用量の監視

```python
def get_session_size(session_dir: Path) -> int:
    """セッションのディスク使用量を取得（バイト）"""
    total = 0
    for file_path in session_dir.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total
```

### 3. 並列実行時の競合回避

- セッション ID にマイクロ秒を含める
- ファイルロックの使用
- 一時ファイルの適切な管理

---

## 🚀 移行ガイド

### 既存出力の移行

1. **現状の確認**

   ```bash
   # 既存の出力ファイルを確認
   ls -la output/
   ```

2. **移行スクリプトの実行**

   ```python
   # 既存の出力を新しい構造に移行
   python -m src.utils.migrate_output --source output --target output/sessions/legacy
   ```

3. **検証**
   - ファイルの整合性確認
   - パスの更新
   - テスト実行

---

## ✅ チェックリスト

### 実装時

- [ ] セッションディレクトリの作成
- [ ] メタデータの保存
- [ ] サマリーの生成
- [ ] 最新セッションへのリンク
- [ ] フェーズ別ディレクトリ構造
- [ ] ファイル命名規則の統一
- [ ] クリーンアップ機能の実装

### 運用時

- [ ] 定期的なアーカイブ実行
- [ ] ディスク使用量の監視
- [ ] 古いセッションの削除
- [ ] バックアップ戦略の確立

---

## 📚 参考実装

実装例は `src/utils/output_manager.py` を参照してください。
