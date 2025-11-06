"""Output file management with session-based organization."""

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OutputManager:
    """出力ファイル管理クラス（セッションベース）"""

    def __init__(self, output_base: Path):
        """OutputManagerを初期化

        Args:
            output_base: 出力ベースディレクトリ（例: output/）
        """
        self.output_base = Path(output_base)
        self.sessions_dir = self.output_base / "sessions"
        self.latest_link = self.output_base / "latest"
        self.archive_dir = self.output_base / "archive"
        self.shared_dir = self.output_base / "shared"

        # ディレクトリを作成
        self._setup_directories()

    def _setup_directories(self) -> None:
        """必要なディレクトリを作成（最小限のみ）"""
        # セッション管理に必要なディレクトリのみ作成
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        # Ground Truthデータ用（既存のlabelsディレクトリをsharedに移動する場合は使用）
        (self.shared_dir / "labels").mkdir(exist_ok=True)

    def create_session(self) -> Path:
        """新しい実行セッション用のディレクトリを作成

        Returns:
            セッションディレクトリのパス
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.sessions_dir / session_id

        # フェーズ別ディレクトリを作成
        phase_dirs = [
            "phase1_extraction/frames",
            "phase2_detection/images",
            "phase3_transform",
            "phase4_aggregation",
            "phase5_visualization/graphs",
            "phase5_visualization/floormaps",
        ]

        for phase_dir in phase_dirs:
            (session_dir / phase_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"セッションディレクトリを作成しました: {session_dir}")
        return session_dir

    def save_metadata(
        self, session_dir: Path, config: Dict, args: Optional[Dict] = None
    ) -> None:
        """セッションメタデータを保存

        Args:
            session_dir: セッションディレクトリ
            config: 設定情報
            args: コマンドライン引数（オプション）
        """
        metadata = {
            "session_id": session_dir.name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "arguments": args or {},
        }

        metadata_path = session_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.debug(f"メタデータを保存しました: {metadata_path}")

    def save_summary(self, session_dir: Path, summary: Dict) -> None:
        """実行サマリーを保存

        Args:
            session_dir: セッションディレクトリ
            summary: サマリーデータ
        """
        summary_data = {
            "session_id": session_dir.name,
            "timestamp": datetime.now().isoformat(),
            "status": summary.get("status", "completed"),
            **summary,
        }

        summary_path = session_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

        logger.debug(f"サマリーを保存しました: {summary_path}")

    def update_latest_link(self, session_dir: Path) -> None:
        """最新セッションへのシンボリックリンクを更新

        Args:
            session_dir: セッションディレクトリ
        """
        try:
            if self.latest_link.exists():
                if self.latest_link.is_symlink():
                    self.latest_link.unlink()
                else:
                    # シンボリックリンクでない場合は削除
                    self.latest_link.unlink()

            # 相対パスでシンボリックリンクを作成
            relative_path = session_dir.relative_to(self.output_base)
            self.latest_link.symlink_to(relative_path)

            logger.debug(f"最新セッションリンクを更新しました: {self.latest_link} -> {session_dir}")
        except OSError as e:
            # Windowsなど、シンボリックリンクが使えない環境ではスキップ
            logger.warning(f"シンボリックリンクの作成に失敗しました: {e}")

    def get_latest_session(self) -> Optional[Path]:
        """最新のセッションを取得

        Returns:
            最新セッションディレクトリ（存在しない場合はNone）
        """
        if self.latest_link.exists() and self.latest_link.is_symlink():
            resolved = self.latest_link.resolve()
            if resolved.exists():
                return resolved

        # フォールバック: セッション一覧から最新を取得
        sessions = self.find_sessions()
        return sessions[0] if sessions else None

    def find_sessions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        pattern: Optional[str] = None,
    ) -> List[Path]:
        """条件に一致するセッションを検索

        Args:
            start_date: 開始日時（この日時以降のセッション）
            end_date: 終了日時（この日時以前のセッション）
            pattern: セッションIDに含まれるパターン

        Returns:
            セッションディレクトリのリスト（新しい順）
        """
        sessions = []

        if not self.sessions_dir.exists():
            return sessions

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            # パターンフィルタ
            if pattern and pattern not in session_dir.name:
                continue

            # 日時フィルタ
            if start_date or end_date:
                try:
                    session_date = datetime.strptime(session_dir.name, "%Y%m%d_%H%M%S")
                    if start_date and session_date < start_date:
                        continue
                    if end_date and session_date > end_date:
                        continue
                except ValueError:
                    # 日時パースに失敗した場合はスキップ
                    continue

            sessions.append(session_dir)

        # 新しい順にソート
        return sorted(sessions, key=lambda p: p.name, reverse=True)

    def archive_old_sessions(self, days: int = 30) -> int:
        """古いセッションをアーカイブ

        Args:
            days: アーカイブ対象の日数（この日数以上古いセッション）

        Returns:
            アーカイブしたセッション数
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        archived_count = 0

        for session_dir in self.find_sessions(end_date=cutoff_date):
            try:
                archive_path = self.archive_dir / session_dir.name
                if archive_path.exists():
                    # 既にアーカイブされている場合はスキップ
                    continue

                shutil.move(str(session_dir), str(archive_path))
                archived_count += 1
                logger.info(f"セッションをアーカイブしました: {session_dir.name}")
            except Exception as e:
                logger.error(f"アーカイブに失敗しました ({session_dir.name}): {e}")

        return archived_count

    def get_session_size(self, session_dir: Path) -> int:
        """セッションのディスク使用量を取得（バイト）

        Args:
            session_dir: セッションディレクトリ

        Returns:
            ディスク使用量（バイト）
        """
        total = 0
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                try:
                    total += file_path.stat().st_size
                except OSError:
                    # ファイルアクセスエラーは無視
                    pass
        return total

    def cleanup_old_archives(self, days: int = 90) -> int:
        """古いアーカイブを削除

        Args:
            days: 削除対象の日数（この日数以上古いアーカイブ）

        Returns:
            削除したアーカイブ数
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0

        if not self.archive_dir.exists():
            return 0

        for archive_dir in self.archive_dir.iterdir():
            if not archive_dir.is_dir():
                continue

            try:
                archive_date = datetime.strptime(archive_dir.name, "%Y%m%d_%H%M%S")
                if archive_date < cutoff_date:
                    shutil.rmtree(archive_dir)
                    deleted_count += 1
                    logger.info(f"アーカイブを削除しました: {archive_dir.name}")
            except (ValueError, OSError) as e:
                logger.warning(f"アーカイブの削除に失敗しました ({archive_dir.name}): {e}")

        return deleted_count


def format_file_size(size_bytes: int) -> str:
    """ファイルサイズを人間が読める形式に変換

    Args:
        size_bytes: バイト数

    Returns:
        フォーマットされた文字列（例: "1.5 MB"）
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
