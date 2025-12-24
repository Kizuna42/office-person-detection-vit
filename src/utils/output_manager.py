"""Output file management with session-based organization."""

from contextlib import suppress
from datetime import datetime
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


def get_reproducibility_info() -> dict[str, Any]:
    """再現性のための環境情報を取得

    Returns:
        環境情報の辞書
    """
    info: dict[str, Any] = {
        "python_version": sys.version,
        "python_version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }

    # Git情報を取得（失敗した場合はスキップ）
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        info["git_commit"] = git_commit

        git_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        info["git_branch"] = git_branch

        # ワーキングディレクトリがクリーンかどうか
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        info["git_dirty"] = len(git_status) > 0

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        # Gitが利用できない場合はスキップ
        info["git_commit"] = None
        info["git_branch"] = None
        info["git_dirty"] = None

    return info


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

        # ディレクトリを作成
        self._setup_directories()

    def _setup_directories(self) -> None:
        """必要なディレクトリを作成（最小限のみ）"""
        # セッション管理に必要なディレクトリのみ作成
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> Path:
        """新しい実行セッション用のディレクトリを作成

        Returns:
            セッションディレクトリのパス
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.sessions_dir / session_id

        # フェーズ別ディレクトリを作成（番号プレフィックス形式）
        phase_dirs = [
            "01_extraction/frames",
            "02_detection/images",
            "03_tracking",
            "04_transform",
            "05_aggregation",
            "06_visualization/graphs",
            "06_visualization/floormaps",
        ]

        for phase_dir in phase_dirs:
            (session_dir / phase_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"セッションディレクトリを作成しました: {session_dir}")
        return session_dir

    def save_metadata(self, session_dir: Path, config: dict, args: dict | None = None) -> None:
        """セッションメタデータを保存

        Args:
            session_dir: セッションディレクトリ
            config: 設定情報
            args: コマンドライン引数（オプション）
        """
        metadata = {
            "session_id": session_dir.name,
            "timestamp": datetime.now().isoformat(),
            "reproducibility": get_reproducibility_info(),
            "config": config,
            "arguments": args or {},
        }

        metadata_path = session_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.debug(f"メタデータを保存しました: {metadata_path}")

    def save_summary(self, session_dir: Path, summary: dict) -> None:
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
            # 既存のリンクまたはディレクトリを削除
            if self.latest_link.exists():
                if self.latest_link.is_symlink():
                    self.latest_link.unlink()
                elif self.latest_link.is_dir():
                    # ディレクトリの場合は削除できないので警告を出してスキップ
                    logger.warning(
                        f"output/latestがディレクトリとして存在します。シンボリックリンクを作成できません: {self.latest_link}"
                    )
                    return
                else:
                    # ファイルの場合は削除
                    self.latest_link.unlink()

            # 相対パスでシンボリックリンクを作成
            relative_path = session_dir.relative_to(self.output_base)
            self.latest_link.symlink_to(relative_path)

            logger.info(f"最新セッションリンクを更新しました: {self.latest_link} -> {session_dir}")
        except OSError as e:
            # Windowsなど、シンボリックリンクが使えない環境ではスキップ
            logger.warning(f"シンボリックリンクの作成に失敗しました: {e}")
        except ValueError as e:
            # 相対パスの計算に失敗した場合
            logger.warning(f"シンボリックリンクの作成に失敗しました（相対パス計算エラー）: {e}")

    def get_latest_session(self) -> Path | None:
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
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        pattern: str | None = None,
    ) -> list[Path]:
        """条件に一致するセッションを検索

        Args:
            start_date: 開始日時（この日時以降のセッション）
            end_date: 終了日時（この日時以前のセッション）
            pattern: セッションIDに含まれるパターン

        Returns:
            セッションディレクトリのリスト（新しい順）
        """
        sessions: list[Path] = []

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
                with suppress(OSError):
                    total += file_path.stat().st_size
        return total


def format_file_size(size_bytes: int) -> str:
    """ファイルサイズを人間が読める形式に変換

    Args:
        size_bytes: バイト数

    Returns:
        フォーマットされた文字列（例: "1.5 MB"）
    """
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size = size / 1024.0
    return f"{size:.2f} PB"


def setup_output_directories(output_dir: Path) -> None:
    """出力ディレクトリを作成（後方互換性のため保持）

    セッション管理が無効な場合に使用される従来のディレクトリ構造を作成します。

    Args:
        output_dir: 出力ディレクトリのパス
    """
    for subdir in ["detections", "floormaps", "graphs", "labels"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
