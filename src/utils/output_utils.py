"""Output directory management utilities."""

from pathlib import Path


# 後方互換性のため、既存の関数も保持
def setup_output_directories(output_dir: Path) -> None:
    """出力ディレクトリを作成（後方互換性のため保持）

    Args:
        output_dir: 出力ディレクトリのパス
    """
    for subdir in ["detections", "floormaps", "graphs", "labels"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
