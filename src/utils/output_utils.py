"""Output directory management utilities."""

from pathlib import Path


def setup_output_directories(output_dir: Path) -> None:
    """出力ディレクトリを作成
    
    Args:
        output_dir: 出力ディレクトリのパス
    """
    for subdir in ['detections', 'floormaps', 'graphs', 'labels']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

