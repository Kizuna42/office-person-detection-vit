"""Output directory management utilities.

注意: このモジュールは後方互換性のため保持されています。
main.pyでは使用されていません（セッション管理が有効な場合はOutputManagerを使用）。
tools/内のスクリプトで使用される可能性があります。
"""

from pathlib import Path


# 後方互換性のため、既存の関数も保持
# main.pyでは使用されていません（セッション管理が有効な場合はOutputManagerを使用）
def setup_output_directories(output_dir: Path) -> None:
    """出力ディレクトリを作成（後方互換性のため保持）

    Args:
        output_dir: 出力ディレクトリのパス
    """
    for subdir in ["detections", "floormaps", "graphs", "labels"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
