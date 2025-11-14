"""Streamlit app entry point for interactive visualization.

⚠️ DEPRECATED: このファイルは非推奨です。
直接 `tools/interactive_visualizer.py` を使用するか、
`streamlit run tools/interactive_visualizer.py` を実行してください。

このファイルは後方互換性のために残されていますが、
将来のバージョンで削除される予定です。
"""

from pathlib import Path
import sys
import warnings

# 非推奨警告を表示
warnings.warn(
    "visualizer_app.py は非推奨です。" "代わりに `streamlit run tools/interactive_visualizer.py` を使用してください。",
    DeprecationWarning,
    stacklevel=2,
)

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.interactive_visualizer import main

if __name__ == "__main__":
    main()
