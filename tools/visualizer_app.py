"""Streamlit app entry point for interactive visualization."""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.interactive_visualizer import main

if __name__ == "__main__":
    main()
