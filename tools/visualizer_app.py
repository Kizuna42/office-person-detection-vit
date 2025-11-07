"""Streamlit app entry point for interactive visualization."""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.interactive_visualizer import main

if __name__ == "__main__":
    main()
