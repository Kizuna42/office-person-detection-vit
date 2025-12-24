"""トラッキングベンチマークCLIランナー

使用例:
    python -m src.benchmark.tracking_runner --gt output/labels/result_fixed.json --pred output/tracking.csv
"""

from src.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
