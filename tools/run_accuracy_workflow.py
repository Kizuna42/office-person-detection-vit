#!/usr/bin/env python3
"""座標変換精度向上ワークフロー実行スクリプト。

全ての診断・検証ツールを順番に実行し、総合レポートを生成します。

実行順序:
1. 変換精度診断 (visualize_transform_accuracy.py)
2. グリッドサーチ (parameter_grid_search.py)
3. ゾーン可視化 (zone_visualizer.py)
4. 総合レポート生成

使用方法:
    python tools/run_accuracy_workflow.py [--run-pipeline] [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """コマンドを実行。

    Returns:
        (成功フラグ, 出力)
    """
    logger.info(f"実行中: {description}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            logger.error(f"エラー: {output}")
            return False, output
        return True, output
    except Exception as e:
        logger.error(f"例外: {e}")
        return False, str(e)


def generate_summary_report(output_dir: Path) -> dict:
    """総合レポートを生成。"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "phases": {},
        "recommendations": [],
    }

    # 変換精度レポートを読み込み
    accuracy_report_path = output_dir / "transform_accuracy_report.json"
    if accuracy_report_path.exists():
        with open(accuracy_report_path, encoding="utf-8") as f:
            accuracy_data = json.load(f)
        report["phases"]["accuracy"] = {
            "rmse": accuracy_data.get("statistics", {}).get("rmse"),
            "max_error": accuracy_data.get("statistics", {}).get("max_error"),
            "num_valid": accuracy_data.get("num_valid"),
        }

        rmse = accuracy_data.get("statistics", {}).get("rmse", float("inf"))
        if rmse > 30:
            report["recommendations"].append("RMSE が 30px を超えています。対応点の追加・見直しが必要です。")
        elif rmse > 10:
            report["recommendations"].append("RMSE が 10px を超えています。パラメータの微調整を行ってください。")

    # グリッドサーチ結果を読み込み
    grid_search_path = output_dir / "grid_search_results.json"
    if grid_search_path.exists():
        with open(grid_search_path, encoding="utf-8") as f:
            grid_data = json.load(f)
        results = grid_data.get("results", [])
        if results:
            best = results[0]
            report["phases"]["grid_search"] = {
                "best_params": {
                    "pitch_deg": best.get("pitch_deg"),
                    "yaw_deg": best.get("yaw_deg"),
                    "height_m": best.get("height_m"),
                    "focal_length": best.get("focal_length"),
                },
                "best_rmse": best.get("rmse"),
            }

    # ゾーン分類率を読み込み（JSONからは取れないのでログから推測）
    zone_vis_path = output_dir / "zone_visualization.png"
    if zone_vis_path.exists():
        report["phases"]["zone_visualization"] = {
            "status": "completed",
            "image_path": str(zone_vis_path),
        }

    # 推奨事項を追加
    if not report["recommendations"]:
        report["recommendations"].append("目標精度を達成しています。End-to-End検証を実行してください。")

    return report


def main():
    parser = argparse.ArgumentParser(description="精度向上ワークフロー")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--output-dir",
        default="output/calibration",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="パイプラインも再実行",
    )
    parser.add_argument(
        "--skip-grid-search",
        action="store_true",
        help="グリッドサーチをスキップ",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("座標変換精度向上ワークフロー")
    print("=" * 70)
    print(f"出力ディレクトリ: {output_dir}")
    print("=" * 70)
    print()

    results = {}

    # Step 1: パイプライン実行（オプション）
    if args.run_pipeline:
        print("\n[Step 0] パイプライン実行")
        print("-" * 50)
        success, output = run_command(
            [sys.executable, "main.py"],
            "メインパイプライン実行",
        )
        results["pipeline"] = {"success": success}
        if not success:
            print("パイプラインの実行に失敗しました。続行しますか？")

    # Step 1: 変換精度診断
    print("\n[Step 1] 変換精度診断")
    print("-" * 50)
    success, output = run_command(
        [
            sys.executable,
            "tools/visualize_transform_accuracy.py",
            "--output-dir",
            str(output_dir),
        ],
        "変換精度診断ツール",
    )
    results["accuracy"] = {"success": success}
    print(output)

    # Step 2: グリッドサーチ
    if not args.skip_grid_search:
        print("\n[Step 2] パラメータグリッドサーチ")
        print("-" * 50)
        success, output = run_command(
            [
                sys.executable,
                "tools/parameter_grid_search.py",
                "--pitch-range",
                "5,25,2",
                "--yaw-range",
                "0,50,5",
                "--height-range",
                "1.8,3.0,0.2",
                "--focal-range",
                "1000,1500,100",
                "--top-n",
                "10",
                "--output-dir",
                str(output_dir),
            ],
            "パラメータグリッドサーチ",
        )
        results["grid_search"] = {"success": success}
        print(output)
    else:
        print("\n[Step 2] グリッドサーチをスキップ")

    # Step 3: ゾーン可視化
    print("\n[Step 3] ゾーン可視化")
    print("-" * 50)
    success, output = run_command(
        [
            sys.executable,
            "tools/zone_visualizer.py",
            "--output-dir",
            str(output_dir),
        ],
        "ゾーン可視化",
    )
    results["zone_viz"] = {"success": success}
    print(output)

    # Step 4: 総合レポート生成
    print("\n[Step 4] 総合レポート生成")
    print("-" * 50)
    summary = generate_summary_report(output_dir)
    summary["results"] = results

    summary_path = output_dir / "accuracy_workflow_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"総合レポートを保存: {summary_path}")

    # 結果を表示
    print("\n" + "=" * 70)
    print("ワークフロー完了")
    print("=" * 70)

    if summary["phases"].get("accuracy"):
        acc = summary["phases"]["accuracy"]
        rmse = acc.get("rmse", "N/A")
        max_err = acc.get("max_error", "N/A")
        print("\n現在の精度:")
        print(f"  RMSE: {rmse:.2f} px" if isinstance(rmse, float) else f"  RMSE: {rmse}")
        print(f"  最大誤差: {max_err:.2f} px" if isinstance(max_err, float) else f"  最大誤差: {max_err}")

    if summary["phases"].get("grid_search"):
        gs = summary["phases"]["grid_search"]
        best_rmse = gs.get("best_rmse", "N/A")
        best_params = gs.get("best_params", {})
        print("\nグリッドサーチ結果:")
        print(f"  最良RMSE: {best_rmse:.2f} px" if isinstance(best_rmse, float) else f"  最良RMSE: {best_rmse}")
        if best_params:
            print("  推奨パラメータ:")
            print(f"    pitch_deg: {best_params.get('pitch_deg')}")
            print(f"    yaw_deg: {best_params.get('yaw_deg')}")
            print(f"    height_m: {best_params.get('height_m')}")
            print(f"    focal_length: {best_params.get('focal_length')}")

    print("\n推奨事項:")
    for rec in summary["recommendations"]:
        print(f"  - {rec}")

    print("\n出力ファイル:")
    for f in output_dir.glob("*"):
        print(f"  - {f}")

    print("\n" + "=" * 70)
    print("次のステップ:")
    print("=" * 70)
    print("""
1. 対応点の追加・見直し:
   python tools/correspondence_collector.py

2. パラメータの微調整:
   python tools/calibration_tool.py

3. 調整後にパイプラインを再実行:
   python main.py

4. 結果の確認:
   python tools/run_accuracy_workflow.py
""")


if __name__ == "__main__":
    main()
