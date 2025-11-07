"""Command-line argument parsing."""

import argparse


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパースする

    Returns:
        パース済み引数
    """
    parser = argparse.ArgumentParser(description="オフィス人物検出システム - ViTベースの人物検出とゾーン別集計")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）",
    )

    parser.add_argument("--debug", action="store_true", help="デバッグモードで実行（詳細ログ、中間結果出力）")

    parser.add_argument("--evaluate", action="store_true", help="精度評価モードで実行（Ground Truthとの比較）")

    parser.add_argument("--fine-tune", action="store_true", help="ファインチューニングモードで実行")

    parser.add_argument("--start-time", type=str, help="開始時刻を指定（HH:MM形式）、指定しない場合は自動検出")

    parser.add_argument("--end-time", type=str, help="終了時刻を指定（HH:MM形式）、指定しない場合は自動検出")

    parser.add_argument(
        "--timestamps-only", action="store_true", help="5分刻みのフレーム抽出+タイムスタンプOCRのみを実行"
    )

    return parser.parse_args()
