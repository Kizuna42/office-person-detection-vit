#!/usr/bin/env python
"""依存関係のインストール確認スクリプト

このスクリプトは、プロジェクトに必要な依存関係が正しくインストールされているかを確認します。
"""

import sys

# 必須パッケージ
REQUIRED_PACKAGES = {
    "torch": "PyTorch",
    "torchvision": "TorchVision",
    "transformers": "Transformers",
    "timm": "TIMM",
    "cv2": "OpenCV",
    "PIL": "Pillow",
    "numpy": "NumPy",
    "yaml": "PyYAML",
    "matplotlib": "Matplotlib",
    "pytesseract": "pytesseract",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
    "Levenshtein": "python-Levenshtein",
    "skimage": "scikit-image",
    "pandas": "pandas",
}

# オプションパッケージ
OPTIONAL_PACKAGES = {
    "paddleocr": "PaddleOCR",
    "easyocr": "EasyOCR",
}


def check_package(package_name: str, display_name: str) -> tuple[bool, str]:
    """パッケージのインストール状況を確認

    Args:
        package_name: インポート名
        display_name: 表示名

    Returns:
        (インストール済みか, バージョン情報)
    """
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "未インストール"


def check_tesseract() -> tuple[bool, str]:
    """Tesseract OCRのインストール状況を確認"""
    try:
        import pytesseract

        version = pytesseract.get_tesseract_version()
        return True, version
    except Exception as e:
        return False, f"エラー: {e}"


def main():
    """メイン処理"""
    print("=" * 80)
    print("依存関係のインストール確認")
    print("=" * 80)
    print()

    # Pythonバージョン確認
    python_version = sys.version.split()[0]
    print(f"Python バージョン: {python_version}")
    if sys.version_info < (3, 10):
        print("⚠️  警告: Python 3.10以上を推奨します")
    print()

    # 必須パッケージの確認
    print("【必須パッケージ】")
    print("-" * 80)
    all_required_ok = True
    for package_name, display_name in REQUIRED_PACKAGES.items():
        installed, version = check_package(package_name, display_name)
        status = "✅" if installed else "❌"
        print(f"{status} {display_name:20s} {version}")
        if not installed:
            all_required_ok = False
    print()

    # オプションパッケージの確認
    print("【オプションパッケージ】")
    print("-" * 80)
    for package_name, display_name in OPTIONAL_PACKAGES.items():
        installed, version = check_package(package_name, display_name)
        status = "✅" if installed else "⚠️  （オプション）"
        print(f"{status} {display_name:20s} {version}")
    print()

    # Tesseract OCRの確認
    print("【システム依存関係】")
    print("-" * 80)
    tesseract_ok, tesseract_version = check_tesseract()
    status = "✅" if tesseract_ok else "❌"
    print(f"{status} Tesseract OCR        {tesseract_version}")
    if not tesseract_ok:
        print("   インストール方法:")
        print("   macOS: brew install tesseract tesseract-lang")
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
    print()

    # 結果サマリー
    print("=" * 80)
    if all_required_ok and tesseract_ok:
        print("✅ すべての必須依存関係がインストールされています")
        return 0
    else:
        print("❌ 一部の依存関係が不足しています")
        print("   pip install -r requirements.txt を実行してください")
        return 1


if __name__ == "__main__":
    sys.exit(main())
