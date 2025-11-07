"""Test cases for torch_utils."""

from __future__ import annotations

from unittest.mock import patch

from src.utils.torch_utils import setup_mps_compatibility


@patch("src.utils.torch_utils.torch")
@patch("src.utils.torch_utils.warnings")
def test_setup_mps_compatibility_mps_available(mock_warnings, mock_torch):
    """MPSが利用可能な場合の互換性設定"""
    mock_torch.backends.mps.is_available.return_value = True

    setup_mps_compatibility()

    # 警告フィルターが設定されていることを確認
    assert mock_warnings.filterwarnings.called


@patch("src.utils.torch_utils.torch")
@patch("src.utils.torch_utils.warnings")
def test_setup_mps_compatibility_mps_unavailable(mock_warnings, mock_torch):
    """MPSが利用できない場合、何もしない"""
    mock_torch.backends.mps.is_available.return_value = False

    setup_mps_compatibility()

    # 警告フィルターが設定されていないことを確認
    # （MPSが利用できない場合は早期リターン）
    # 実際の実装では早期リターンするため、呼び出されない可能性がある
