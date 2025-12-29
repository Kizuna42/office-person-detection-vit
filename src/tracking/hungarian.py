"""Hungarian Algorithm implementation for assignment problem.

Uses scipy.optimize.linear_sum_assignment for guaranteed optimal solution.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class HungarianAlgorithm:
    """ハンガリアンアルゴリズム（割り当て問題の解法）

    scipy.optimize.linear_sum_assignment を使用して最適解を保証します。
    """

    def __init__(self):
        """HungarianAlgorithmを初期化"""
        logger.debug("HungarianAlgorithm initialized (using scipy optimal solver)")

    def solve(self, cost_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        """割り当て問題を解く（最適解保証）

        Args:
            cost_matrix: コスト行列 (n, m)
                cost_matrix[i, j] は i を j に割り当てるコスト

        Returns:
            (割り当て配列, 総コスト)
            割り当て配列: 各行が割り当てられた列のインデックス（-1は未割り当て）
        """
        if cost_matrix.size == 0:
            return np.array([], dtype=np.int32), 0.0

        n_rows, _n_cols = cost_matrix.shape

        # scipy.optimize.linear_sum_assignment で最適解を求める
        # 大きなコスト（inf等）を扱えるように、非常に大きい値を有限値に置換
        cost_finite = np.where(np.isinf(cost_matrix), 1e9, cost_matrix)

        try:
            row_idx, col_idx = linear_sum_assignment(cost_finite)
        except ValueError as e:
            logger.warning(f"linear_sum_assignment failed: {e}, falling back to greedy")
            return self._greedy_assignment(cost_matrix)

        # 結果を割り当て配列に変換
        assignment = np.full(n_rows, -1, dtype=np.int32)
        total_cost = 0.0

        for r, c in zip(row_idx, col_idx, strict=True):
            # 元のコスト行列で inf だった場合は未割り当てとする
            if not np.isinf(cost_matrix[r, c]):
                assignment[r] = c
                total_cost += cost_matrix[r, c]

        return assignment, total_cost

    def _greedy_assignment(self, cost_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        """貪欲法による割り当て（フォールバック用）

        Args:
            cost_matrix: コスト行列 (n, m)

        Returns:
            (割り当て配列, 総コスト)
        """
        n, m = cost_matrix.shape
        assignment = np.full(n, -1, dtype=np.int32)
        used = np.zeros(m, dtype=bool)
        total_cost = 0.0

        # 各行について、最小コストの未使用列を選択
        for i in range(n):
            min_cost = np.inf
            min_j = -1

            for j in range(m):
                if not used[j] and cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    min_j = j

            if min_j >= 0:
                assignment[i] = min_j
                used[min_j] = True
                total_cost += min_cost

        return assignment, total_cost
