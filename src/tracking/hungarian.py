"""Hungarian Algorithm implementation for assignment problem."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class HungarianAlgorithm:
    """ハンガリアンアルゴリズム（割り当て問題の解法）

    コスト行列から最適な割り当てを見つけます。
    """

    def __init__(self):
        """HungarianAlgorithmを初期化"""
        logger.debug("HungarianAlgorithm initialized")

    def solve(self, cost_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        """割り当て問題を解く

        Args:
            cost_matrix: コスト行列 (n, m)
                cost_matrix[i, j] は i を j に割り当てるコスト

        Returns:
            (割り当て配列, 総コスト)
            割り当て配列: 各行が割り当てられた列のインデックス（-1は未割り当て）
        """
        if cost_matrix.size == 0:
            return np.array([], dtype=np.int32), 0.0

        _rows, _cols = cost_matrix.shape

        # 簡易実装: 貪欲法（本番環境ではscipy.optimize.linear_sum_assignmentを使用推奨）
        assignment, total_cost = self._greedy_assignment(cost_matrix)

        return assignment, total_cost

    def _greedy_assignment(self, cost_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        """貪欲法による割り当て（簡易実装）

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
