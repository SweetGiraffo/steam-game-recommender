import unittest
import numpy as np
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, map_at_k


class TestRankingMetrics(unittest.TestCase):
    def test_precision_at_k(self):
        actual = [1, 2, 3]
        predicted = [1, 4, 2, 5, 6]  # 2 hits in top 5
        self.assertAlmostEqual(precision_at_k(actual, predicted, k=5), 2.0 / 5.0)
        self.assertAlmostEqual(precision_at_k(actual, predicted, k=2), 1.0 / 2.0)
        self.assertAlmostEqual(precision_at_k([], predicted, k=5), 0.0)

    def test_recall_at_k(self):
        actual = [1, 2, 3, 4]
        predicted = [1, 5, 6, 2, 7]  # 2 hits out of 4 actuals
        self.assertAlmostEqual(recall_at_k(actual, predicted, k=5), 2.0 / 4.0)
        self.assertAlmostEqual(recall_at_k(actual, predicted, k=1), 1.0 / 4.0)

    def test_perfect_ndcg(self):
        actual = [10, 20, 30]
        predicted = [10, 20, 30, 40, 50]
        self.assertAlmostEqual(ndcg_at_k(actual, predicted, k=3), 1.0)

    def test_zero_ndcg(self):
        actual = [1, 2]
        predicted = [3, 4, 5]
        self.assertAlmostEqual(ndcg_at_k(actual, predicted, k=3), 0.0)

    def test_map_at_k(self):
        actual = [1, 3]
        # Predicted: 1 is rank 1 (hit, p=1/1=1), 2 is rank 2 (miss), 3 is rank 3 (hit, p=2/3)
        # Average Precision = (1.0 + 2/3) / 2 = 1.6667 / 2 = 0.8333
        predicted = [1, 2, 3, 4]
        self.assertAlmostEqual(map_at_k(actual, predicted, k=4), (1.0 + 2.0 / 3.0) / 2.0)


if __name__ == '__main__':
    unittest.main()
