import unittest
import numpy as np
from scipy.sparse import csr_matrix
from src.data.dataset import load_dataset, train_test_split_warm, train_test_split_cold_items


class TestDataPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.sparse_mat, cls.mappings = load_dataset()

    def test_matrix_dimensions(self):
        self.assertGreater(self.sparse_mat.shape[0], 1000, 'User count should exceed 1000')
        self.assertGreater(self.sparse_mat.shape[1], 1000, 'Game count should exceed 1000')
        self.assertTrue(isinstance(self.sparse_mat, csr_matrix))

    def test_mappings_consistency(self):
        user2idx = self.mappings['user2idx']
        idx2user = self.mappings['idx2user']
        game2idx = self.mappings['game2idx']
        idx2game = self.mappings['idx2game']

        # Spot check round-trips
        first_user = list(user2idx.keys())[0]
        u_idx = user2idx[first_user]
        self.assertEqual(idx2user[str(u_idx)], first_user)

        first_game = list(game2idx.keys())[0]
        g_idx = game2idx[first_game]
        self.assertEqual(idx2game[str(g_idx)], first_game)

    def test_warm_split(self):
        train_sparse, test_gt, train_gt = train_test_split_warm(self.df, test_ratio=0.2)
        self.assertEqual(train_sparse.shape, self.sparse_mat.shape)
        self.assertGreater(len(test_gt), 1000)

        # Check that no item is in both train and test for the same user
        for u_idx, test_items in list(test_gt.items())[:100]:
            train_items = set(train_sparse[u_idx].indices)
            overlap = train_items.intersection(set(test_items))
            self.assertEqual(len(overlap), 0, f'Data leakage detected for user {u_idx}')


if __name__ == '__main__':
    unittest.main()
