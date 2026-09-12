import unittest
import pickle
import numpy as np
from src.config import MODEL_CACHE_FILE


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MODEL_CACHE_FILE, 'rb') as f:
            cls.artifacts = pickle.load(f)
        cls.cf_model = cls.artifacts['cf_model']
        cls.cb_model = cls.artifacts['cb_model']
        cls.hybrid_model = cls.artifacts['hybrid_model']
        cls.pop_model = cls.artifacts['pop_model']
        cls.sparse_mat = cls.artifacts['sparse_matrix']

    def test_popularity_recommender(self):
        recs = self.pop_model.recommend(user_idx=0, n=5)
        self.assertEqual(len(recs), 5)
        scores = [s for _, s in recs]
        self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))

    def test_content_based_similar_items(self):
        recs = self.cb_model.similar_items(game_idx=0, n=5)
        self.assertEqual(len(recs), 5)
        # Ensure game_idx 0 is not in recs
        rec_ids = [idx for idx, _ in recs]
        self.assertNotIn(0, rec_ids)

    def test_implicit_als_recommend(self):
        recs = self.cf_model.recommend(user_idx=0, n=5)
        self.assertLessEqual(len(recs), 5)
        # Ensure no negative infinity
        for idx, score in recs:
            self.assertGreater(score, -1e20)

    def test_hybrid_recommend(self):
        recs = self.hybrid_model.recommend(user_idx=0, train_matrix=self.sparse_mat, n=5)
        self.assertEqual(len(recs), 5)

    def test_cold_start_custom(self):
        recs = self.hybrid_model.recommend_custom(liked_game_indices=[0, 1], n=5)
        self.assertEqual(len(recs), 5)
        rec_ids = [idx for idx, _ in recs]
        self.assertNotIn(0, rec_ids)
        self.assertNotIn(1, rec_ids)


if __name__ == '__main__':
    unittest.main()
