"""
Test suites meant for testing the base functionality of the Embeddings feedback functions.
"""

import unittest
from trulens_eval.feedback.embeddings import Embeddings
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.embeddings = Embeddings(embed_model=self.embed_model)

    def test_cosine_distance_type(self):
        assert type(self.embeddings.cosine_distance("apple","banana")) == np.float64

    def test_cosine_distance_order(self):
        assert self.embeddings.cosine_distance("apple","john f kennedy") > self.embeddings.cosine_distance("apple","banana")
        assert self.embeddings.cosine_distance("apple","banana") > self.embeddings.cosine_distance("apple","apple")

    def test_manhattan_distance_type(self):
        assert type(self.embeddings.manhattan_distance("apple", "banana")) == np.float64

    def test_manhattan_distance_order(self):
        assert self.embeddings.manhattan_distance("apple", "john f kennedy") > self.embeddings.manhattan_distance("apple", "banana")
        assert self.embeddings.manhattan_distance("apple", "banana") > self.embeddings.manhattan_distance("apple", "apple")

    def test_euclidean_distance_type(self):
        assert type(self.embeddings.euclidean_distance("apple", "banana")) == np.float64

    def test_euclidean_distance_order(self):
        assert self.embeddings.euclidean_distance("apple", "john f kennedy") > self.embeddings.euclidean_distance("apple", "banana")
        assert self.embeddings.euclidean_distance("apple", "banana") > self.embeddings.euclidean_distance("apple", "apple")
if __name__ == "__main__":
    unittest.main()