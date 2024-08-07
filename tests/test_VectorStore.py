import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.VectorStore import VectorStore

class VectorStoreTests(unittest.TestCase):
    def test_empty_store(self):
        store = VectorStore(dimension=128)
        assert store.get_all_labels().shape == (), f"Expected (), got {store.get_all_labels().shape}"
        assert store.get_all_sources().shape == (), f"Expected (), got {store.get_all_sources().shape}"
        assert store.get_all_vectors().shape == (0, 128), f"Expected (0, dimension), got {store.get_all_vectors().shape}"

    def test_add_vector(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        store.add_vector(random_vector, "label1", "source1")
        assert store.get_all_vectors().shape == (1, 128), f"Expected (1, dimension), got {store.get_all_vectors().shape}"
        assert store.get_all_labels() == ["label1"]
        assert store.get_all_sources() == ["source1"]

    def test_get_vector(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        store.add_vector(random_vector, "label1", "source1")
        vector = store.get_vector("source1")
        assert np.allclose(vector, random_vector), f"Expected {random_vector}, got {store.get_vector('source1')}"
    def test_update_vector(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        random_vector[0] = 0.0
        store.add_vector(random_vector, "label1", "source1")
        store.get_vector("source1")[0] == 0.0, f"Expected 0.0, got {store.get_vector('source1')[0]}"
        random_vector[0] = 1.0
        store.update_vector("source1", random_vector, "label2")
        assert store.get_vector("source1")[0] == 1.0, f"Expected 1.0, got {store.get_vector('source1')[0]}"
        labels = store.get_all_labels()
        assert np.array_equal(labels, ["label1", "label2"]) # Note that now `label1` will be empty.
        vectors = store.get_vectors_by_label("label1")
        assert vectors.shape == (0,), f"Expected empty, got {store.get_vectors_by_label('label1').shape}"
        assert store.get_all_sources() == ["source1"]
        assert store.get_all_vectors().shape == (1, 128), f"Expected (1, dimension), got {store.get_all_vectors().shape}"

    def test_search(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        store.add_vector(random_vector, "label1", "source1")
        search_results = store.search(random_vector, 1)
        assert len(search_results) == 1, f"Expected 1, got {len(search_results)}"
        distance, label, source = search_results[0]
        assert distance == 0.0, f"Expected 0.0, got {distance}"
        assert label == "label1", f"Expected 'label1', got {label}"
        assert source == "source1", f"Expected 'source1', got {source}"

    def test_get_vectors_by_label(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        store.add_vector(random_vector, "label1", "source1")
        vectors = store.get_vectors_by_label("label1")
        assert vectors.shape == (1, 128), f"Expected (1, dimension), got {vectors.shape}"
        assert np.allclose(vectors[0], random_vector), f"Expected {random_vector}, got {vectors[0]}"

    def test_get_all_functions(self):
        store = VectorStore(dimension=128)
        random_vectors = []
        for i in range(10):
            random_vector = np.random.rand(128).astype('float32')
            store.add_vector(random_vector, f"label{i}", f"source{i}")
            random_vectors.append(random_vector)
        all_vectors = store.get_all_vectors()
        assert all_vectors.shape == (10, 128), f"Expected (1, dimension), got {all_vectors.shape}"
        for i in range(10):
            assert np.allclose(all_vectors[i], random_vectors[i]), f"Unexpected difference at vector {i}"
        all_labels = store.get_all_labels()
        assert np.array_equal(all_labels, [f"label{i}" for i in range(10)]), f"Expected {[f'label{i}' for i in range(10)]}, got {all_labels}"
        all_sources = store.get_all_sources()
        assert np.array_equal(all_sources, [f"source{i}" for i in range(10)]), f"Expected {[f'source{i}' for i in range(10)]}, got {all_sources}"


    def test_compute_distances(self):
        store = VectorStore(dimension=128)
        random_vectors = []
        for i in range(10):
            random_vector = np.random.rand(128).astype('float32')
            store.add_vector(random_vector, f"label{i}", f"source{i}")
            random_vectors.append(random_vector)
        all_vectors = store.get_all_vectors()
        distances = store.compute_distances(random_vectors[0], all_vectors)
        assert distances.shape == (10,), f"Expected (10,), got {distances.shape}"
        assert distances[0] == 0.0, f"Expected 0.0, got {distances[0]}"
        assert np.all(distances[1:] > 0.0), f"Expected all distances to be greater than 0.0, got {distances[1:]}"

    def test_get_sources_by_label(self):
        store = VectorStore(dimension=128)
        random_vectors = []
        for i in range(10):
            random_vector = np.random.rand(128).astype('float32')
            store.add_vector(random_vector, f"label{i}", f"source{i}")
            random_vectors.append(random_vector)
        sources = store.get_sources_by_label("label0")
        assert sources.shape == (1,), f"Expected (1,), got {sources.shape}"
        assert sources[0] == "source0", f"Expected 'source0', got {sources[0]}"

if __name__ == '__main__':
    unittest.main()
