import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.vectorstore import VectorStore
from src.ikepono.labeledimageembedding import LabeledImageEmbedding


class VectorStoreTests(unittest.TestCase):
    def test_empty_store(self):
        store = VectorStore(dimension=128)
        assert store.all_labels().shape == (), f"Expected (), got {store.all_labels().shape}"
        assert store.all_sources().shape == (), f"Expected (), got {store.all_sources().shape}"
        assert store.all_vectors().shape == (
        0, 128), f"Expected (0, dimension), got {store.all_vectors().shape}"

    def test_add_vector(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        store._add_embedding(random_vector, "label1", Path("source1"), np.int64(0))
        assert store.all_vectors().shape == (
        1, 128), f"Expected (1, dimension), got {store.all_vectors().shape}"
        assert store.all_labels() == ["label1"]
        assert store.all_sources() == [Path("source1")]

    def test_get_vector(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        store._add_embedding(random_vector, "label1", Path("source1"), np.int64(0))
        vector = store.vector_for_source(Path("source1"))
        assert np.allclose(vector, random_vector), f"Expected {random_vector}, got {store.vector_for_source('source1')}"

    def test_update_vector(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        random_vector[0] = 0.0
        store._add_embedding(random_vector, "label1", Path("source1"), np.int64(0))
        store.vector_for_source(Path("source1"))[0] == 0.0, f"Expected 0.0, got {store.vector_for_source(Path('source1'))[0]}"
        random_vector[0] = 1.0
        store.update_or_add_vector(Path("source1"), random_vector, "label2")
        assert store.vector_for_source(Path("source1"))[0] == 1.0, f"Expected 1.0, got {store.vector_for_source('source1')[0]}"
        labels = store.all_labels()
        assert np.array_equal(labels, ["label1", "label2"])  # Note that now `label1` will be empty.
        vectors = store.vectors_for_label("label1")
        assert vectors.shape == (0,), f"Expected empty, got {store.vectors_by_label('label1').shape}"
        assert store.all_sources() == [Path("source1")]
        assert store.all_vectors().shape == (
        1, 128), f"Expected (1, dimension), got {store.all_vectors().shape}"

    def test_search(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        store._add_embedding(random_vector, "label1", Path("source1"), np.int64(0))
        search_results = store.search(random_vector, 1)
        assert len(search_results) == 1, f"Expected 1, got {len(search_results)}"
        distance, label, source = search_results[0]
        assert distance == 0.0, f"Expected 0.0, got {distance}"
        assert label == "label1", f"Expected 'label1', got {label}"
        assert source == Path("source1"), f"Expected 'source1', got {source}"

    def test_get_vectors_by_label(self):
        store = VectorStore(dimension=128)
        random_vector = np.random.rand(128).astype('float32')
        store._add_embedding(random_vector, "label1", Path("source1"), np.int64(0))
        vectors = store.vectors_for_label("label1")
        assert vectors.shape == (1, 128), f"Expected (1, dimension), got {vectors.shape}"
        assert np.allclose(vectors[0], random_vector), f"Expected {random_vector}, got {vectors[0]}"

    def test_get_all_functions(self):
        store = VectorStore(dimension=128)
        random_vectors = []
        for i in range(10):
            random_vector = np.random.rand(128).astype('float32')
            store._add_embedding(random_vector, f"label{i}", Path(f"source{i}"), np.int64(i))
            random_vectors.append(random_vector)
        all_vectors = store.all_vectors()
        assert all_vectors.shape == (10, 128), f"Expected (1, dimension), got {all_vectors.shape}"
        for i in range(10):
            assert np.allclose(all_vectors[i], random_vectors[i]), f"Unexpected difference at vector {i}"
        all_labels = store.all_labels()
        assert np.array_equal(all_labels, [f"label{i}" for i in
                                           range(10)]), f"Expected {[f'label{i}' for i in range(10)]}, got {all_labels}"
        all_sources = store.all_sources()
        assert np.array_equal(all_sources, [Path(f"source{i}") for i in range(
            10)]), f"Expected {[f'source{i}' for i in range(10)]}, got {all_sources}"

    def test_compute_distances(self):
        store = VectorStore(dimension=128)
        random_vectors = []
        for i in range(10):
            random_vector = np.random.rand(128).astype('float32')
            store._add_embedding(random_vector, f"label{i}", Path(f"source{i}"), np.int64(i))
            random_vectors.append(random_vector)
        all_vectors = store.all_vectors()
        distances = store.distances(random_vectors[0], all_vectors)
        assert distances.shape == (10,), f"Expected (10,), got {distances.shape}"
        assert distances[0] == 0.0, f"Expected 0.0, got {distances[0]}"
        assert np.all(distances[1:] > 0.0), f"Expected all distances to be greater than 0.0, got {distances[1:]}"

    def test_get_sources_by_label(self):
        store = VectorStore(dimension=128)
        random_vectors = []
        for i in range(10):
            random_vector = np.random.rand(128).astype('float32')
            store._add_embedding(random_vector, f"label{i}", Path(f"source{i}"), np.int64(i))
            random_vectors.append(random_vector)
        sources = store.sources_for_label("label0")
        assert sources.shape == (1,), f"Expected (1,), got {sources.shape}"
        assert sources[0] == Path("source0"), f"Expected 'source0', got {sources[0]}"

    def test_add_labeled_image_vectors(self):
        store = VectorStore(dimension=128)
        rs = []
        for label in ['foo', 'bar', 'bat']:
            for i in range(10):
                random_vector = np.random.rand(128).astype('float32')
                rs.append(LabeledImageEmbedding(embedding=random_vector, label=f"label_{label}",
                                                source=Path(f"source_{label}_{i}"), dataset_index=np.int64(i)))
        store._add_labeled_image_embeddings(rs)
        all_vectors = store.all_vectors()
        distances = store.distances(rs[0].embedding, all_vectors)
        assert distances.shape == (len(all_vectors),), f"Expected ({len(all_vectors)},), got {distances.shape}"
        assert distances[0] == 0.0, f"Expected 0.0, got {distances[0]}"
        assert np.all(distances[1:] > 0.0), f"Expected all distances to be greater than 0.0, got {distances[1:]}"

    def test_label_to_ids(self):
        store = VectorStore(dimension=128)
        rs = []
        for label in ['foo', 'bar', 'bat']:
            for i in range(10):
                random_vector = np.random.rand(128).astype('float32')
                rs.append(LabeledImageEmbedding(embedding=random_vector, label=f"label_{label}",
                                                source=Path(f"source_{label}_{i}"), dataset_index=np.int64(i)))
        store._add_labeled_image_embeddings(rs)
        idxs = store.dataset_ids_for_label("label_foo")
        assert idxs.shape == (10,), f"Expected (10,), got {idxs.shape}"

    def test_id_for_source(self):
        store = VectorStore(dimension=128)
        rs = []
        for label in ['foo', 'bar', 'bat']:
            for i in range(10):
                random_vector = np.random.rand(128).astype('float32')
                rs.append(LabeledImageEmbedding(embedding=random_vector, label=f"label_{label}",
                                                source=Path(f"source_{label}_{i}"), dataset_index=np.int64(i)))
        store._add_labeled_image_embeddings(rs)
        idx = store.dataset_id_for_source(Path("source_foo_0"))
        assert idx == 0, f"Expected 0, got {idx}"

    def test_save(self):
        assert False

    def test_load(self):
        assert False

if __name__ == '__main__':
    unittest.main()
