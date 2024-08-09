import unittest
import torch.nn as nn
import torch
import sys
import os
from pathlib import Path

from ikepono.labeledimageembedding import LabeledImageEmbedding

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.hardtripletsampler import HardTripletBatchSampler
from src.ikepono.vectorstore import VectorStore
from tests.test_splittableimagedataset import SplittableImageDatasetTests

class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dataset = SplittableImageDatasetTests.simple_dataset()
        cls._vector_store = VectorStore(dimension=128)
        # Mock embeddings for the dataset
        lies = []
        for i in range(len(cls._dataset)):
            v = torch.randn(128)
            lie = LabeledImageEmbedding(embedding=v, label=cls._dataset.labels[i], source=cls._dataset.image_paths[i])
            lies.append(lie)
        cls._vector_store.add_labeled_image_vectors(lies)

    def test_len(self):
        sampler = HardTripletBatchSampler(SamplerTests._dataset, SamplerTests._vector_store, individuals_per_batch=2)
        # 2 individuals per batch, 3 photos per triplet = 6 photos per batch.
        # 10 photos in the dataset, 8 in train, 2 in test. 8 // 6 = 1
        assert len(sampler) == 1, f"Expected 1, got {len(sampler)}"

    def test_iter(self):
        individuals_per_batch = 2
        photos_per_triplet = 3  # Obviously
        sampler = SamplerTests.simple_sampler()
        expected_batch_count = individuals_per_batch * photos_per_triplet
        batch = sampler.__iter__().__next__()
        assert len(batch) == expected_batch_count, f"Expected {expected_batch_count}, got {len(batch)}"

    @classmethod
    def simple_sampler(cls):
        cls.setUpClass()
        individuals_per_batch = 2
        photos_per_triplet = 3  # Obviously
        dataset = SplittableImageDatasetTests.simple_dataset()
        sampler = HardTripletBatchSampler(dataset, SamplerTests._vector_store, individuals_per_batch)
        return sampler