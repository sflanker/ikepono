import unittest
import torch.nn as nn
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ikepono.HardTripletSampler import HardTripletBatchSampler
from src.ikepono.VectorStore import VectorStore

class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._vector_store = VectorStore(dimension=128)
        for individual in range(9):
            for photo in range(4):
                random_vector = torch.rand(128).numpy().astype('float32')
                cls._vector_store.add_vector(random_vector, f"individual_{individual}", f"source_{individual}_{photo}")

    def test_len(self):
        sampler = HardTripletBatchSampler(SamplerTests._vector_store, 3, 4)
        assert len(sampler) == 3, f"Expected 3, got {len(sampler)}"

    def test_iter(self):
        individuals_per_batch = 3
        photos_per_triplet = 3 # Obviously
        sampler = HardTripletBatchSampler(SamplerTests._vector_store, individuals_per_batch, 4)
        for batch in sampler:
            assert len(batch) == individuals_per_batch * photos_per_triplet, f"Expected 3 * 3, got {len(batch)}"
            break