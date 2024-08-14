import unittest

import numpy as np
import torch.nn as nn
import torch
import sys
import os
from pathlib import Path

from torch.utils.data import DataLoader

from ikepono.labeledimageembedding import LabeledImageEmbedding, LabeledImageTensor
from ikepono.splittableimagedataset import SplittableImageDataset

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
        for i in cls._dataset.train_indices:
            v = np.random.rand(128)
            lie = LabeledImageEmbedding(embedding=v, label=cls._dataset.labels[i], source=cls._dataset.image_paths[i], dataset_index=i)
            lies.append(lie)
        # OK, so the problem is that I need to pass down the dataset index to the vector store
        cls._vector_store.initialize(lies)
        for source in cls._vector_store.all_sources():
            index_in_dataset = cls._dataset.image_paths.index(source)
            assert index_in_dataset in cls._dataset.train_indices, f"Expected {index_in_dataset} to be in train indices {cls._dataset.train_indices}"

    def test_len(self):
        sampler = HardTripletBatchSampler(SamplerTests._dataset, n_triplets=3)
        sampler.initialize(SamplerTests._vector_store)
        # 10 photos in the dataset, 8 in train, 2 in test. 8 // 6 = 1
        assert len(sampler) == 1, f"Expected 9, got {len(sampler)}"

    def test_iter(self):
        sampler = SamplerTests.simple_sampler()
        assert isinstance(sampler, HardTripletBatchSampler), f"Expected HardTripletBatchSampler, got {type(sampler)}"
        assert len(sampler) == 1, f"Expected 9, got {len(sampler)}"
        assert sampler.get_initialized(), "Sampler not initialized"
        sampler_iter = iter(sampler)
        counts = {}
        for batch in sampler_iter:
            for sample in batch:
                assert isinstance(sample, np.int64), f"Expected int64, got {type(sample)}"
                assert sample >= 0
                assert sample < (len(SamplerTests._dataset.train_indices) + len(SamplerTests._dataset.test_indices)), f"Expected {sample} to be less than {len(SamplerTests._dataset)}"
                label = SamplerTests._dataset.labels[sample]
                assert label in sampler.vector_store.all_labels(), f"Label {label} not in vector store labels"
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1
            akari_count = counts['Akari']
            assert akari_count >= 3, f"Expected at least 3, got {akari_count}"
            vallaray_count = counts['Vallaray']
            assert vallaray_count >= 3, f"Expected at least 3, got {vallaray_count}"

    def test_batch_is_only_ever_from_trainset(self):
        sampler = SamplerTests.simple_sampler()
        assert isinstance(sampler, HardTripletBatchSampler), f"Expected HardTripletBatchSampler, got {type(sampler)}"
        assert len(sampler) == 1, f"Expected 9, got {len(sampler)}"
        assert sampler.get_initialized(), "Sampler not initialized"
        sampler_iter = iter(sampler)
        for sample in next(sampler_iter):
            assert sample in SamplerTests._dataset.train_indices, f"Expected {sample} to be in train indices {SamplerTests._dataset.train_indices}, got {sample}"

    def test_loader(self):
        data_dir = Path("/mnt/d/scratch_data/mantas/by_name/original/kona")
        dataset = SplittableImageDataset.from_directory(root_dir=data_dir, k=10)
        print("Built dataset")
        sampler = HardTripletBatchSampler(dataset, 3)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            collate_fn=LabeledImageTensor.collate)
        vs = VectorStore(dimension=128)
        lies = []
        for i in dataset.train_indices:
            v = np.random.rand(128)
            lie = LabeledImageEmbedding(embedding=v, label=dataset.labels[i], source=dataset.image_paths[i], dataset_index=i)
            lies.append(lie)
        vs.initialize(lies)
        loader.batch_sampler.initialize(vs)
        batch = next(iter(loader))
        assert len(batch["images"]) == 9, f"Expected 9, got {len(batch['images'])}"

    @classmethod
    def simple_sampler(cls):
        cls.setUpClass()
        n_triplets = 3
        dataset = SplittableImageDatasetTests.simple_dataset()

        sampler = HardTripletBatchSampler(dataset, n_triplets)
        sampler.initialize(cls._vector_store)
        return sampler