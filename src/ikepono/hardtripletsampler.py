from typing import List, Iterable

import numpy as np
from torch.utils.data import Sampler

from ikepono.labeledimageembedding import LabeledImageEmbedding
from ikepono.splittableimagedataset import SplittableImageDataset
from ikepono.vectorstore import VectorStore


class HardTripletBatchSampler(Sampler):
    def __init__(self, dataset: SplittableImageDataset, n_triplets: int):
        self.dataset = dataset
        self.n_triplets = n_triplets
        self.vector_store = None
        self.individuals = None
        self._initialized = False


    def initialize(self, initialized_vector_store: VectorStore):
        assert initialized_vector_store.get_initialized(), "Vector store must be initialized before initializing the sampler"
        self.vector_store = initialized_vector_store
        self.individuals = list(self.vector_store.label_to_dataset_ids.keys())
        self._initialized = True

    def get_initialized(self) -> bool:
        return self._initialized


    # Needs to return _indexes_ of the dataset, not the actual data
    def __iter__(self) -> Iterable[int]:
        assert self._initialized == True, "Sampler must be initialized before iterating"

        for _ in range(len(self)):
            batch = []
            for _ in range(self.n_triplets):
                triplet = []
                individual = np.random.choice(self.individuals)

                positive_indices = self.vector_store.label_to_dataset_ids[individual]
                if len(positive_indices) < 2:
                    print("breaking")
                assert len(positive_indices) >= 2, "No more hard positives"
                primary_index = np.random.choice(positive_indices)
                triplet.append(np.int64(primary_index))

                primary_vector_vector_id = self.vector_store.dataset_id_to_vector_id[primary_index]
                source = self.vector_store.vector_id_to_source[primary_vector_vector_id]
                primary_vector = self.vector_store.vector_for_source(source)
                positive_vectors = self.vector_store.vectors_for_label(individual)
                positive_distances = self.vector_store.distances(primary_vector, positive_vectors)
                num_hardest_positives = 1
                hardest_positives = np.argsort(positive_distances)[-num_hardest_positives:]

                for i, idx in enumerate(positive_indices):
                    if i in hardest_positives and idx != primary_index:
                        triplet.append(np.int64(idx))

                all_dataset_indices = self.vector_store.all_dataset_ids()
                negative_indices = [idx for idx in all_dataset_indices if idx not in positive_indices]
                negative_vector_indices = self.vector_store.vector_indices_for_dataset_indices(negative_indices)
                if len(negative_indices) == 0:
                    print("Breaking")
                assert len(negative_indices) > 0, "No negative indices found"
                negative_vectors = np.vstack([self.vector_store.vector_for_source(self.vector_store.vector_id_to_source[idx])
                                              for idx in negative_vector_indices])
                negative_distances = self.vector_store.distances(primary_vector, negative_vectors)
                num_hardest_negatives = 1
                hardest_negatives = np.argsort(negative_distances)[:num_hardest_negatives]
                for i in hardest_negatives:
                    triplet.append(np.int64(negative_indices[i]))
                batch.extend(triplet)
            yield batch

    def __len__(self):
        train_count = len(self.dataset.train_indices)
        batch_size = self.n_triplets * 3
        batch_count = train_count // batch_size
        return batch_count
