from typing import List, Iterable

import numpy as np
from torch.utils.data import Sampler

from ikepono.labeledimageembedding import LabeledImageEmbedding
from ikepono.splittableimagedataset import SplittableImageDataset
from ikepono.vectorstore import VectorStore


class HardTripletBatchSampler(Sampler):
    def __init__(self, dataset: SplittableImageDataset, vector_store : VectorStore, individuals_per_batch: int, i_near_same = 1, i_near_others=1, max_photos_per_individual:int=-1):
        self.dataset = dataset
        self.vector_store = vector_store
        self.individuals_per_batch = individuals_per_batch
        self.max_photos_per_individual = len(dataset) if max_photos_per_individual == -1 else max_photos_per_individual
        self.triplets = self.max_photos_per_individual // 3
        self.i_far_same = i_near_same
        self.i_near_others = i_near_others

        if self.vector_store.get_all_labels().size == 0:
            self.vector_store.build_labeled_image_embeddings(dataset, individuals_per_batch, max_photos_per_individual)
        self.individuals = self.vector_store.get_all_labels()

    # Needs to return _indexes_ of the dataset, not the actual data
    def __iter__(self) -> Iterable[List[int]]:
        while True:
            batch = []
            num_individuals = min(self.individuals_per_batch, len(self.individuals))
            selected_individuals = np.random.choice(self.individuals, num_individuals, replace=False)

            for individual in selected_individuals:
                positive_indices = self.vector_store.label_to_ids[individual]
                primary_index = np.random.choice(positive_indices)
                batch.append(primary_index)

                primary_vector = self.vector_store.get_vector(self.vector_store.id_to_source[primary_index]).numpy()
                positive_vectors = self.vector_store.get_vectors_by_label(individual)
                positive_distances = self.vector_store.compute_distances(primary_vector, positive_vectors)
                num_hardest_positives = min(self.i_far_same, len(positive_indices) - 1)
                hardest_positives = np.argsort(positive_distances)[-num_hardest_positives:]
                batch.extend(
                    [idx for i, idx in enumerate(positive_indices) if i in hardest_positives and idx != primary_index])

                negative_indices = [idx for label in self.individuals
                                    if label != individual
                                    for idx in self.vector_store.label_to_ids[label]]
                negative_vectors = np.vstack([self.vector_store.get_vector(self.vector_store.id_to_source[idx]).numpy()
                                              for idx in negative_indices])
                negative_distances = self.vector_store.compute_distances(primary_vector, negative_vectors)
                num_hardest_negatives = min(self.i_near_others, len(negative_indices))
                hardest_negatives = np.argsort(negative_distances)[:num_hardest_negatives]
                batch.extend([negative_indices[i] for i in hardest_negatives])
            yield batch
    def __len__(self):
        photos_per_individual = 1 + self.i_far_same + self.i_near_others # primary + distant positives + nearby negatives
        return len(self.dataset) // (self.individuals_per_batch * photos_per_individual)