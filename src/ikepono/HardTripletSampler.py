import numpy as np
from torch.utils.data import Sampler


class HardTripletBatchSampler(Sampler):
    def __init__(self, vector_store, individuals_per_batch, max_photos_per_individual):
        self.vector_store = vector_store
        self.individuals_per_batch = individuals_per_batch
        self.max_photos_per_individual = max_photos_per_individual
        self.triplets = (max_photos_per_individual - 1) // 2

        self.individuals = self.vector_store.get_all_labels()

    def __iter__(self):
        while True:
            batch = []
            selected_individuals = np.random.choice(self.individuals, self.individuals_per_batch, replace=False)

            for individual in selected_individuals:
                positive_vectors = self.vector_store.get_vectors_by_label(individual)
                positive_sources = self.vector_store.get_sources_by_label(individual)
                primary_source = positive_sources[np.random.choice(positive_sources.shape[0])]

                primary_vector = self.vector_store.get_vector(primary_source)
                batch.append(('primary', primary_source, primary_vector))

                positive_distances = self.vector_store.compute_distances(primary_vector, positive_vectors)
                hardest_positives = np.argsort(positive_distances)[-self.triplets:]
                batch.extend([('distant_positive',positive_sources[i],positive_vectors[i]) for i in hardest_positives])

                negative_vectors = np.vstack([self.vector_store.get_vectors_by_label(label)
                                              for label in self.individuals if label != individual])
                negative_distances = self.vector_store.compute_distances(primary_vector, negative_vectors)
                hardest_negatives = np.argsort(negative_distances)[:self.triplets]

                negative_sources = [source for label in self.individuals for source in self.vector_store.get_sources_by_label(label)
                                    if label != individual]
                batch.extend([('nearby_negative',negative_sources[i], negative_vectors[i]) for i in hardest_negatives])

            yield batch

    def __len__(self):
        return len(self.vector_store.get_all_sources()) // (self.individuals_per_batch * self.max_photos_per_individual)