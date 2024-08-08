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

        if len(vector_store.get_all_labels()) == 0:
            vector_store.build_labeled_image_embeddings(dataset, individuals_per_batch, max_photos_per_individual)
        self.individuals = self.vector_store.get_all_labels()

    def __iter__(self):
        while True:
            batch = []
            selected_individuals = np.random.choice(self.individuals, self.individuals_per_batch, replace=True)

            for individual in selected_individuals:
                positive_vectors = self.vector_store.get_vectors_by_label(individual)
                positive_sources = self.vector_store.get_sources_by_label(individual)
                primary_source = positive_sources[np.random.choice(positive_sources.shape[0])]

                primary_vector = self.vector_store.get_vector(primary_source).numpy()
                batch.append(('primary', LabeledImageEmbedding(embedding=primary_vector, label=individual, source=primary_source)))

                positive_distances = self.vector_store.compute_distances(primary_vector, positive_vectors)
                hardest_positives = np.argsort(positive_distances)[-self.i_far_same:]
                batch.extend([('distant_positive', LabeledImageEmbedding(embedding=positive_vectors[i],
                                                                         label = individual,
                                                                         source = positive_sources[i])) for i in hardest_positives])

                negative_vectors = np.vstack([self.vector_store.get_vectors_by_label(label)
                                              for label in self.individuals if label != individual])
                negative_distances = self.vector_store.compute_distances(primary_vector, negative_vectors)
                hardest_negatives = np.argsort(negative_distances)[:self.i_near_others]

                negative_label_sources = [(label,source) for label in self.individuals for source in self.vector_store.get_sources_by_label(label)
                                    if label != individual]

                batch.extend([('nearby_negative', LabeledImageEmbedding(embedding=negative_vectors[i],
                                                                        label = negative_label_sources[i][0],
                                                                        source=negative_label_sources[i][1]) )for i in hardest_negatives])

            yield batch

    def __len__(self):
        photos_per_individual = 1 + self.i_far_same + self.i_near_others # primary + distant positives + nearby negatives
        return len(self.dataset) // (self.individuals_per_batch * photos_per_individual)