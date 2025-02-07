from pathlib import Path
from typing import Iterable as Iter
from typing import List, Tuple, Dict, Any

import faiss
import numpy as np
from ikepono.labeledimageembedding import LabeledImageEmbedding


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension: int = dimension
        self.base_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
        self.index: faiss.IndexIDMap = faiss.IndexIDMap(self.base_index)
        self.vector_id_counter: int = 0
        self.dataset_id_to_vector_id: Dict[int, int] = {}
        self.vector_id_to_dataset_id: Dict[int, int] = {}
        self.vector_id_to_label: Dict[int, str] = {}
        self.vector_id_to_source: Dict[int, str] = {}
        self.label_to_dataset_ids: Dict[str, List[int]] = {}
        self.dataset_id_to_label: Dict[int, str] = {}
        self.source_to_dataset_id: Dict[str, int] = {}
        self.vector_store: Dict[int, np.ndarray] = {}  # Store vectors separately
        self._initialized: bool = False

    def _add_embedding(self, embedding: np.ndarray, label: str, source: Path, dataset_index : np.int64) -> None:
        assert isinstance(embedding, np.ndarray), f"Expected np.ndarray, got {type(embedding)}"
        assert isinstance(label, str), f"Expected str, got {type(label)}"
        assert isinstance(source, Path), f"Expected Path, got {type(source)}"
        assert isinstance(dataset_index, np.int64), f"Expected int64, got {type(dataset_index)}"

        vector_id: int = self.vector_id_counter
        # Note that FAISS is CPU, since FAISS GPU is overkill for mere hundreds of vectors
        vector_array = np.array([embedding]).astype('float32')
        if label not in self.label_to_dataset_ids:
            self.label_to_dataset_ids[label] = []
        self.label_to_dataset_ids[label].append(dataset_index)
        self.dataset_id_to_label[dataset_index] = label
        self.dataset_id_to_vector_id[dataset_index] = vector_id
        self.vector_id_to_dataset_id[vector_id] = dataset_index
        ids = np.array([vector_id]).astype('int64')
        assert vector_array.shape[0] == ids.shape[0], "Vector and ID shapes do not match: {} vs {}".format(vector_array.shape, ids.shape)
        self.index.add_with_ids(vector_array, ids)
        self.vector_id_to_label[vector_id] = label
        self.vector_id_to_source[vector_id] = source
        self.source_to_dataset_id[source] = dataset_index
        self.vector_store[vector_id] = embedding  # Store the vector
        self.vector_id_counter += 1
        return self.vector_id_counter - 1 # Return the ID of the added vector

    def initialize(self, livs : Iter[np.ndarray[LabeledImageEmbedding]]) -> None:
        assert self._initialized == False, "VectorStore already initialized"
        assert isinstance(livs[0].source, Path), f"Expected Path, got {type(livs[0].source)}"
        self._add_labeled_image_embeddings(livs)
        self._initialized = True

    def get_initialized(self) -> bool:
        return self._initialized


    def _add_labeled_image_embeddings(self, livs : Iter[LabeledImageEmbedding]) -> None:
        for liv in livs:
            self._add_embedding(liv.embedding, liv.label, liv.source, np.int64(liv.dataset_index))

    def update_or_add_vector(self, source: Path, new_vector: np.ndarray, new_label: str = None) -> None:
        assert isinstance(source, Path), f"Expected path, got {type(source)}"
        assert isinstance(new_vector, np.ndarray), f"Expected np.ndarray, got {type(new_vector)}"
        assert new_label is None or isinstance(new_label, str), f"Expected str or None, got {type(new_label)}"

        if source in self.source_to_dataset_id:
            dataset_id: int = self.source_to_dataset_id[source]
            vector_id: int = self.dataset_id_to_vector_id[dataset_id]
            old_label: str = self.vector_id_to_label[vector_id]

            # Update vector in FAISS and our separate storage
            self.index.remove_ids(np.array([vector_id]))
            self.index.add_with_ids(np.array([new_vector]).astype('float32'), np.array([vector_id]))
            self.vector_store[vector_id] = new_vector

            # Update label if necessary
            if new_label is not None and new_label != old_label:
                self.vector_id_to_label[vector_id] = new_label
                self.label_to_dataset_ids[old_label].remove(dataset_id)
                self.label_to_dataset_ids.setdefault(new_label, []).append(dataset_id)
        else:
            assert False, "I think I can remove this block and change fn to update only"
            # Add the new vector if it doesn't exist
            vector_id = self._add_embedding(new_vector, new_label, source)
            dataset_id = self.vector_id_to_dataset_id[vector_id]
            self.source_to_dataset_id[source] = dataset_id
            self.vector_id_to_source[vector_id] = source
            self.vector_id_to_label[vector_id] = new_label


    def vector_for_source(self, source: str) -> np.ndarray:
        if source in self.source_to_dataset_id:
            dataset_id: int = self.source_to_dataset_id[source]
            vector_id: int = self.dataset_id_to_vector_id[dataset_id]
            return self.vector_store[vector_id]
        else:
            raise ValueError(f"Source '{source}' not found in the store.")

    def vector_for_dataset_id(self, dataset_id: int) -> np.ndarray:
        vector_id = self.dataset_id_to_vector_id[dataset_id]
        if vector_id in self.vector_store:
            return self.vector_store[vector_id]
        else:
            raise ValueError(f"ID '{vector_id}' not found in the store.")

    def vectors_for_dataset_ids(self, dataset_ids: List[int]) -> np.ndarray:
        vector_ids = [self.dataset_id_to_vector_id[dataset_id] for dataset_id in dataset_ids]
        vectors = [self.vector_store[vector_id] for vector_id in vector_ids]
        return np.array(vectors)

    def vectors_for_label(self, label: Any) -> np.ndarray:
        if label in self.label_to_dataset_ids:
            dataset_ids = self.label_to_dataset_ids[label]
            vector_ids = self.vector_indices_for_dataset_indices(dataset_ids)
            vectors = [self.vector_store[vector_id] for vector_id in vector_ids]
            return np.array(vectors)
        else:
            return np.ndarray([])

    def sources_for_label(self, label: Any) -> np.ndarray:
        if label in self.label_to_dataset_ids:
            dataset_ids = self.label_to_dataset_ids[label]
            vector_ids = [self.dataset_id_to_vector_id[dataset_id] for dataset_id in dataset_ids]
            sources = [self.vector_id_to_source[vector_id] for vector_id in vector_ids]
            return np.array(sources)
        else:
            return np.ndarray([])

    def vector_indices_for_dataset_indices(self, dataset_indices: List[int]) -> List[int]:
        return [self.dataset_id_to_vector_id[dataset_id] for dataset_id in dataset_indices]

    def dataset_indices_for_vector_indices(self, vector_indices: List[int]) -> List[int]:
        return [self.vector_id_to_dataset_id[vector_id] for vector_id in vector_indices]

    def dataset_ids_for_label(self, label: Any) -> np.ndarray:
        if label in self.label_to_dataset_ids:
            return np.array(self.label_to_dataset_ids[label])
        else:
            return np.ndarray([])

    def dataset_id_for_source(self, source: str) -> int:
        if source in self.source_to_dataset_id:
            return self.source_to_dataset_id[source]

    def label_for_dataset_id(self, dataset_id: int) -> str:
        vector_id = self.dataset_id_to_vector_id[dataset_id]
        if vector_id in self.vector_id_to_label:
            return self.vector_id_to_label[vector_id]
        else:
            raise ValueError(f"ID '{vector_id}' not found in the store.")

    def label_for_source(self, source: str) -> str:
        if source in self.source_to_dataset_id:
            dataset_id: int = self.source_to_dataset_id[source]
            vector_id: int = self.dataset_id_to_vector_id[dataset_id]
            return self.vector_id_to_label[vector_id]
        else:
            raise ValueError(f"Source '{source}' not found in the store.")

    def all_vectors(self) -> np.ndarray:
        if len(self.vector_store) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        else:
            return np.stack(list(self.vector_store.values()))

    def all_labels(self) -> np.ndarray:
        if len(self.label_to_dataset_ids) == 0:
            return np.ndarray([])
        else:
            return np.stack(list(self.label_to_dataset_ids.keys()))

    def all_sources(self) -> np.ndarray:
        if len(self.source_to_dataset_id) == 0:
            return np.ndarray([])
        else:
            return np.stack(list(self.source_to_dataset_id.keys()))

    def all_vector_ids(self) -> np.ndarray:
        return np.array(list(self.vector_store.keys()))

    def all_dataset_ids(self) -> np.ndarray:
        return np.array(list(self.dataset_id_to_vector_id.keys()))


    @staticmethod
    def distances(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        return np.linalg.norm(vectors - query_vector, axis=1)

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, Any, str]]:
        all_vectors = self.all_vectors()
        distances = self.distances(query_vector, all_vectors)
        indices = np.argsort(distances)[:k]
        return [(distances[i], self.vector_id_to_label[i], self.vector_id_to_source[i]) for i in indices]