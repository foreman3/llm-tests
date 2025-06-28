import os
import pickle
from typing import List, Iterable

import faiss
import numpy as np

class VectorStore:
    """Simple FAISS-based vector store for embeddings."""

    def __init__(self, dim: int, store_path: str | None = None):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.ids: List[str] = []
        self.store_path = store_path
        if store_path and os.path.exists(store_path):
            with open(store_path, "rb") as f:
                self.ids, index_data = pickle.load(f)
                self.index = faiss.deserialize_index(index_data)

    def add(self, ids: Iterable[str], embeddings: Iterable[List[float]]):
        vecs = np.array(list(embeddings)).astype('float32')
        self.index.add(vecs)
        self.ids.extend(list(ids))
        self._save()

    def query(self, embedding: List[float], k: int = 5):
        vec = np.array([embedding]).astype('float32')
        distances, idx = self.index.search(vec, k)
        results = []
        for i, dist in zip(idx[0], distances[0]):
            if i < len(self.ids):
                results.append((self.ids[i], float(dist)))
        return results

    def _save(self):
        if not self.store_path:
            return
        data = faiss.serialize_index(self.index)
        with open(self.store_path, "wb") as f:
            pickle.dump((self.ids, data), f)
