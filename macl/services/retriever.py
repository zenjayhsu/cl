from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalHit:
    doc_id: str
    score: float
    metadata: Dict[str, object]


@dataclass
class TfidfRetriever:
    documents: Dict[str, str] = field(default_factory=dict)
    metadatas: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def add_document(self, doc_id: str, document: str, metadata: Dict[str, object]) -> None:
        self.documents[doc_id] = document
        self.metadatas[doc_id] = metadata

    def delete_document(self, doc_id: str) -> None:
        self.documents.pop(doc_id, None)
        self.metadatas.pop(doc_id, None)

    def update_document(self, doc_id: str, document: str, metadata: Dict[str, object]) -> None:
        self.add_document(doc_id=doc_id, document=document, metadata=metadata)

    def search(self, query: str, k: int = 5) -> List[RetrievalHit]:
        if not self.documents:
            return []
        ids = list(self.documents.keys())
        corpus = [self.documents[doc_id] for doc_id in ids]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(corpus + [query])
        doc_matrix = matrix[:-1]
        query_vec = matrix[-1]
        similarities = cosine_similarity(query_vec, doc_matrix)[0]
        order = np.argsort(similarities)[::-1][:k]
        hits: List[RetrievalHit] = []
        for index in order:
            doc_id = ids[index]
            hits.append(RetrievalHit(doc_id=doc_id, score=float(similarities[index]), metadata=self.metadatas[doc_id]))
        return hits
