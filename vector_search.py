"""
Module 3: Vector Search Engine (dual-backend)

Backends:
  1. 'transformer' - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
     - Semantic understanding, ideal for production
     - For Korean-specialized: switch to jhgan/ko-sroberta-multitask (768-dim)

  2. 'tfidf' - scikit-learn TfidfVectorizer with Korean-aware tokenization
     - Keyword-based, no external downloads needed, works offline

Auto-fallback: if transformer backend fails to load, uses tfidf.
Both backends expose the same interface.
"""

import numpy as np
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def korean_tokenize(text: str) -> list:
    """Korean-friendly tokenization: Korean words, English words, or numbers."""
    tokens = re.findall(r"[가-힣]{2,}|[a-zA-Z]{2,}|\d+", text)
    return [t.lower() for t in tokens]


class VectorSearchEngine:
    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        index_dir: str = "vector_store",
    ):
        self.requested_backend = backend
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)

        self.backend: Optional[str] = None
        self.model = None
        self.index: Optional[faiss.Index] = None
        self.matrix = None
        self.documents: List[Dict] = []
        self.dim: int = 0

        self._init_backend()

    def _init_backend(self):
        if self.requested_backend in ("auto", "transformer"):
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[backend] Attempting transformer: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.dim = self.model.get_sentence_embedding_dimension()
                self.backend = "transformer"
                print(f"[backend] ✓ transformer loaded (dim={self.dim})")
                return
            except Exception as e:
                if self.requested_backend == "transformer":
                    raise
                print(f"[backend] ✗ transformer unavailable ({type(e).__name__})")
                print(f"[backend] → falling back to tfidf")

        self.model = TfidfVectorizer(
            tokenizer=korean_tokenize, token_pattern=None,
            min_df=1, max_df=0.95, sublinear_tf=True,
        )
        self.backend = "tfidf"
        print(f"[backend] ✓ tfidf initialized")

    def build_index(self, documents: List[Dict], text_field: str = "summary"):
        self.documents = documents
        texts = [doc.get(text_field) or doc.get("raw_text", "")[:800]
                 for doc in documents]

        if self.backend == "transformer":
            embeddings = self.model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)
            print(f"[index] FAISS: {self.index.ntotal} vectors, dim={self.dim}")
        else:
            self.matrix = self.model.fit_transform(texts)
            self.dim = self.matrix.shape[1]
            print(f"[index] TF-IDF: {self.matrix.shape[0]} vectors, vocab={self.dim}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.backend == "transformer":
            q_emb = self.model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32)
            scores, indices = self.index.search(q_emb, top_k)
            return [
                {"doc": self.documents[idx], "score": float(score)}
                for score, idx in zip(scores[0], indices[0])
                if 0 <= idx < len(self.documents)
            ]
        else:
            q_vec = self.model.transform([query])
            sims = cosine_similarity(q_vec, self.matrix)[0]
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [
                {"doc": self.documents[int(i)], "score": float(sims[i])}
                for i in top_idx if sims[i] > 0
            ]

    def hybrid_search(self, query: str, top_k: int = 5, keyword_boost: float = 0.2) -> List[Dict]:
        semantic_results = self.search(query, top_k=min(top_k * 2, max(len(self.documents), 1)))
        query_tokens = set(korean_tokenize(query))
        for r in semantic_results:
            doc_text = (r["doc"].get("summary", "") + " " +
                        r["doc"].get("raw_text", "")[:800])
            doc_tokens = set(korean_tokenize(doc_text))
            overlap = (len(query_tokens & doc_tokens) / len(query_tokens)
                       if query_tokens else 0)
            r["hybrid_score"] = r["score"] + keyword_boost * overlap
        semantic_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return semantic_results[:top_k]

    def save(self, name: str = "main"):
        meta = {"backend": self.backend, "documents": self.documents}
        if self.backend == "transformer":
            faiss.write_index(self.index, str(self.index_dir / f"{name}.faiss"))
        else:
            with open(self.index_dir / f"{name}.tfidf.pkl", "wb") as f:
                pickle.dump({"vectorizer": self.model, "matrix": self.matrix}, f)
        with open(self.index_dir / f"{name}.meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        print(f"[save] → {self.index_dir}/{name}.*  ({self.backend})")

    def load(self, name: str = "main"):
        with open(self.index_dir / f"{name}.meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.documents = meta["documents"]
        self.backend = meta["backend"]
        if self.backend == "transformer":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.index = faiss.read_index(str(self.index_dir / f"{name}.faiss"))
        else:
            with open(self.index_dir / f"{name}.tfidf.pkl", "rb") as f:
                d = pickle.load(f)
            self.model = d["vectorizer"]
            self.matrix = d["matrix"]
        print(f"[load] ← {name} ({self.backend}, {len(self.documents)} docs)")


if __name__ == "__main__":
    docs = [
        {"id": 1, "summary": "자문회의 비용 지출 - 빅데이터센터 운영 전문가 자문"},
        {"id": 2, "summary": "회의 후 식대 지출 - 내부 회의 식사 비용"},
        {"id": 3, "summary": "세미나 발표비 및 교통비 지출"},
        {"id": 4, "summary": "인쇄비용 지출 - 보고서 출력 및 제본"},
    ]
    engine = VectorSearchEngine(backend="auto")
    engine.build_index(docs)
    for q in ["자문회의", "식대", "인쇄", "빅데이터 분석 회의"]:
        print(f"\n=== Query: {q} ===")
        for r in engine.hybrid_search(q, top_k=3):
            print(f"  [sem={r['score']:.3f} hyb={r['hybrid_score']:.3f}] {r['doc']['summary']}")
