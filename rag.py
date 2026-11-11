"""ChromaDB layer (criterion 7): two collections, similarity search with metadata filters.

  transactions     : one doc per transaction (text + category), metadata = date/month/amount/category/merchant
  budget_guidance  : chunks of the .md files in data/budget_guidance/, split at '## ' headings

Embeddings (all local, free, no API quota). EMBED_BACKEND in .env:
  default               Chroma's built-in ONNX all-MiniLM-L6-v2 (small download on first use, no torch)
  sentence-transformers all-MiniLM-L6-v2 via the sentence-transformers package (needs torch)
  hash                  tiny keyword-hash embedding for offline tests only
"""
from __future__ import annotations
import os, re, zlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

ROOT = Path(__file__).resolve().parent
GUIDANCE_DIR = ROOT / "data" / "budget_guidance"
CHROMA_DIR = ROOT / "chroma_db"
TXN_COLLECTION, GUIDE_COLLECTION = "transactions", "budget_guidance"


class HashEmbedding(EmbeddingFunction[Documents]):
    """Bag-of-words hashing. Not semantic; exists so tests run offline."""
    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for doc in input:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in re.findall(r"[a-z0-9]+", doc.lower()):
                v[zlib.crc32(tok.encode()) % self.dim] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return out

    @staticmethod
    def name() -> str:
        return "hash-embedding"

    def get_config(self):
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding(**config)


def get_embedding_function():
    backend = os.getenv("EMBED_BACKEND", "default").strip().lower()
    if backend == "hash":
        return HashEmbedding()
    from chromadb.utils import embedding_functions as ef
    if backend == "sentence-transformers":
        return ef.SentenceTransformerEmbeddingFunction(model_name=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    return ef.DefaultEmbeddingFunction()


def _recreate(client, name, ef):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name, embedding_function=ef, metadata={"hnsw:space": "cosine"})


@dataclass
class RagContext:
    client: object
    ef: object
    txns: object
    guide: object

    @classmethod
    def create(cls, df: pd.DataFrame, client=None, ef=None) -> "RagContext":
        """Build both indexes. Transactions are re-indexed for every upload; guidance is (re)built from files."""
        client = client or chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = ef or get_embedding_function()
        ctx = cls(client, ef, _recreate(client, TXN_COLLECTION, ef), _recreate(client, GUIDE_COLLECTION, ef))
        ctx.index_transactions(df)
        ctx.index_guidance()
        return ctx

    # ---------- indexing ----------
    def index_transactions(self, df: pd.DataFrame) -> int:
        d = df.copy()
        ids = d["txn_id"].astype(str).tolist()
        docs = (d["text"] + " | category: " + d["category"].astype(str)).tolist()
        metas = [{"date": r.date.strftime("%Y-%m-%d"), "month": r.date.strftime("%Y-%m"), "amount": float(r.amount),
                  "category": str(r.category), "merchant": str(r.merchant), "description": str(r.description)}
                 for r in d.itertuples()]
        for s in range(0, len(ids), 100):
            self.txns.upsert(ids=ids[s:s + 100], documents=docs[s:s + 100], metadatas=metas[s:s + 100])
        return len(ids)

    def index_guidance(self) -> int:
        ids, docs, metas = [], [], []
        for f in sorted(GUIDANCE_DIR.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            title = text.split("\n", 1)[0].lstrip("# ").strip()
            for i, chunk in enumerate(re.split(r"\n(?=## )", text)[1:]):
                heading, _, body = chunk.partition("\n")
                ids.append(f"{f.stem}-{i}")
                docs.append(f"{heading.lstrip('# ').strip()}: {body.strip()}")
                metas.append({"source": f.name, "topic": title, "section": heading.lstrip("# ").strip()})
        if ids:
            self.guide.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    # ---------- retrieval ----------
    def search_transactions(self, query: str, n: int = 12, category: str | None = None, month: str | None = None) -> list[dict]:
        conds = []
        if category: conds.append({"category": category})
        if month: conds.append({"month": month})
        where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
        r = self.txns.query(query_texts=[query], n_results=min(n, self.txns.count()), where=where)
        return [{"id": i, "distance": round(d, 3), **m}
                for i, d, m in zip(r["ids"][0], r["distances"][0], r["metadatas"][0])]

    def search_guidance(self, query: str, n: int = 3) -> list[dict]:
        r = self.guide.query(query_texts=[query], n_results=min(n, self.guide.count()))
        return [{"id": i, "distance": round(d, 3), "text": t, **m}
                for i, d, t, m in zip(r["ids"][0], r["distances"][0], r["documents"][0], r["metadatas"][0])]
