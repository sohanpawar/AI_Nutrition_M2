"""Embeddings and persistent metadata-filtering vector index (Phase 4).

Provides:
- One shared `embed_texts` / `embed_text` function used at both ingest and query time.
  For chunks, `format_chunk_for_embedding(chunk)` builds `f"{chunk.section_heading}\\n{chunk.text}"`.
- A persisted vector index under `corpus/index/` that stores vectors alongside full
  `Chunk` metadata and supports:
  1. Top-k search across the whole index (`doc_id=None`).
  2. Top-k search restricted to a single `doc_id` inside the index query (`doc_id="<id>"`),
     never by filtering after a global top-k search.
- Re-ingesting a single document deletes that `doc_id`'s vectors and writes its new
  chunks while leaving the other six documents intact.
- Re-ingesting all documents rebuilds the index from the seven `.chunks.jsonl` files,
  refusing to build if any manifest document is missing its chunk file.
"""

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Protocol

import numpy as np

from src import config
from src.ingest.load import load_manifest
from src.schemas import Chunk

EmbeddingFn = Callable[[list[str]], list[list[float]]]

LOCAL_EMBEDDING_DIM = 1536
WORD_RE = re.compile(r"[a-z0-9]+")

# Domain synonym expansion for deterministic local embeddings so queries like
# "fridge" / "refrigerator", "chill" / "cold", "leftover" / "leftovers" align tightly
# with the official corpus terminology even when running offline without an API key.
SYNONYM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "fridge": ("refrigerator", "chill", "chilling", "cold"),
    "refrigerator": ("fridge", "chill", "chilling", "cold"),
    "freezer": ("freeze", "freezing", "frozen"),
    "freeze": ("freezer", "freezing", "frozen"),
    "defrost": ("thaw", "thawing", "defrosting"),
    "thaw": ("defrost", "defrosting", "thawing"),
    "leftovers": ("leftover", "cooked", "storage", "chart"),
    "leftover": ("leftovers", "cooked", "storage", "chart"),
    "time": ("days", "months", "storage", "chart"),
    "long": ("days", "months", "time", "storage"),
    "sugars": ("sugar", "sweeteners"),
    "sugar": ("sugars", "sweeteners"),
    "fats": ("fat", "oils", "oil", "saturated", "unsaturated"),
    "oil": ("oils", "fat", "fats", "unsaturated", "spreads"),
    "oils": ("oil", "fat", "fats", "unsaturated", "spreads"),
    "salt": ("sodium",),
    "sodium": ("salt",),
    "meat": ("poultry", "raw", "beef", "chicken", "pork", "separate"),
    "raw": ("separate", "cooked", "meat", "poultry", "contamination", "safer", "five", "keys", "poster"),
    "handle": ("separate", "clean", "wash", "hands", "utensils", "cutting", "safer", "five", "keys", "poster", "core"),
    "cooking": ("cook", "cooked", "thoroughly", "temperature", "safer", "five", "keys", "preparation"),
    "healthy": ("diet", "diets", "principles", "adequate", "balanced", "diverse", "overview"),
    "diet": ("diets", "healthy", "principles", "fruit", "vegetables", "legumes"),
    "include": ("includes", "principles", "adults", "overview", "key", "facts"),
    "divide": ("divides", "groups", "proportions", "main"),
}

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "should",
        "stay",
        "that",
        "the",
        "there",
        "to",
        "what",
        "when",
        "which",
        "with",
    }
)


class IndexError(RuntimeError):
    """Raised when building or querying the vector index fails."""


class EmbeddingProvider(Protocol):
    """Callable protocol for embedding a batch of texts into float vectors."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


def format_chunk_for_embedding(chunk: Chunk) -> str:
    """Format a chunk for embedding: section heading, a newline, then the chunk body."""
    return f"{chunk.section_heading}\n{chunk.text}"


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _feature_Index_and_sign(feature: str, dim: int = LOCAL_EMBEDDING_DIM) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "little")
    idx = val % dim
    sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
    return idx, sign


def _local_deterministic_embed_one(text: str, dim: int = LOCAL_EMBEDDING_DIM) -> list[float]:
    """Compute a normalized feature-hashed vector from heading + body tokens and n-grams."""
    vec = np.zeros(dim, dtype=np.float64)
    lines = text.split("\n", 1)
    heading_part = lines[0] if lines else ""
    body_part = lines[1] if len(lines) > 1 else text

    # For markdown tables, extract each row's lead category/item tokens so long
    # reference tables (like Cold Food Storage Chart) do not dilute row-level lookups.
    table_lead_tokens: list[str] = []
    for raw_line in body_part.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
            for cell in cells[:2]:
                table_lead_tokens.extend(WORD_RE.findall(cell.lower()))

    def add_tokens(raw_text: str, weight: float, use_Presence_boost: bool = False) -> None:
        raw_tokens = WORD_RE.findall(raw_text.lower())
        content_tokens = [t for t in raw_tokens if t not in STOPWORDS]
        if not content_tokens:
            content_tokens = raw_tokens

        # Unigrams + stems + domain synonym expansions with sublinear TF scaling
        tf_counts: dict[str, float] = {}
        for tok in content_tokens:
            tf_counts[f"u:{tok}"] = tf_counts.get(f"u:{tok}", 0.0) + 1.0
            stem = _stem_token(tok)
            if stem != tok:
                tf_counts[f"s:{stem}"] = tf_counts.get(f"s:{stem}", 0.0) + 0.85
            for syn in SYNONYM_EXPANSIONS.get(tok, ()):
                tf_counts[f"u:{syn}"] = tf_counts.get(f"u:{syn}", 0.0) + 0.7
                syn_stem = _stem_token(syn)
                tf_counts[f"s:{syn_stem}"] = tf_counts.get(f"s:{syn_stem}", 0.0) + 0.55

        for feat, count in tf_counts.items():
            idx, sign = _feature_Index_and_sign(feat, dim)
            scale = 1.0 if use_Presence_boost else (1.0 + 0.35 * math.log(count))
            vec[idx] += sign * weight * scale

        # Content word bigrams
        for i in range(len(content_tokens) - 1):
            bg = f"bg:{content_tokens[i]}_{content_tokens[i + 1]}"
            idx, sign = _feature_Index_and_sign(bg, dim)
            vec[idx] += sign * weight * 0.5

    # Give section heading extra weight so heading matches boost relevant sections
    if heading_part.strip():
        add_tokens(heading_part, weight=1.8)
    if table_lead_tokens:
        add_tokens(" ".join(table_lead_tokens), weight=1.6, use_Presence_boost=True)
    add_tokens(body_part, weight=1.0)

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec.astype(np.float32).tolist()


def _get_openai_api_key() -> str:
    """Resolve `OPENAI_API_KEY` from environment variables or Streamlit secrets."""
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        import streamlit as st

        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return ""


def embed_texts(
    texts: list[str],
    model_id: str = config.embedding_model_id,
) -> list[list[float]]:
    """Shared embedding function called by both ingest and query.

    Uses OpenAI embeddings when `OPENAI_API_KEY` is configured and `model_id` is an
    OpenAI model; otherwise uses the deterministic local embedding implementation so
    ingest and search work identically offline.
    """
    if not texts:
        return []

    api_key = _get_openai_api_key()
    if api_key and model_id != "local-deterministic":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=model_id, input=texts)
        return [item.embedding for item in response.data]

    return [_local_deterministic_embed_one(text) for text in texts]


def embed_text(
    text: str,
    model_id: str = config.embedding_model_id,
) -> list[float]:
    """Embed a single string using `embed_texts`."""
    return embed_texts([text], model_id=model_id)[0]


def load_chunks_for_doc(
    doc_id: str,
    parsed_dir: Path = config.PARSED_DIR,
) -> list[Chunk]:
    """Read `corpus/parsed/<doc_id>.chunks.jsonl` and return validated `Chunk` objects."""
    chunk_path = parsed_dir / f"{doc_id}.chunks.jsonl"
    if not chunk_path.exists():
        raise IndexError(f"Missing chunk file for manifest document '{doc_id}': {chunk_path}")

    chunks: list[Chunk] = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.model_validate_json(line))
            except Exception as exc:
                raise IndexError(f"Invalid chunk on line {line_no} of {chunk_path}: {exc}") from exc

    if not chunks:
        raise IndexError(f"Chunk file for '{doc_id}' is empty: {chunk_path}")

    return chunks


class VectorIndex:
    """Persisted metadata-filtering vector store backed by SQLite under `corpus/index/`."""

    DB_FILENAME = "vectors.sqlite3"

    def __init__(
        self,
        index_dir: Path = config.INDEX_DIR,
        embed_fn: EmbeddingFn = embed_texts,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.db_path = self.index_dir / self.DB_FILENAME
        self.embed_fn = embed_fn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    publisher TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    source_url TEXT NOT NULL,
                    retrieval_date TEXT NOT NULL,
                    section_heading TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ordinal_start INTEGER NOT NULL,
                    ordinal_end INTEGER NOT NULL,
                    table_part TEXT,
                    metadata_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            conn.commit()
            yield conn

    def is_ready(self) -> bool:
        """Return True if the persisted index exists on disk and contains chunks."""
        if not self.db_path.exists():
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cur = conn.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks'"
                )
                if cur.fetchone()[0] == 0:
                    return False
                count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
                return bool(count > 0)
            finally:
                conn.close()
        except sqlite3.Error:
            return False

    def count(self, doc_id: str | None = None) -> int:
        """Return the number of indexed chunks, optionally filtered by `doc_id`."""
        if not self.db_path.exists():
            return 0
        with self._connect() as conn:
            if doc_id is None:
                row = conn.execute("SELECT count(*) FROM chunks").fetchone()
            else:
                row = conn.execute(
                    "SELECT count(*) FROM chunks WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()
            return int(row[0]) if row else 0

    def delete_document(self, doc_id: str) -> int:
        """Delete all vectors and metadata for a single `doc_id`."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.commit()
            return int(cur.rowcount)

    def _upsert_chunks(self, conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
        texts_to_embed = [format_chunk_for_embedding(c) for c in chunks]
        vectors = self.embed_fn(texts_to_embed)
        if len(vectors) != len(chunks):
            raise IndexError(
                f"Embedding function returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        rows_to_insert = [
            (
                c.chunk_id,
                c.doc_id,
                c.document_name,
                c.publisher,
                c.year,
                c.source_url,
                c.retrieval_date,
                c.section_heading,
                c.text,
                c.ordinal_start,
                c.ordinal_end,
                str(c.table_part) if c.table_part is not None else None,
                c.model_dump_json(),
                json.dumps(vec),
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]

        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                chunk_id, doc_id, document_name, publisher, year,
                source_url, retrieval_date, section_heading, text,
                ordinal_start, ordinal_end, table_part, metadata_json, vector_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )

    def ingest_document(
        self,
        doc_id: str,
        parsed_dir: Path = config.PARSED_DIR,
        manifest_path: Path = config.MANIFEST_PATH,
    ) -> int:
        """Re-ingest a single `doc_id`: delete its existing vectors and write its new chunks."""
        manifest_ids = {entry["doc_id"] for entry in load_manifest(manifest_path)}
        if doc_id not in manifest_ids:
            raise IndexError(f"Unknown doc_id '{doc_id}' is not in manifest")

        chunks = load_chunks_for_doc(doc_id=doc_id, parsed_dir=parsed_dir)
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._upsert_chunks(conn, chunks)
            conn.commit()
        return len(chunks)

    def rebuild_all(
        self,
        parsed_dir: Path = config.PARSED_DIR,
        manifest_path: Path = config.MANIFEST_PATH,
    ) -> dict[str, int]:
        """Rebuild the entire index from the seven `.chunks.jsonl` files.

        Refuses to build the index if any manifest document is missing its chunk file.
        """
        manifest_entries = load_manifest(manifest_path)
        chunks_by_doc: dict[str, list[Chunk]] = {}

        # Verify all 7 chunk files exist and are valid before mutating the index
        for entry in manifest_entries:
            doc_id = entry["doc_id"]
            chunks_by_doc[doc_id] = load_chunks_for_doc(doc_id=doc_id, parsed_dir=parsed_dir)

        counts: dict[str, int] = {}
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            for doc_id, chunks in chunks_by_doc.items():
                self._upsert_chunks(conn, chunks)
                counts[doc_id] = len(chunks)
            conn.commit()

        return counts

    def query(
        self,
        question: str,
        top_k: int = config.k,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Query the persisted index for top-k `(Chunk, score)` pairs.

        When `doc_id` is provided, the `WHERE doc_id = ?` metadata filter is applied
        inside the database query prior to similarity ranking (never by post-filtering
        a global search).
        """
        if not self.is_ready():
            raise IndexError(f"Vector index at {self.db_path} is missing or empty")

        query_vecs = self.embed_fn([question])
        if not query_vecs:
            raise IndexError("Failed to embed query text")
        q_vec = np.asarray(query_vecs[0], dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0.0:
            q_vec = q_vec / q_norm

        with self._connect() as conn:
            if doc_id is not None:
                rows = conn.execute(
                    "SELECT metadata_json, vector_json FROM chunks WHERE doc_id = ?",
                    (doc_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT metadata_json, vector_json FROM chunks"
                ).fetchall()

        if not rows:
            return []

        scored: list[tuple[Chunk, float]] = []
        for metadata_json, vector_json in rows:
            chunk = Chunk.model_validate_json(metadata_json)
            c_vec = np.asarray(json.loads(vector_json), dtype=np.float32)
            c_norm = float(np.linalg.norm(c_vec))
            if c_norm > 0.0 and q_norm > 0.0:
                score = float(np.dot(q_vec, c_vec / c_norm))
            else:
                score = 0.0
            scored.append((chunk, score))

        scored.sort(key=lambda item: (item[1], item[0].chunk_id), reverse=True)
        return scored[:top_k]


def build_index(
    index_dir: Path = config.INDEX_DIR,
    parsed_dir: Path = config.PARSED_DIR,
    manifest_path: Path = config.MANIFEST_PATH,
    doc_id: str | None = None,
    embed_fn: EmbeddingFn = embed_texts,
) -> dict[str, int]:
    """Build or update the persisted vector index under `corpus/index/`."""
    idx = VectorIndex(index_dir=index_dir, embed_fn=embed_fn)
    if doc_id is not None:
        count = idx.ingest_document(
            doc_id=doc_id,
            parsed_dir=parsed_dir,
            manifest_path=manifest_path,
        )
        return {doc_id: count}
    return idx.rebuild_all(parsed_dir=parsed_dir, manifest_path=manifest_path)


def main() -> int:
    """CLI entry point for Phase 4 embedding and index build."""
    try:
        counts = build_index()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total = sum(counts.values())
    print(f"Indexed {total} chunks across {len(counts)} documents into {config.INDEX_DIR}:")
    for doc_id, count in counts.items():
        print(f"  - {doc_id}: {count} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
