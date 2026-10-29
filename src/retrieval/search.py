"""Vector search with score floor and per-document caps (Phase 5).

Resolves any document filter or alias, queries the persisted vector index,
drops chunks below `config.score_floor`, applies the total cap `config.k` and
the per-document cap `config.m` (so `kitchen-companion` cannot crowd out shorter
documents), and returns `(surviving_chunks, searched_records)` without calling
any chat model.
"""

from dataclasses import dataclass
from pathlib import Path

from src import config
from src.ingest.embed_index import VectorIndex
from src.ingest.load import load_manifest
from src.retrieval.resolve import resolve_document
from src.schemas import Chunk, SearchedDocument


@dataclass(frozen=True)
class RetrievalResult:
    """Chunks passing the score floor and caps, plus the record of searched documents."""

    chunks: list[Chunk]
    searched: list[SearchedDocument]
    resolved_doc_id: str | None = None


def build_search_record(
    resolved_doc_id: str | None,
    manifest_path: Path = config.MANIFEST_PATH,
) -> list[SearchedDocument]:
    """Return the `SearchedDocument` list for either one filtered `doc_id` or all 7 documents."""
    entries = load_manifest(manifest_path)
    if resolved_doc_id is not None:
        return [
            SearchedDocument(
                doc_id=entry["doc_id"],
                document_name=entry["document_name"],
                publisher=entry["publisher"],
            )
            for entry in entries
            if entry["doc_id"] == resolved_doc_id
        ]

    return [
        SearchedDocument(
            doc_id=entry["doc_id"],
            document_name=entry["document_name"],
            publisher=entry["publisher"],
        )
        for entry in entries
    ]


def retrieve(
    question: str,
    document_filter: str | None = None,
    index: VectorIndex | None = None,
    manifest_path: Path = config.MANIFEST_PATH,
    score_floor: float | None = None,
    k: int | None = None,
    m: int | None = None,
) -> RetrievalResult:
    """Resolve optional document filter/alias and retrieve chunks above `score_floor`.

    - Never hard-codes `score_floor`, `k`, or `m`; defaults are read from `src.config`.
    - Raises `InvalidDocumentFilterError` before querying the index if an explicit
      `document_filter` does not match any manifest `doc_id` or alias.
    - Applies `m` per-document cap before filling up to `k` total chunks.
    """
    floor = config.score_floor if score_floor is None else score_floor
    max_total = config.k if k is None else k
    max_per_doc = config.m if m is None else m

    resolved_doc_id = resolve_document(
        question=question,
        document_filter=document_filter,
        manifest_path=manifest_path,
    )
    searched = build_search_record(resolved_doc_id, manifest_path=manifest_path)

    vec_index = index if index is not None else VectorIndex(index_dir=config.INDEX_DIR)

    # Fetch candidate pool from the index so per-document cap `m` can prevent
    # `kitchen-companion` (105 chunks) from crowding out shorter documents before `k` is reached.
    candidate_limit = max_total if resolved_doc_id is not None else max(vec_index.count(), max_total)
    scored_candidates = vec_index.query(
        question=question,
        top_k=candidate_limit,
        doc_id=resolved_doc_id,
    )

    surviving: list[Chunk] = []
    per_doc_counts: dict[str, int] = {}

    for chunk, score in scored_candidates:
        if score < floor:
            continue
        doc_count = per_doc_counts.get(chunk.doc_id, 0)
        if doc_count >= max_per_doc:
            continue
        surviving.append(chunk)
        per_doc_counts[chunk.doc_id] = doc_count + 1
        if len(surviving) >= max_total:
            break

    return RetrievalResult(
        chunks=surviving,
        searched=searched,
        resolved_doc_id=resolved_doc_id,
    )
