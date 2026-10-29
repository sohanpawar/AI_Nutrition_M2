"""Citation verification and claim filter (Phase 6).

Verifies every generated claim deterministically in code without calling a model:
- Requires non-empty claim `text` and a non-empty `chunk_ids` list.
- Every `chunk_id` must have been in this document's retrieval set for the
  request and must belong to that section's `doc_id`.
- Invalid claims are dropped immediately (never sent back to a model).
- If a document loses every claim (or abstained), its section is dropped.
- Citation metadata (`document_name`, `publisher`, `year`, `source_url`, and
  `section_heading`) is copied strictly from the retrieved `Chunk` metadata,
  ignoring anything the model text might state. `source_url` is that document's
  official publication page, stored on the chunk at ingest.
"""

from collections.abc import Sequence

from src.schemas import Chunk, Claim, DocumentGeneration, Section


def verify_document_claims(
    generation: DocumentGeneration,
    retrieved_chunks: Sequence[Chunk],
) -> Section | None:
    """Validate one `DocumentGeneration` against `retrieved_chunks` and return a `Section` or `None`."""
    if generation.abstain or not generation.claims:
        return None

    target_doc_id = generation.doc_id
    doc_chunks_by_id: dict[str, Chunk] = {
        chunk.chunk_id: chunk
        for chunk in retrieved_chunks
        if chunk.doc_id == target_doc_id
    }
    if not doc_chunks_by_id:
        return None

    valid_claims: list[Claim] = []
    for raw_claim in generation.claims:
        text = raw_claim.text.strip()
        if not text:
            continue

        if not raw_claim.chunk_ids:
            continue

        cleaned_ids = [cid.strip() for cid in raw_claim.chunk_ids if cid and cid.strip()]
        if not cleaned_ids or len(cleaned_ids) != len(raw_claim.chunk_ids):
            continue

        # Every cited chunk_id must belong to this document's retrieved chunk set
        if not all(cid in doc_chunks_by_id for cid in cleaned_ids):
            continue

        # Deduplicate while preserving order
        seen_ids: list[str] = []
        for cid in cleaned_ids:
            if cid not in seen_ids:
                seen_ids.append(cid)

        # Read section_heading strictly from chunk metadata
        primary_chunk = doc_chunks_by_id[seen_ids[0]]
        valid_claims.append(
            Claim(
                text=text,
                chunk_ids=seen_ids,
                section_heading=primary_chunk.section_heading,
            )
        )

    if not valid_claims:
        return None

    # Read document_name, publisher, year, and source_url strictly from chunk metadata
    ref_chunk = next(iter(doc_chunks_by_id.values()))
    return Section(
        doc_id=ref_chunk.doc_id,
        document_name=ref_chunk.document_name,
        publisher=ref_chunk.publisher,
        year=ref_chunk.year,
        source_url=ref_chunk.source_url,
        claims=valid_claims,
    )


def filter_citations(
    generations: Sequence[DocumentGeneration],
    retrieved_chunks: Sequence[Chunk],
) -> list[Section]:
    """Filter a list of per-document generations against `retrieved_chunks`."""
    surviving_sections: list[Section] = []
    for gen in generations:
        section = verify_document_claims(gen, retrieved_chunks)
        if section is not None:
            surviving_sections.append(section)
    return surviving_sections
