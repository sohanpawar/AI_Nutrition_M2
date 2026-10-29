"""Deterministic multi-document section assembler and response envelope builder (Phase 6).

Orders surviving sections in `corpus/manifest.yaml` order, renders them one
after another without calling a language model and without adding any
synthesizing prose (such as "both authorities agree" or "the guidelines say"),
and constructs the unified `AskResponse` envelope for `grounded_answer` or
`not_in_corpus`.
"""

from collections.abc import Sequence
from pathlib import Path

from src import config
from src.answer.citations import filter_citations
from src.ingest.load import load_manifest
from src.schemas import (
    AskResponse,
    Chunk,
    DocumentGeneration,
    SearchedDocument,
    Section,
)

NOT_IN_CORPUS_PREFIX = (
    "The official dietary guidance corpus does not cover this question."
)


def get_manifest_order(manifest_path: Path = config.MANIFEST_PATH) -> list[str]:
    """Return the ordered list of `doc_id`s from `corpus/manifest.yaml`."""
    return [entry["doc_id"] for entry in load_manifest(manifest_path)]


def order_sections_by_manifest(
    sections: Sequence[Section],
    manifest_path: Path = config.MANIFEST_PATH,
) -> list[Section]:
    """Sort surviving `Section`s strictly according to `corpus/manifest.yaml` order."""
    manifest_ids = get_manifest_order(manifest_path)
    rank_by_id = {doc_id: idx for idx, doc_id in enumerate(manifest_ids)}
    rank_by_name = {
        entry["document_name"]: idx
        for idx, entry in enumerate(load_manifest(manifest_path))
    }

    def section_rank(sec: Section) -> int:
        if sec.doc_id and sec.doc_id in rank_by_id:
            return rank_by_id[sec.doc_id]
        return rank_by_name.get(sec.document_name, len(manifest_ids))

    return sorted(sections, key=section_rank)


def render_section(section: Section) -> str:
    """Render a single `Section` into the architecture's text block format."""
    header_line = f"{section.document_name} — {section.publisher}, {section.year}"
    url_line = section.source_url
    claim_lines = [
        f"- {claim.text} ({claim.section_heading})"
        if claim.section_heading
        else f"- {claim.text}"
        for claim in section.claims
    ]
    return "\n".join([header_line, url_line, *claim_lines])


def render_sections(
    sections: Sequence[Section],
    manifest_path: Path = config.MANIFEST_PATH,
) -> str:
    """Order `sections` by manifest order and concatenate their rendered blocks with a blank line."""
    ordered = order_sections_by_manifest(sections, manifest_path=manifest_path)
    return "\n\n".join(render_section(sec) for sec in ordered)


def format_not_in_corpus_message(searched: Sequence[SearchedDocument]) -> str:
    """Build the corpus-refusal message naming the documents that were searched."""
    if not searched:
        return NOT_IN_CORPUS_PREFIX
    searched_names = ", ".join(
        f"{doc.document_name} ({doc.publisher})" for doc in searched
    )
    return f"{NOT_IN_CORPUS_PREFIX} Searched: {searched_names}."


def build_response(
    sections: Sequence[Section],
    searched: Sequence[SearchedDocument],
    manifest_path: Path = config.MANIFEST_PATH,
) -> AskResponse:
    """Build the `AskResponse` envelope from verified `sections` and `searched` records.

    - If at least one section survives: `outcome="grounded_answer"`, ordered `sections`,
      `searched`, and `message=None`.
    - If no section survives: `outcome="not_in_corpus"`, `sections=[]`, `searched`,
      and a `message` stating that the guidance does not cover the question.
    """
    ordered = order_sections_by_manifest(sections, manifest_path=manifest_path)
    searched_list = list(searched)

    if not ordered:
        return AskResponse(
            outcome="not_in_corpus",
            searched=searched_list,
            sections=[],
            message=format_not_in_corpus_message(searched_list),
        )

    return AskResponse(
        outcome="grounded_answer",
        searched=searched_list,
        sections=ordered,
        message=None,
    )


def check_and_assemble(
    generations: Sequence[DocumentGeneration],
    retrieved_chunks: Sequence[Chunk],
    searched: Sequence[SearchedDocument],
    manifest_path: Path = config.MANIFEST_PATH,
) -> AskResponse:
    """Run the Phase 6 citation check and assemble the final `AskResponse` envelope."""
    surviving = filter_citations(generations, retrieved_chunks)
    return build_response(
        sections=surviving,
        searched=searched,
        manifest_path=manifest_path,
    )
