"""Shared Pydantic schemas for blocks, documents, chunks, claims, and the /ask API."""

from typing import Literal
from pydantic import BaseModel, Field

BlockType = Literal["heading", "paragraph", "list_item", "table"]
Outcome = Literal["grounded_answer", "not_in_corpus", "out_of_scope"]


class Block(BaseModel):
    """A single block from blocks.jsonl."""

    document_id: str
    ordinal: int
    type: BlockType
    text: str
    heading_path: list[str]
    page: int | None = None
    level: int | None = None
    header: list[str] | None = None
    rows: list[list[str]] | None = None


class Document(BaseModel):
    """Metadata and blocks for one manifest guidance document."""

    doc_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    retrieval_date: str = ""
    aliases: list[str] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)


class Chunk(BaseModel):
    """A structure-aware citation unit indexed for retrieval."""

    chunk_id: str
    doc_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    retrieval_date: str
    section_heading: str
    text: str
    ordinal_start: int
    ordinal_end: int
    table_part: int | str | None = None


class Claim(BaseModel):
    """A single claim grounded in retrieved chunks of one document."""

    text: str
    chunk_ids: list[str]
    section_heading: str | None = None


class DocumentGeneration(BaseModel):
    """Structured output returned by the per-document generator."""

    doc_id: str
    abstain: bool
    claims: list[Claim] = Field(default_factory=list)


class Section(BaseModel):
    """A single-document section in a grounded answer."""

    doc_id: str | None = None
    document_name: str
    publisher: str
    year: int
    source_url: str
    claims: list[Claim] = Field(default_factory=list)


class SearchedDocument(BaseModel):
    """Metadata naming a document consulted during retrieval."""

    doc_id: str
    document_name: str
    publisher: str


class AskRequest(BaseModel):
    """Request payload for POST /ask."""

    question: str
    document_filter: str | None = None


class AskResponse(BaseModel):
    """Unified response envelope for grounded_answer, not_in_corpus, and out_of_scope."""

    outcome: Outcome
    searched: list[SearchedDocument] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    message: str | None = None
