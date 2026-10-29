"""Document alias and filter resolver (Phase 5).

Resolves explicit `document_filter` arguments and in-question document aliases
against `corpus/manifest.yaml`:
- Explicit `document_filter` wins over alias detection inside the question.
- An explicit `document_filter` that matches neither a `doc_id` nor a manifest
  alias raises `InvalidDocumentFilterError` (which the API turns into HTTP 400).
- Topic words such as "fridge" do not resolve to a document.
"""

from pathlib import Path
import re

from src import config
from src.ingest.load import load_manifest


class InvalidDocumentFilterError(ValueError):
    """Raised when an explicit `document_filter` matches no manifest `doc_id` or alias."""


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation except alphanumeric spaces, and collapse whitespace."""
    cleaned = re.sub(r"[^a-z0-9°]+", " ", text.lower())
    return " ".join(cleaned.split())


def build_alias_lookup(
    manifest_path: Path = config.MANIFEST_PATH,
) -> tuple[set[str], dict[str, str]]:
    """Return `(valid_doc_ids, normalized_alias_to_doc_id)` from `corpus/manifest.yaml`."""
    entries = load_manifest(manifest_path)
    valid_doc_ids: set[str] = set()
    alias_map: dict[str, str] = {}

    for entry in entries:
        doc_id: str = entry["doc_id"]
        valid_doc_ids.add(doc_id)
        alias_map[_normalize(doc_id)] = doc_id
        alias_map[doc_id.lower()] = doc_id

        doc_name = entry.get("document_name", "")
        if doc_name:
            alias_map[_normalize(doc_name)] = doc_id

        for alias in entry.get("aliases") or []:
            norm_alias = _normalize(alias)
            if norm_alias:
                alias_map[norm_alias] = doc_id

    return valid_doc_ids, alias_map


def resolve_explicit_filter(
    document_filter: str,
    manifest_path: Path = config.MANIFEST_PATH,
) -> str:
    """Resolve an explicit `document_filter` string to a `doc_id` or raise `InvalidDocumentFilterError`."""
    raw = document_filter.strip()
    if not raw:
        raise InvalidDocumentFilterError("Empty document_filter")

    valid_doc_ids, alias_map = build_alias_lookup(manifest_path)
    if raw in valid_doc_ids:
        return raw
    if raw.lower() in valid_doc_ids:
        return raw.lower()

    norm = _normalize(raw)
    if norm in alias_map:
        return alias_map[norm]

    raise InvalidDocumentFilterError(
        f"Unknown document_filter: {document_filter!r} matches no manifest doc_id or alias"
    )


def detect_alias_in_question(
    question: str,
    manifest_path: Path = config.MANIFEST_PATH,
) -> str | None:
    """Detect whether `question` names a single manifest document via its aliases.

    If zero or more than one distinct document is named in the question, returns `None`
    so retrieval searches the full corpus.
    """
    norm_question = f" {_normalize(question)} "
    if not norm_question.strip():
        return None

    _valid_doc_ids, alias_map = build_alias_lookup(manifest_path)
    matched_doc_ids: set[str] = set()

    # Sort aliases longest-first so more specific phrases match cleanly
    for norm_alias, doc_id in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
        if not norm_alias:
            continue
        if f" {norm_alias} " in norm_question:
            matched_doc_ids.add(doc_id)

    if len(matched_doc_ids) == 1:
        return next(iter(matched_doc_ids))
    return None


def resolve_document(
    question: str,
    document_filter: str | None = None,
    manifest_path: Path = config.MANIFEST_PATH,
) -> str | None:
    """Resolve the target `doc_id` (or `None` for global search).

    - If `document_filter` is provided, it wins over any alias in `question`, and raises
      `InvalidDocumentFilterError` if it matches no `doc_id` or alias.
    - Otherwise, checks `question` for a manifest alias.
    """
    if document_filter is not None:
        return resolve_explicit_filter(document_filter, manifest_path=manifest_path)
    return detect_alias_in_question(question, manifest_path=manifest_path)
