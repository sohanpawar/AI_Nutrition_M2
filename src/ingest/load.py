"""Corpus downloader and block loader (Phase 2).

Downloads `blocks.jsonl` from `config.corpus_url` to `corpus/blocks.jsonl`,
validates the manifest and block schema, verifies that the set of `document_id`
values in `blocks.jsonl` matches the seven manifest documents exactly, groups
blocks by `document_id` sorted by `ordinal`, and writes `corpus/parsed/<doc_id>.json`.
"""

from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

import requests
import yaml

from src import config
from src.schemas import Block, Document


class CorpusLoadError(RuntimeError):
    """Raised when downloading, validating, or grouping the corpus fails."""


def load_manifest(manifest_path: Path = config.MANIFEST_PATH) -> list[dict[str, Any]]:
    """Load and validate the seven document entries from `corpus/manifest.yaml`."""
    if not manifest_path.exists():
        raise CorpusLoadError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    docs = data.get("documents")
    if not isinstance(docs, list) or len(docs) != 7:
        raise CorpusLoadError(
            f"Manifest must list exactly 7 documents, found {0 if not isinstance(docs, list) else len(docs)}"
        )

    seen_ids: set[str] = set()
    for entry in docs:
        doc_id = entry.get("doc_id")
        if not doc_id or not isinstance(doc_id, str):
            raise CorpusLoadError("Manifest document is missing a valid 'doc_id'")
        if doc_id in seen_ids:
            raise CorpusLoadError(f"Duplicate doc_id in manifest: {doc_id}")
        seen_ids.add(doc_id)

        publisher = entry.get("publisher")
        if not publisher or not isinstance(publisher, str) or not publisher.strip():
            raise CorpusLoadError(f"Manifest document '{doc_id}' has a blank or missing 'publisher'")

        year = entry.get("year")
        if year is None or year == "" or isinstance(year, bool) or not isinstance(year, int) or year <= 0:
            raise CorpusLoadError(f"Manifest document '{doc_id}' has a blank or invalid 'year'")

        document_name = entry.get("document_name")
        if not document_name or not isinstance(document_name, str) or not document_name.strip():
            raise CorpusLoadError(f"Manifest document '{doc_id}' has a blank or missing 'document_name'")

    return docs


def download_blocks(
    url: str = config.corpus_url,
    dest_path: Path = config.BLOCKS_PATH,
    timeout: int = 30,
) -> Path:
    """Download `blocks.jsonl` atomically so a failed download never writes a partial file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".jsonl.tmp")
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content = response.text
        if not content.strip():
            raise CorpusLoadError(f"Downloaded corpus from {url} is empty")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(dest_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        if isinstance(exc, CorpusLoadError):
            raise
        raise CorpusLoadError(f"Failed to download corpus from {url}: {exc}") from exc

    return dest_path


def parse_blocks_file(blocks_path: Path = config.BLOCKS_PATH) -> list[Block]:
    """Parse each non-empty line of `blocks.jsonl` into a validated `Block`."""
    if not blocks_path.exists():
        raise CorpusLoadError(f"Blocks file does not exist: {blocks_path}")

    blocks: list[Block] = []
    required_keys = {"document_id", "ordinal", "type", "text", "heading_path"}

    with open(blocks_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusLoadError(f"Invalid JSON on line {line_no} of {blocks_path}: {exc}") from exc

            if not isinstance(payload, dict):
                raise CorpusLoadError(f"Line {line_no} of {blocks_path} is not a JSON object")

            missing = required_keys - set(payload.keys())
            if missing:
                raise CorpusLoadError(
                    f"Line {line_no} of {blocks_path} is missing required fields: {sorted(missing)}"
                )

            try:
                block = Block.model_validate(payload)
            except Exception as exc:
                raise CorpusLoadError(f"Invalid block schema on line {line_no} of {blocks_path}: {exc}") from exc

            blocks.append(block)

    if not blocks:
        raise CorpusLoadError(f"No blocks found in {blocks_path}")

    return blocks


def group_and_write_documents(
    blocks_path: Path = config.BLOCKS_PATH,
    manifest_path: Path = config.MANIFEST_PATH,
    parsed_dir: Path = config.PARSED_DIR,
    dataset_source_url: str = config.source_url,
    retrieval_date: str | None = None,
) -> dict[str, Document]:
    """Group blocks by `document_id` sorted by `ordinal` and write `corpus/parsed/<doc_id>.json`.

    Does not require network access if `blocks_path` already exists on disk.
    Fails without writing partial outputs if any manifest id is missing or an unknown id is present.
    """
    manifest_entries = load_manifest(manifest_path)
    manifest_by_id = {entry["doc_id"]: entry for entry in manifest_entries}
    expected_ids = set(manifest_by_id.keys())

    blocks = parse_blocks_file(blocks_path)
    found_ids = {b.document_id for b in blocks}

    missing_ids = expected_ids - found_ids
    unknown_ids = found_ids - expected_ids
    if missing_ids or unknown_ids:
        details: list[str] = []
        if missing_ids:
            details.append(f"missing manifest ids: {sorted(missing_ids)}")
        if unknown_ids:
            details.append(f"unknown document_ids: {sorted(unknown_ids)}")
        raise CorpusLoadError(
            f"Corpus document_id set does not match manifest ({'; '.join(details)})"
        )

    grouped_blocks: dict[str, list[Block]] = {doc_id: [] for doc_id in expected_ids}
    for block in blocks:
        grouped_blocks[block.document_id].append(block)

    ingest_day = retrieval_date or date.today().isoformat()
    documents: dict[str, Document] = {}

    for entry in manifest_entries:
        doc_id = entry["doc_id"]
        ordered_blocks = sorted(grouped_blocks[doc_id], key=lambda b: b.ordinal)
        doc = Document(
            doc_id=doc_id,
            document_name=entry["document_name"],
            publisher=entry["publisher"],
            year=entry["year"],
            source_url=dataset_source_url,
            retrieval_date=ingest_day,
            aliases=list(entry.get("aliases") or []),
            blocks=ordered_blocks,
        )
        documents[doc_id] = doc

    parsed_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, doc in documents.items():
        out_path = parsed_dir / f"{doc_id}.json"
        tmp_path = parsed_dir / f"{doc_id}.json.tmp"
        tmp_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

    return documents


def load_corpus(
    download: bool = True,
    url: str = config.corpus_url,
    blocks_path: Path = config.BLOCKS_PATH,
    manifest_path: Path = config.MANIFEST_PATH,
    parsed_dir: Path = config.PARSED_DIR,
    dataset_source_url: str = config.source_url,
    retrieval_date: str | None = None,
) -> dict[str, Document]:
    """Download `blocks.jsonl` (when `download=True`) and write `corpus/parsed/<doc_id>.json`."""
    if download:
        download_blocks(url=url, dest_path=blocks_path)
    return group_and_write_documents(
        blocks_path=blocks_path,
        manifest_path=manifest_path,
        parsed_dir=parsed_dir,
        dataset_source_url=dataset_source_url,
        retrieval_date=retrieval_date,
    )


def main() -> int:
    """CLI entry point for Phase 2 corpus download and grouping."""
    try:
        docs = load_corpus(download=True)
    except CorpusLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total_blocks = sum(len(doc.blocks) for doc in docs.values())
    print(f"Loaded {len(docs)} documents ({total_blocks} total blocks):")
    for doc_id, doc in docs.items():
        print(f"  - {doc_id}: {len(doc.blocks)} blocks ({doc.publisher}, {doc.year})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
