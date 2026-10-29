"""Structure-aware block chunker (Phase 3).

Walks one document's blocks in `ordinal` order and builds structure-preserving
chunks without using fixed-size character or token windows with overlap:
- Heading blocks open a section; following `paragraph` and `list_item` blocks
  sharing that `heading_path` join it until a heading changes the path or a
  table block is encountered.
- Sections exceeding `chunk_token_budget` split on `list_item` / `paragraph`
  block boundaries while repeating `section_heading` on every child. A
  `list_item` is never cut in half.
- Each `table` block stays as a single chunk with its header attached unless it
  exceeds `table_token_hard_cap`, in which case it splits by row groups,
  repeats `header` on every piece, and sets `table_part`. No row is ever split
  across pieces.
"""

from pathlib import Path
import re
import sys

from src import config
from src.ingest.load import load_manifest
from src.schemas import Block, Chunk, Document

TOKEN_PATTERN = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Estimate token count deterministically without fixed-window slicing."""
    return len(TOKEN_PATTERN.findall(text))


def render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table with `header` and `rows` kept intact."""
    header_cells = [str(cell).strip() for cell in header]
    header_line = "| " + " | ".join(header_cells) + " |"
    separator_line = "| " + " | ".join("---" for _ in header_cells) + " |"
    row_lines = [
        "| " + " | ".join(str(cell).strip() for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def _extract_table_header_and_rows(block: Block) -> tuple[list[str], list[list[str]]]:
    """Return `(header, rows)` from `block`, falling back to markdown table parsing if needed."""
    if block.header is not None and block.rows is not None:
        return list(block.header), [list(r) for r in block.rows]

    lines = [line.strip() for line in block.text.strip().splitlines() if line.strip()]
    parsed_rows: list[list[str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip("|").strip()
        cells = [c.strip() for c in stripped.split("|")]
        if idx == 1 and all(set(c) <= {"-", ":"} and "-" in c for c in cells if c):
            continue
        parsed_rows.append(cells)

    if not parsed_rows:
        return [], []
    header = block.header if block.header is not None else parsed_rows[0]
    rows = block.rows if block.rows is not None else parsed_rows[1:]
    return list(header), [list(r) for r in rows]


def _format_prose_block(block: Block) -> str:
    """Format a prose block (`paragraph` or `list_item`) for inclusion in a section chunk."""
    text = block.text.strip()
    if block.type == "list_item" and not text.startswith(("-", "*", "•")):
        return f"- {text}"
    return text


def _chunk_table_block(
    doc: Document,
    block: Block,
    table_token_hard_cap: int,
) -> list[Chunk]:
    """Chunk a single `table` block, splitting by row groups if over `table_token_hard_cap`."""
    section_heading = block.heading_path[-1] if block.heading_path else doc.document_name
    header, rows = _extract_table_header_and_rows(block)

    full_text = block.text.strip()
    if header and rows and not full_text:
        full_text = render_markdown_table(header, rows)
    elif header and all(h not in full_text for h in header if h):
        full_text = render_markdown_table(header, rows)

    if estimate_tokens(full_text) <= table_token_hard_cap or len(rows) <= 1:
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}-{block.ordinal}-{block.ordinal}",
                doc_id=doc.doc_id,
                document_name=doc.document_name,
                publisher=doc.publisher,
                year=doc.year,
                source_url=doc.source_url,
                retrieval_date=doc.retrieval_date,
                section_heading=section_heading,
                text=full_text,
                ordinal_start=block.ordinal,
                ordinal_end=block.ordinal,
                table_part=None,
            )
        ]

    # Split by row groups while repeating the header on every piece and keeping each row intact
    row_groups: list[list[list[str]]] = []
    current_rows: list[list[str]] = []

    for row in rows:
        candidate_rows = [*current_rows, row]
        candidate_text = render_markdown_table(header, candidate_rows)
        if current_rows and estimate_tokens(candidate_text) > table_token_hard_cap:
            row_groups.append(current_rows)
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        row_groups.append(current_rows)

    chunks: list[Chunk] = []
    for part_idx, group in enumerate(row_groups, start=1):
        piece_text = render_markdown_table(header, group)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{block.ordinal}-{block.ordinal}-p{part_idx}",
                doc_id=doc.doc_id,
                document_name=doc.document_name,
                publisher=doc.publisher,
                year=doc.year,
                source_url=doc.source_url,
                retrieval_date=doc.retrieval_date,
                section_heading=section_heading,
                text=piece_text,
                ordinal_start=block.ordinal,
                ordinal_end=block.ordinal,
                table_part=part_idx,
            )
        )
    return chunks


def _split_section_units(
    units: list[list[Block]],
    budget: int,
) -> list[list[Block]]:
    """Pack indivisible units into groups that fit within `budget` tokens."""
    packed: list[list[Block]] = []
    current_blocks: list[Block] = []

    for unit in units:
        candidate_blocks = [*current_blocks, *unit]
        candidate_text = "\n\n".join(_format_prose_block(b) for b in candidate_blocks)
        if current_blocks and estimate_tokens(candidate_text) > budget:
            packed.append(current_blocks)
            current_blocks = list(unit)
        else:
            current_blocks = candidate_blocks

    if current_blocks:
        packed.append(current_blocks)

    return packed


def _chunk_prose_section(
    doc: Document,
    heading_block: Block | None,
    content_blocks: list[Block],
    chunk_token_budget: int,
) -> list[Chunk]:
    """Chunk one prose section (`paragraph` and `list_item` blocks sharing `heading_path`).

    If the section exceeds `chunk_token_budget`, splits first on `list_item` boundaries,
    then on `paragraph` boundaries. Never splits inside a `list_item`.
    """
    if not content_blocks:
        return []

    ref_block = content_blocks[0]
    section_heading = (
        ref_block.heading_path[-1]
        if ref_block.heading_path
        else (heading_block.text.strip() if heading_block else doc.document_name)
    )

    full_text = "\n\n".join(_format_prose_block(b) for b in content_blocks)
    if estimate_tokens(full_text) <= chunk_token_budget or len(content_blocks) == 1:
        ordinal_start = (
            heading_block.ordinal
            if heading_block is not None and heading_block.heading_path == ref_block.heading_path
            else content_blocks[0].ordinal
        )
        ordinal_end = content_blocks[-1].ordinal
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}-{ordinal_start}-{ordinal_end}",
                doc_id=doc.doc_id,
                document_name=doc.document_name,
                publisher=doc.publisher,
                year=doc.year,
                source_url=doc.source_url,
                retrieval_date=doc.retrieval_date,
                section_heading=section_heading,
                text=full_text,
                ordinal_start=ordinal_start,
                ordinal_end=ordinal_end,
                table_part=None,
            )
        ]

    # Stage 1: split on list_item boundaries (contiguous paragraphs stay grouped unless still over budget)
    stage1_units: list[list[Block]] = []
    para_run: list[Block] = []
    for b in content_blocks:
        if b.type == "list_item":
            if para_run:
                stage1_units.append(para_run)
                para_run = []
            stage1_units.append([b])
        else:
            para_run.append(b)
    if para_run:
        stage1_units.append(para_run)

    # Stage 2: for any paragraph run that alone exceeds budget, split on individual paragraph boundaries
    refined_units: list[list[Block]] = []
    for unit in stage1_units:
        unit_text = "\n\n".join(_format_prose_block(b) for b in unit)
        if len(unit) > 1 and estimate_tokens(unit_text) > chunk_token_budget:
            for b in unit:
                refined_units.append([b])
        else:
            refined_units.append(unit)

    groups = _split_section_units(refined_units, chunk_token_budget)

    chunks: list[Chunk] = []
    for idx, group in enumerate(groups):
        if (
            idx == 0
            and heading_block is not None
            and heading_block.heading_path == ref_block.heading_path
        ):
            ordinal_start = heading_block.ordinal
        else:
            ordinal_start = group[0].ordinal
        ordinal_end = group[-1].ordinal
        group_text = "\n\n".join(_format_prose_block(b) for b in group)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{ordinal_start}-{ordinal_end}",
                doc_id=doc.doc_id,
                document_name=doc.document_name,
                publisher=doc.publisher,
                year=doc.year,
                source_url=doc.source_url,
                retrieval_date=doc.retrieval_date,
                section_heading=section_heading,
                text=group_text,
                ordinal_start=ordinal_start,
                ordinal_end=ordinal_end,
                table_part=None,
            )
        )
    return chunks


def chunk_document(
    doc: Document,
    chunk_token_budget: int = config.chunk_token_budget,
    table_token_hard_cap: int = config.table_token_hard_cap,
) -> list[Chunk]:
    """Walk one `Document`'s blocks in `ordinal` order and return structure-aware `Chunk`s."""
    ordered_blocks = sorted(doc.blocks, key=lambda b: b.ordinal)
    chunks: list[Chunk] = []

    current_heading_block: Block | None = None
    current_heading_path: list[str] | None = None
    current_content: list[Block] = []

    def flush_section() -> None:
        nonlocal current_heading_block, current_content
        if current_content:
            chunks.extend(
                _chunk_prose_section(
                    doc=doc,
                    heading_block=current_heading_block,
                    content_blocks=current_content,
                    chunk_token_budget=chunk_token_budget,
                )
            )
            current_heading_block = None
            current_content = []

    for block in ordered_blocks:
        if block.type == "heading":
            if current_content and block.heading_path != current_heading_path:
                flush_section()
            current_heading_block = block
            current_heading_path = list(block.heading_path)
        elif block.type == "table":
            flush_section()
            chunks.extend(
                _chunk_table_block(
                    doc=doc,
                    block=block,
                    table_token_hard_cap=table_token_hard_cap,
                )
            )
            current_heading_path = list(block.heading_path)
        else:
            # paragraph or list_item
            if current_content and block.heading_path != current_heading_path:
                flush_section()
            if current_heading_path != block.heading_path:
                current_heading_path = list(block.heading_path)
            current_content.append(block)

    flush_section()
    return chunks


def write_document_chunks(
    doc: Document,
    parsed_dir: Path = config.PARSED_DIR,
    chunk_token_budget: int = config.chunk_token_budget,
    table_token_hard_cap: int = config.table_token_hard_cap,
) -> list[Chunk]:
    """Chunk `doc` and write `corpus/parsed/<doc_id>.chunks.jsonl` atomically."""
    chunks = chunk_document(
        doc=doc,
        chunk_token_budget=chunk_token_budget,
        table_token_hard_cap=table_token_hard_cap,
    )
    parsed_dir.mkdir(parents=True, exist_ok=True)
    out_path = parsed_dir / f"{doc.doc_id}.chunks.jsonl"
    tmp_path = parsed_dir / f"{doc.doc_id}.chunks.jsonl.tmp"

    lines = [chunk.model_dump_json() for chunk in chunks]
    tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp_path.replace(out_path)
    return chunks


def chunk_all_documents(
    parsed_dir: Path = config.PARSED_DIR,
    manifest_path: Path = config.MANIFEST_PATH,
    chunk_token_budget: int = config.chunk_token_budget,
    table_token_hard_cap: int = config.table_token_hard_cap,
) -> dict[str, list[Chunk]]:
    """Chunk all seven parsed documents in manifest order and write `.chunks.jsonl` files."""
    manifest_entries = load_manifest(manifest_path)
    all_chunks: dict[str, list[Chunk]] = {}

    for entry in manifest_entries:
        doc_id = entry["doc_id"]
        doc_path = parsed_dir / f"{doc_id}.json"
        if not doc_path.exists():
            raise FileNotFoundError(f"Parsed document file not found: {doc_path}")
        doc = Document.model_validate_json(doc_path.read_text(encoding="utf-8"))
        all_chunks[doc_id] = write_document_chunks(
            doc=doc,
            parsed_dir=parsed_dir,
            chunk_token_budget=chunk_token_budget,
            table_token_hard_cap=table_token_hard_cap,
        )

    return all_chunks


def main() -> int:
    """CLI entry point for Phase 3 structure-aware chunking."""
    try:
        chunks_by_doc = chunk_all_documents()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total = sum(len(c) for c in chunks_by_doc.values())
    print(
        f"Chunked {len(chunks_by_doc)} documents into {total} total chunks "
        f"(chunk_token_budget={config.chunk_token_budget}, table_token_hard_cap={config.table_token_hard_cap}):"
    )
    for doc_id, chunks in chunks_by_doc.items():
        split_tables = [c for c in chunks if c.table_part is not None]
        split_note = f", {len(split_tables)} split-table pieces" if split_tables else ""
        print(f"  - {doc_id}: {len(chunks)} chunks{split_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
