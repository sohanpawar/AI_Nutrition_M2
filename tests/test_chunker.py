"""Tests for Phase 3 structure-aware chunking."""

import inspect
import unittest

from src import config
from src.ingest import chunk as chunk_module
from src.ingest.chunk import chunk_document, estimate_tokens, render_markdown_table
from src.ingest.load import load_manifest
from src.schemas import Block, Chunk, Document


def _make_doc(
    doc_id: str = "test-doc",
    blocks: list[Block] | None = None,
) -> Document:
    return Document(
        doc_id=doc_id,
        document_name="Test Guidance Document",
        publisher="Test Health Authority",
        year=2024,
        source_url=config.source_url,
        retrieval_date="2026-10-27",
        aliases=["test doc"],
        blocks=blocks or [],
    )


class TestStructureAwareChunker(unittest.TestCase):
    """Verify all Phase 3 exit checks."""

    def test_section_of_three_list_items_never_cuts_mid_item(self) -> None:
        item_1 = "Keep raw meat, poultry, and seafood separate from ready-to-eat foods in your shopping cart and refrigerator."
        item_2 = "Use separate cutting boards and plates for raw meat, poultry, and seafood versus produce and bread."
        item_3 = "Wash hands, cutting boards, dishes, and utensils with hot, soapy water after they touch raw meat."

        blocks = [
            Block(
                document_id="who-five-keys",
                ordinal=1,
                type="heading",
                text="Separate raw and cooked",
                heading_path=["Five Keys", "Separate raw and cooked"],
            ),
            Block(
                document_id="who-five-keys",
                ordinal=2,
                type="list_item",
                text=item_1,
                heading_path=["Five Keys", "Separate raw and cooked"],
            ),
            Block(
                document_id="who-five-keys",
                ordinal=3,
                type="list_item",
                text=item_2,
                heading_path=["Five Keys", "Separate raw and cooked"],
            ),
            Block(
                document_id="who-five-keys",
                ordinal=4,
                type="list_item",
                text=item_3,
                heading_path=["Five Keys", "Separate raw and cooked"],
            ),
        ]
        doc = _make_doc(doc_id="who-five-keys", blocks=blocks)

        # Force splitting across list-item boundaries with a budget that fits ~1-2 items
        chunks = chunk_document(doc, chunk_token_budget=25, table_token_hard_cap=800)
        self.assertGreaterEqual(len(chunks), 2)

        for c in chunks:
            self.assertEqual(c.section_heading, "Separate raw and cooked")

        # Each list item must appear completely intact in exactly one chunk, never cut mid-item
        for item in (item_1, item_2, item_3):
            containing = [c for c in chunks if item in c.text]
            self.assertEqual(
                len(containing),
                1,
                f"Expected list item {item!r} to be intact in exactly 1 chunk, got {len(containing)}",
            )

    def test_table_smaller_than_hard_cap_stays_one_chunk_with_header(self) -> None:
        header = ["Age", "Recommended maximum free sugars intake", "Sugar cubes"]
        rows = [
            ["4 to 6 years", "No more than 19g a day", "5 cubes"],
            ["7 to 10 years", "No more than 24g a day", "6 cubes"],
            ["From 11 years, including adults", "No more than 30g a day", "7 cubes"],
        ]
        table_md = render_markdown_table(header, rows)
        blocks = [
            Block(
                document_id="eatwell-guide",
                ordinal=70,
                type="heading",
                text="Cutting down on sugar",
                heading_path=["Cutting down on sugar"],
            ),
            Block(
                document_id="eatwell-guide",
                ordinal=71,
                type="table",
                text=table_md,
                heading_path=["Cutting down on sugar"],
                header=header,
                rows=rows,
            ),
        ]
        doc = _make_doc(doc_id="eatwell-guide", blocks=blocks)

        chunks = chunk_document(doc, chunk_token_budget=50, table_token_hard_cap=500)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertIsNone(chunk.table_part)
        self.assertEqual(chunk.section_heading, "Cutting down on sugar")
        for col in header:
            self.assertIn(col, chunk.text)
        for row in rows:
            for cell in row:
                self.assertIn(cell, chunk.text)

    def test_table_over_hard_cap_splits_by_rows_with_header_and_table_part(self) -> None:
        header = ["Food", "Type", "Refrigerator [40°F (4°C) or below]", "Freezer [0°F (-18°C) or below]"]
        rows = [
            ["Fresh Beef, Veal, Lamb, Pork", "Steaks", "3 to 5 days", "6 to 12 months"],
            ["Fresh Beef, Veal, Lamb, Pork", "Chops", "3 to 5 days", "4 to 6 months"],
            ["Fresh Beef, Veal, Lamb, Pork", "Roasts", "3 to 5 days", "4 to 12 months"],
            ["Cooked Poultry", "Chicken or turkey, whole", "3 to 4 days", "4 months"],
            ["Leftovers", "Cooked meat or poultry", "3 to 4 days", "2 to 6 months"],
        ]
        table_md = render_markdown_table(header, rows)
        blocks = [
            Block(
                document_id="cold-food-storage",
                ordinal=2,
                type="table",
                text=table_md,
                heading_path=["Cold Food Storage Chart"],
                header=header,
                rows=rows,
            )
        ]
        doc = _make_doc(doc_id="cold-food-storage", blocks=blocks)

        # Hard cap forces multiple row-group pieces
        chunks = chunk_document(doc, chunk_token_budget=500, table_token_hard_cap=65)
        self.assertGreater(len(chunks), 1)

        for idx, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk.table_part, idx)
            self.assertEqual(chunk.section_heading, "Cold Food Storage Chart")
            for col in header:
                self.assertIn(col, chunk.text)

        # Every row must stay intact on a single piece, never split across pieces
        for row in rows:
            formatted_row = "| " + " | ".join(row) + " |"
            matching_pieces = [c for c in chunks if formatted_row in c.text]
            self.assertEqual(
                len(matching_pieces),
                1,
                f"Row {formatted_row!r} must appear intact in exactly one piece",
            )

    def test_two_row_cold_storage_fixture_keeps_food_with_both_time_cells(self) -> None:
        header = ["Food", "Type", "Refrigerator [40°F (4°C) or below]", "Freezer [0°F (-18°C) or below]"]
        row_1 = ["Leftovers", "Cooked meat or poultry", "3 to 4 days", "2 to 6 months"]
        row_2 = ["Salads", "Egg, chicken, ham, tuna & macaroni salads", "3 to 4 days", "Does not freeze well"]
        table_md = render_markdown_table(header, [row_1, row_2])

        blocks = [
            Block(
                document_id="cold-food-storage",
                ordinal=2,
                type="table",
                text=table_md,
                heading_path=["Cold Food Storage Chart"],
                header=header,
                rows=[row_1, row_2],
            )
        ]
        doc = _make_doc(doc_id="cold-food-storage", blocks=blocks)

        # Even if hard cap splits the 2 rows into 2 pieces, each food stays with both of its time cells
        for cap in (500, 45):
            chunks = chunk_document(doc, chunk_token_budget=500, table_token_hard_cap=cap)
            for row in (row_1, row_2):
                food, food_type, fridge_time, freezer_time = row
                pieces_with_food = [c for c in chunks if food_type in c.text]
                self.assertEqual(len(pieces_with_food), 1)
                piece_text = pieces_with_food[0].text
                self.assertIn(food, piece_text)
                self.assertIn(fridge_time, piece_text)
                self.assertIn(freezer_time, piece_text)

    def test_real_corpus_chunks_have_required_metadata_and_dataset_url(self) -> None:
        manifest_entries = load_manifest(config.MANIFEST_PATH)
        self.assertEqual(len(manifest_entries), 7)

        for entry in manifest_entries:
            doc_id = entry["doc_id"]
            chunk_file = config.PARSED_DIR / f"{doc_id}.chunks.jsonl"
            self.assertTrue(chunk_file.exists(), f"Missing chunk file: {chunk_file}")

            lines = [line for line in chunk_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreater(len(lines), 0, f"No chunks in {chunk_file}")

            seen_chunk_ids: set[str] = set()
            for line in lines:
                chunk = Chunk.model_validate_json(line)
                self.assertEqual(chunk.doc_id, doc_id)
                self.assertEqual(chunk.document_name, entry["document_name"])
                self.assertEqual(chunk.publisher, entry["publisher"])
                self.assertEqual(chunk.year, entry["year"])
                self.assertEqual(chunk.source_url, config.source_url)
                self.assertTrue(chunk.retrieval_date.strip())
                self.assertTrue(chunk.section_heading.strip())
                self.assertTrue(chunk.text.strip())
                self.assertLessEqual(chunk.ordinal_start, chunk.ordinal_end)
                self.assertTrue(chunk.chunk_id.startswith(f"{doc_id}-"))
                self.assertNotIn(chunk.chunk_id, seen_chunk_ids)
                seen_chunk_ids.add(chunk.chunk_id)

    def test_no_fixed_character_or_token_window_with_overlap(self) -> None:
        source = inspect.getsource(chunk_module)
        self.assertNotIn("chunk_overlap", source)
        self.assertNotIn("RecursiveCharacterTextSplitter", source)
        self.assertNotIn("CharacterTextSplitter", source)


if __name__ == "__main__":
    unittest.main()
