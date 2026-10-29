"""Tests for Phase 6 deterministic section assembly and response envelope building."""

import unittest

from src import config
from src.answer.assemble import (
    build_response,
    check_and_assemble,
    order_sections_by_manifest,
    render_sections,
)
from src.schemas import Chunk, Claim, DocumentGeneration, SearchedDocument


class TestAssembler(unittest.TestCase):
    """Verify all Phase 6 assembler exit checks."""

    def setUp(self) -> None:
        self.who_chunk = Chunk(
            chunk_id="who-healthy-diet-34-43",
            doc_id="who-healthy-diet",
            document_name="WHO Healthy diet, Fact sheet N°394",
            publisher="World Health Organization",
            year=2020,
            source_url=config.source_url,
            retrieval_date="2026-10-27",
            section_heading="Fats",
            text="Unsaturated fats are preferable to saturated fats.",
            ordinal_start=34,
            ordinal_end=43,
        )
        self.cold_chunk = Chunk(
            chunk_id="cold-food-storage-2-2-p2",
            doc_id="cold-food-storage",
            document_name="Cold Food Storage Chart",
            publisher="USDA Food Safety and Inspection Service",
            year=2023,
            source_url=config.source_url,
            retrieval_date="2026-10-27",
            section_heading="Cold Food Storage Chart",
            text="| Leftovers | Cooked meat or poultry | 3 to 4 days | 2 to 6 months |",
            ordinal_start=2,
            ordinal_end=2,
            table_part=2,
        )
        self.searched = [
            SearchedDocument(
                doc_id="who-healthy-diet",
                document_name="WHO Healthy diet, Fact sheet N°394",
                publisher="World Health Organization",
            ),
            SearchedDocument(
                doc_id="cold-food-storage",
                document_name="Cold Food Storage Chart",
                publisher="USDA Food Safety and Inspection Service",
            ),
        ]

    def test_two_fixture_sections_render_as_two_blocks_in_manifest_order_without_synthesis(self) -> None:
        # Provide cold-food-storage before who-healthy-diet to verify manifest ordering
        generations = [
            DocumentGeneration(
                doc_id="cold-food-storage",
                abstain=False,
                claims=[
                    Claim(
                        text="Leftover cooked meat or poultry keeps 3 to 4 days in the refrigerator.",
                        chunk_ids=["cold-food-storage-2-2-p2"],
                    )
                ],
            ),
            DocumentGeneration(
                doc_id="who-healthy-diet",
                abstain=False,
                claims=[
                    Claim(
                        text="Unsaturated fats are preferable to saturated fats.",
                        chunk_ids=["who-healthy-diet-34-43"],
                    )
                ],
            ),
        ]

        response = check_and_assemble(
            generations=generations,
            retrieved_chunks=[self.cold_chunk, self.who_chunk],
            searched=self.searched,
        )

        self.assertEqual(response.outcome, "grounded_answer")
        self.assertIsNone(response.message)
        self.assertEqual(len(response.sections), 2)
        # Manifest order puts who-healthy-diet before cold-food-storage
        self.assertEqual(
            [sec.doc_id for sec in response.sections],
            ["who-healthy-diet", "cold-food-storage"],
        )
        for sec in response.sections:
            self.assertEqual(sec.source_url, config.source_url)

        rendered = render_sections(response.sections)
        blocks = rendered.split("\n\n")
        self.assertEqual(len(blocks), 2)
        self.assertIn("WHO Healthy diet, Fact sheet N°394 — World Health Organization, 2020", blocks[0])
        self.assertIn(config.source_url, blocks[0])
        self.assertIn("(Fats)", blocks[0])

        self.assertIn(
            "Cold Food Storage Chart — USDA Food Safety and Inspection Service, 2023",
            blocks[1],
        )
        self.assertIn(config.source_url, blocks[1])
        self.assertIn("(Cold Food Storage Chart)", blocks[1])

        lowered = rendered.lower()
        self.assertNotIn("the guidelines say", lowered)
        self.assertNotIn("both authorities agree", lowered)

    def test_empty_surviving_sections_builds_not_in_corpus_envelope(self) -> None:
        response = build_response(sections=[], searched=self.searched)
        self.assertEqual(response.outcome, "not_in_corpus")
        self.assertEqual(response.sections, [])
        self.assertEqual(response.searched, self.searched)
        self.assertIsNotNone(response.message)
        assert response.message is not None
        self.assertIn("WHO Healthy diet, Fact sheet N°394", response.message)
        self.assertIn("Cold Food Storage Chart", response.message)


if __name__ == "__main__":
    unittest.main()
