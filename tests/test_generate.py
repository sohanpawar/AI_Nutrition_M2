"""Tests for Phase 7 per-document answer generator."""

from collections.abc import Sequence
import json
import unittest

from src import config
from src.answer.assemble import check_and_assemble
from src.answer.citations import filter_citations
from src.answer.generate import (
    build_document_prompt,
    generate_per_document,
)
from src.schemas import Chunk, DocumentGeneration, SearchedDocument


def _make_chunk(
    chunk_id: str,
    doc_id: str,
    document_name: str,
    publisher: str,
    year: int,
    section_heading: str,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        document_name=document_name,
        publisher=publisher,
        year=year,
        source_url=config.source_url,
        retrieval_date="2026-10-27",
        section_heading=section_heading,
        text=text,
        ordinal_start=1,
        ordinal_end=2,
    )


class TestPerDocumentGenerator(unittest.TestCase):
    """Verify all Phase 7 exit criteria."""

    def setUp(self) -> None:
        self.cold_chunk = _make_chunk(
            chunk_id="cold-food-storage-2-2-p2",
            doc_id="cold-food-storage",
            document_name="Cold Food Storage Chart",
            publisher="USDA Food Safety and Inspection Service",
            year=2023,
            section_heading="Cold Food Storage Chart",
            text="| Leftovers | Cooked meat or poultry | 3 to 4 days | 2 to 6 months |",
        )
        self.fsa_chunk = _make_chunk(
            chunk_id="fsa-chill-6-12",
            doc_id="fsa-chill",
            document_name="How to chill, freeze and defrost food safely",
            publisher="UK Food Standards Agency",
            year=2024,
            section_heading="1. Chilling food",
            text="Your fridge should be set between 0 and 5°C.",
        )
        self.who_chunk = _make_chunk(
            chunk_id="who-healthy-diet-34-43",
            doc_id="who-healthy-diet",
            document_name="WHO Healthy diet, Fact sheet N°394",
            publisher="World Health Organization",
            year=2020,
            section_heading="Fats",
            text="Unsaturated fats are preferable to saturated fats.",
        )
        self.retrieved = [self.cold_chunk, self.fsa_chunk, self.who_chunk]
        self.searched = [
            SearchedDocument(
                doc_id="cold-food-storage",
                document_name="Cold Food Storage Chart",
                publisher="USDA Food Safety and Inspection Service",
            ),
            SearchedDocument(
                doc_id="fsa-chill",
                document_name="How to chill, freeze and defrost food safely",
                publisher="UK Food Standards Agency",
            ),
            SearchedDocument(
                doc_id="who-healthy-diet",
                document_name="WHO Healthy diet, Fact sheet N°394",
                publisher="World Health Organization",
            ),
        ]

    def test_invoked_once_per_doc_id_and_prompt_is_isolated(self) -> None:
        calls: list[tuple[str, str, list[str]]] = []

        def fake_model(
            doc_id: str,
            prompt: str,
            question: str,
            chunks: Sequence[Chunk],
        ) -> str:
            calls.append((doc_id, prompt, [c.chunk_id for c in chunks]))
            return json.dumps(
                {
                    "doc_id": doc_id,
                    "abstain": False,
                    "claims": [
                        {
                            "text": f"Supported fact from {doc_id}.",
                            "chunk_ids": [chunks[0].chunk_id],
                        }
                    ],
                }
            )

        generations = generate_per_document(
            question="What temperature should my fridge be, and how long can leftovers stay there?",
            retrieved_chunks=self.retrieved,
            chat_model=fake_model,
        )

        # Invoked once per surviving doc_id (3 docs in self.retrieved, skipping the other 4 manifest docs)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(generations), 3)
        self.assertEqual(
            [doc_id for doc_id, _prompt, _cids in calls],
            ["cold-food-storage", "fsa-chill", "who-healthy-diet"],
        )

        # Assert the prompt for cold-food-storage does not contain an fsa-chill or who-healthy-diet chunk id or text
        cold_prompt = dict((doc_id, prompt) for doc_id, prompt, _cids in calls)["cold-food-storage"]
        self.assertIn("cold-food-storage-2-2-p2", cold_prompt)
        self.assertIn("Cold Food Storage Chart", cold_prompt)
        self.assertNotIn("fsa-chill", cold_prompt)
        self.assertNotIn("fsa-chill-6-12", cold_prompt)
        self.assertNotIn("who-healthy-diet", cold_prompt)
        self.assertNotIn("who-healthy-diet-34-43", cold_prompt)
        self.assertNotIn("0 and 5°C", cold_prompt)

    def test_fake_that_abstains_for_every_document_produces_not_in_corpus_in_phase_6(self) -> None:
        def abstaining_fake(
            doc_id: str,
            prompt: str,
            question: str,
            chunks: Sequence[Chunk],
        ) -> str:
            return json.dumps(
                {
                    "doc_id": doc_id,
                    "abstain": True,
                    "claims": [],
                }
            )

        generations = generate_per_document(
            question="How long can cooked ostrich stay in the refrigerator?",
            retrieved_chunks=self.retrieved,
            chat_model=abstaining_fake,
        )
        self.assertTrue(all(g.abstain for g in generations))

        response = check_and_assemble(
            generations=generations,
            retrieved_chunks=self.retrieved,
            searched=self.searched,
        )
        self.assertEqual(response.outcome, "not_in_corpus")
        self.assertEqual(response.sections, [])
        self.assertEqual(response.searched, self.searched)

    def test_fake_that_invents_chunk_id_produces_no_claim_after_citation_check(self) -> None:
        def inventing_fake(
            doc_id: str,
            prompt: str,
            question: str,
            chunks: Sequence[Chunk],
        ) -> str:
            return json.dumps(
                {
                    "doc_id": doc_id,
                    "abstain": False,
                    "claims": [
                        {
                            "text": "Hallucinated claim with invented chunk id.",
                            "chunk_ids": [f"{doc_id}-invented-999"],
                        }
                    ],
                }
            )

        generations = generate_per_document(
            question="What temperature should my fridge be?",
            retrieved_chunks=self.retrieved,
            chat_model=inventing_fake,
        )
        surviving_sections = filter_citations(generations, self.retrieved)
        self.assertEqual(surviving_sections, [])

        response = check_and_assemble(
            generations=generations,
            retrieved_chunks=self.retrieved,
            searched=self.searched,
        )
        self.assertEqual(response.outcome, "not_in_corpus")
        self.assertEqual(response.sections, [])

    def test_build_document_prompt_rejects_foreign_chunk(self) -> None:
        with self.assertRaises(ValueError):
            build_document_prompt(
                question="Test question",
                doc_id="cold-food-storage",
                chunks=[self.cold_chunk, self.fsa_chunk],
            )


if __name__ == "__main__":
    unittest.main()
