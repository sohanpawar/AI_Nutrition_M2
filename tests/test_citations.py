"""Tests for Phase 6 citation verification."""

import unittest

from src import config
from src.answer.assemble import check_and_assemble
from src.answer.citations import filter_citations, verify_document_claims
from src.schemas import Chunk, Claim, DocumentGeneration, SearchedDocument


def _make_chunk(
    chunk_id: str,
    doc_id: str,
    document_name: str,
    publisher: str,
    year: int,
    section_heading: str,
    text: str = "Official guidance chunk text.",
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


class TestCitationCheck(unittest.TestCase):
    """Verify all Phase 6 citation check exit criteria."""

    def setUp(self) -> None:
        self.who_chunk = _make_chunk(
            chunk_id="who-healthy-diet-34-43",
            doc_id="who-healthy-diet",
            document_name="WHO Healthy diet, Fact sheet N°394",
            publisher="World Health Organization",
            year=2020,
            section_heading="Fats",
            text="Total fat should not exceed 30% of total energy intake.",
        )
        self.cold_chunk = _make_chunk(
            chunk_id="cold-food-storage-2-2-p2",
            doc_id="cold-food-storage",
            document_name="Cold Food Storage Chart",
            publisher="USDA Food Safety and Inspection Service",
            year=2023,
            section_heading="Cold Food Storage Chart",
            text="| Leftovers | Cooked meat or poultry | 3 to 4 days | 2 to 6 months |",
        )
        self.retrieved = [self.who_chunk, self.cold_chunk]
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

    def test_claim_citing_chunk_from_another_doc_id_is_dropped(self) -> None:
        # Generation for who-healthy-diet cites a cold-food-storage chunk
        gen = DocumentGeneration(
            doc_id="who-healthy-diet",
            abstain=False,
            claims=[
                Claim(
                    text="Leftovers keep 3 to 4 days in the refrigerator.",
                    chunk_ids=["cold-food-storage-2-2-p2"],
                )
            ],
        )
        section = verify_document_claims(gen, self.retrieved)
        self.assertIsNone(section)

    def test_claim_citing_unretrieved_chunk_id_is_dropped(self) -> None:
        gen = DocumentGeneration(
            doc_id="who-healthy-diet",
            abstain=False,
            claims=[
                Claim(
                    text="Free sugars should be reduced to less than 10% of total energy intake.",
                    chunk_ids=["who-healthy-diet-55-60"],  # not in self.retrieved
                )
            ],
        )
        section = verify_document_claims(gen, self.retrieved)
        self.assertIsNone(section)

    def test_claim_with_empty_chunk_ids_list_is_dropped(self) -> None:
        gen = DocumentGeneration(
            doc_id="who-healthy-diet",
            abstain=False,
            claims=[
                Claim(
                    text="Unsourced claim with no chunk ids.",
                    chunk_ids=[],
                ),
                Claim(
                    text="Total fat should not exceed 30% of total energy intake.",
                    chunk_ids=["who-healthy-diet-34-43"],
                ),
            ],
        )
        section = verify_document_claims(gen, self.retrieved)
        self.assertIsNotNone(section)
        assert section is not None
        self.assertEqual(len(section.claims), 1)
        self.assertEqual(
            section.claims[0].text,
            "Total fat should not exceed 30% of total energy intake.",
        )

    def test_when_only_claims_were_invalid_outcome_is_not_in_corpus_and_searched_non_empty(self) -> None:
        generations = [
            DocumentGeneration(
                doc_id="who-healthy-diet",
                abstain=False,
                claims=[
                    Claim(text="Invented id claim.", chunk_ids=["who-healthy-diet-999-999"]),
                    Claim(text="Empty ids claim.", chunk_ids=[]),
                ],
            ),
            DocumentGeneration(
                doc_id="cold-food-storage",
                abstain=False,
                claims=[
                    Claim(text="Cross-doc id claim.", chunk_ids=["who-healthy-diet-34-43"]),
                ],
            ),
        ]
        response = check_and_assemble(
            generations=generations,
            retrieved_chunks=self.retrieved,
            searched=self.searched,
        )
        self.assertEqual(response.outcome, "not_in_corpus")
        self.assertEqual(response.sections, [])
        self.assertGreater(len(response.searched), 0)
        self.assertEqual(response.searched, self.searched)
        self.assertIsNotNone(response.message)
        assert response.message is not None
        self.assertIn("does not cover", response.message)

    def test_citation_fields_match_chunk_record_even_if_claim_text_names_different_publisher(self) -> None:
        gen = DocumentGeneration(
            doc_id="who-healthy-diet",
            abstain=False,
            claims=[
                Claim(
                    text="According to Fictional Publisher (1999), total fat should be under 30%.",
                    chunk_ids=["who-healthy-diet-34-43"],
                    section_heading="Invented Section Heading",
                )
            ],
        )
        sections = filter_citations([gen], self.retrieved)
        self.assertEqual(len(sections), 1)
        sec = sections[0]
        self.assertEqual(sec.doc_id, "who-healthy-diet")
        self.assertEqual(sec.document_name, self.who_chunk.document_name)
        self.assertEqual(sec.publisher, "World Health Organization")
        self.assertEqual(sec.year, 2020)
        self.assertEqual(sec.source_url, config.source_url)
        self.assertEqual(sec.claims[0].section_heading, "Fats")


if __name__ == "__main__":
    unittest.main()
