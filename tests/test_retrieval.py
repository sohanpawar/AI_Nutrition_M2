"""Tests for Phase 5 document alias resolution and vector retrieval."""

import inspect
import unittest
from unittest.mock import patch

from src import config
from src.ingest.embed_index import VectorIndex
from src.retrieval import search as search_module
from src.retrieval.resolve import (
    InvalidDocumentFilterError,
    detect_alias_in_question,
    resolve_document,
    resolve_explicit_filter,
)
from src.retrieval.search import retrieve


class TestDocumentResolverAndRetriever(unittest.TestCase):
    """Verify all Phase 5 exit checks."""

    def test_required_manifest_aliases_resolve(self) -> None:
        alias_expectations = [
            ("Eatwell", "eatwell-guide"),
            ("the Eatwell Guide", "eatwell-guide"),
            ("cold food storage", "cold-food-storage"),
            ("storage chart", "cold-food-storage"),
            ("five keys", "who-five-keys"),
            ("kitchen companion", "kitchen-companion"),
            ("FSA chill", "fsa-chill"),
            ("how to chill, freeze and defrost", "fsa-chill"),
            ("WHO healthy diet", "who-healthy-diet"),
            ("fact sheet 394", "who-healthy-diet"),
            ("what are healthy diets", "fao-who-healthy-diets"),
            ("FAO WHO", "fao-who-healthy-diets"),
        ]
        for alias, expected_doc_id in alias_expectations:
            with self.subTest(alias=alias):
                self.assertEqual(resolve_explicit_filter(alias), expected_doc_id)
                self.assertEqual(
                    detect_alias_in_question(f"What does {alias} say about food?"),
                    expected_doc_id,
                )

    def test_explicit_document_filter_wins_over_question_alias(self) -> None:
        resolved = resolve_document(
            question="How does the Eatwell Guide divide food groups?",
            document_filter="who-healthy-diet",
        )
        self.assertEqual(resolved, "who-healthy-diet")

    def test_eatwell_question_resolves_and_returns_only_eatwell_guide(self) -> None:
        result = retrieve("How does the Eatwell Guide divide food groups?")
        self.assertEqual(result.resolved_doc_id, "eatwell-guide")
        self.assertEqual([s.doc_id for s in result.searched], ["eatwell-guide"])
        self.assertGreater(len(result.chunks), 0)
        for chunk in result.chunks:
            self.assertEqual(chunk.doc_id, "eatwell-guide")

    def test_global_query_returns_multiple_doc_ids_and_respects_per_doc_cap_m(self) -> None:
        result = retrieve(
            "What temperature should my fridge be, and how long can leftovers stay there?"
        )
        self.assertIsNone(result.resolved_doc_id)
        self.assertEqual(len(result.searched), 7)
        self.assertLessEqual(len(result.chunks), config.k)

        doc_counts: dict[str, int] = {}
        for chunk in result.chunks:
            doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1

        self.assertGreater(len(doc_counts), 1, f"Expected multiple doc_ids, got {doc_counts}")
        for doc_id, count in doc_counts.items():
            self.assertLessEqual(
                count,
                config.m,
                f"doc_id {doc_id!r} contributed {count} chunks, exceeding m={config.m}",
            )

    def test_nonsense_in_scope_question_under_floor_returns_empty_chunks_and_all_seven_searched(self) -> None:
        result = retrieve("Can astronauts grow purple crystals on Jupiter?")
        self.assertEqual(result.chunks, [])
        self.assertEqual(len(result.searched), 7)
        expected_ids = [
            "who-healthy-diet",
            "fao-who-healthy-diets",
            "eatwell-guide",
            "cold-food-storage",
            "fsa-chill",
            "who-five-keys",
            "kitchen-companion",
        ]
        self.assertEqual([s.doc_id for s in result.searched], expected_ids)
        for s in result.searched:
            self.assertTrue(s.document_name.strip())
            self.assertTrue(s.publisher.strip())

    def test_unknown_document_filter_raises_and_does_not_query_index(self) -> None:
        with patch.object(
            VectorIndex,
            "query",
            side_effect=AssertionError("Index must not be queried for an unknown document_filter"),
        ) as mock_query:
            with self.assertRaises(InvalidDocumentFilterError):
                retrieve(
                    question="What does a healthy diet include?",
                    document_filter="not-a-document",
                )
            mock_query.assert_not_called()

    def test_topic_word_fridge_does_not_resolve(self) -> None:
        # In a question, "fridge" alone must not select cold-food-storage
        self.assertIsNone(detect_alias_in_question("What temperature should my fridge be?"))
        # As an explicit document_filter, "fridge" is rejected and does not query the index
        with patch.object(
            VectorIndex,
            "query",
            side_effect=AssertionError("Index must not be queried when document_filter is 'fridge'"),
        ) as mock_query:
            with self.assertRaises(InvalidDocumentFilterError):
                retrieve(
                    question="What temperature should my fridge be?",
                    document_filter="fridge",
                )
            mock_query.assert_not_called()

    def test_thresholds_come_from_config_not_hardcoded_in_search(self) -> None:
        sig = inspect.signature(retrieve)
        self.assertIsNone(sig.parameters["score_floor"].default)
        self.assertIsNone(sig.parameters["k"].default)
        self.assertIsNone(sig.parameters["m"].default)


if __name__ == "__main__":
    unittest.main()
