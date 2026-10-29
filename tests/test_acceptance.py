"""Acceptance test suite against the ingested corpus and generator (Phase 10)."""

import re
import unittest

from fastapi.testclient import TestClient

from src import config
from src.api.ask import app, handle_ask
from src.ingest.embed_index import load_chunks_for_doc
from src.ingest.load import load_manifest
from src.retrieval.search import retrieve
from src.schemas import AskRequest

DIET_IDS: frozenset[str] = frozenset(
    {"who-healthy-diet", "fao-who-healthy-diets", "eatwell-guide"}
)
SAFETY_IDS_OIL: frozenset[str] = frozenset(
    {"who-five-keys", "kitchen-companion", "fsa-chill"}
)
ALL_MANIFEST_IDS: tuple[str, ...] = (
    "who-healthy-diet",
    "fao-who-healthy-diets",
    "eatwell-guide",
    "cold-food-storage",
    "fsa-chill",
    "who-five-keys",
    "kitchen-companion",
)


def _load_all_corpus_chunks_by_id() -> dict[str, str]:
    chunks_by_id: dict[str, str] = {}
    for doc_id in ALL_MANIFEST_IDS:
        for chunk in load_chunks_for_doc(doc_id):
            chunks_by_id[chunk.chunk_id] = chunk.text
    return chunks_by_id


class TestAcceptanceSet(unittest.TestCase):
    """Verify all Phase 10 acceptance questions and citation integrity checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.corpus_chunks_by_id = _load_all_corpus_chunks_by_id()

    def _assert_grounded_citations_in_retrieval_set(self, question: str) -> dict:
        retrieval_result = retrieve(question)
        retrieved_ids = {c.chunk_id for c in retrieval_result.chunks}
        retrieved_doc_by_chunk = {c.chunk_id: c.doc_id for c in retrieval_result.chunks}

        resp = self.client.post("/ask", json={"question": question})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["outcome"], "grounded_answer")
        self.assertIsNone(body["message"])
        self.assertGreater(len(body["sections"]), 0)

        for sec in body["sections"]:
            expected_url = next(
                entry["source_url"]
                for entry in load_manifest(config.MANIFEST_PATH)
                if entry["doc_id"] == sec["doc_id"]
            )
            self.assertEqual(sec["source_url"], expected_url)
            self.assertNotIn("blocks.jsonl", sec["source_url"])
            self.assertTrue(sec["document_name"])
            self.assertTrue(sec["publisher"])
            self.assertIsInstance(sec["year"], int)
            self.assertGreater(len(sec["claims"]), 0)
            for claim in sec["claims"]:
                self.assertTrue(claim["text"].strip())
                self.assertTrue(claim["section_heading"])
                self.assertGreater(len(claim["chunk_ids"]), 0)
                for cid in claim["chunk_ids"]:
                    self.assertIn(
                        cid,
                        retrieved_ids,
                        f"Claim cited chunk_id {cid!r} not in retrieval set {retrieved_ids}",
                    )
                    self.assertEqual(
                        retrieved_doc_by_chunk[cid],
                        sec["doc_id"],
                        f"Chunk {cid!r} belongs to {retrieved_doc_by_chunk[cid]!r}, not section {sec['doc_id']!r}",
                    )
        return body

    def test_1_what_does_a_healthy_diet_include(self) -> None:
        question = "What does a healthy diet include?"
        body = self._assert_grounded_citations_in_retrieval_set(question)
        doc_ids = [sec["doc_id"] for sec in body["sections"]]

        # who-healthy-diet and fao-who-healthy-diets required; eatwell-guide allowed only as its own section
        self.assertIn("who-healthy-diet", doc_ids)
        self.assertIn("fao-who-healthy-diets", doc_ids)
        for doc_id in doc_ids:
            self.assertIn(
                doc_id,
                DIET_IDS,
                f"Unexpected doc_id {doc_id!r} for healthy diet question",
            )
        # Ensure each section is a single distinct document
        self.assertEqual(len(doc_ids), len(set(doc_ids)))

    def test_2_how_does_the_eatwell_guide_divide_food_groups(self) -> None:
        question = "How does the Eatwell Guide divide food groups?"
        body = self._assert_grounded_citations_in_retrieval_set(question)
        doc_ids = [sec["doc_id"] for sec in body["sections"]]
        self.assertEqual(doc_ids, ["eatwell-guide"])

    def test_3_how_should_i_handle_raw_meat_when_cooking(self) -> None:
        question = "How should I handle raw meat when cooking?"
        body = self._assert_grounded_citations_in_retrieval_set(question)
        doc_ids = [sec["doc_id"] for sec in body["sections"]]

        # who-five-keys required; kitchen-companion allowed as its own section
        self.assertIn("who-five-keys", doc_ids)
        allowed = {"who-five-keys", "kitchen-companion"}
        for doc_id in doc_ids:
            self.assertIn(
                doc_id,
                allowed,
                f"Unexpected doc_id {doc_id!r} for raw meat handling question",
            )
        self.assertEqual(len(doc_ids), len(set(doc_ids)))

    def test_4_fridge_temperature_and_leftovers_time(self) -> None:
        question = "What temperature should my fridge be, and how long can leftovers stay there?"
        body = self._assert_grounded_citations_in_retrieval_set(question)
        doc_ids = [sec["doc_id"] for sec in body["sections"]]

        # cold-food-storage and fsa-chill only where each states the fact
        allowed = {"cold-food-storage", "fsa-chill"}
        self.assertTrue(set(doc_ids).issubset(allowed), f"Unexpected doc_ids: {doc_ids}")
        self.assertIn("cold-food-storage", doc_ids)
        self.assertIn("fsa-chill", doc_ids)

        # Fail if a hold-time claim's cited chunk does not state that hold time
        hold_time_pattern = re.compile(
            r"\b(\d+\s+to\s+\d+\s+(?:days?|months?)|(?:two|2|3|4)\s+days?|\d+\s+months?)\b",
            re.IGNORECASE,
        )
        for sec in body["sections"]:
            for claim in sec["claims"]:
                matches = hold_time_pattern.findall(claim["text"])
                for hold_phrase in matches:
                    cited_texts = " ".join(
                        self.corpus_chunks_by_id[cid] for cid in claim["chunk_ids"]
                    ).lower()
                    self.assertIn(
                        hold_phrase.lower(),
                        cited_texts,
                        f"Hold time {hold_phrase!r} in claim {claim['text']!r} not found in cited chunks {claim['chunk_ids']}",
                    )

    def test_5_which_oil_should_i_use_and_is_it_safe_to_reuse(self) -> None:
        question = "Which oil should I use, and is it safe to reuse?"
        body = self._assert_grounded_citations_in_retrieval_set(question)
        allowed = DIET_IDS | SAFETY_IDS_OIL

        for sec in body["sections"]:
            self.assertIn(sec["doc_id"], allowed)
            # Fail if any one section's chunk_ids span a diet doc_id and a safety doc_id
            chunk_doc_ids: set[str] = set()
            for claim in sec["claims"]:
                for cid in claim["chunk_ids"]:
                    for candidate_doc in ALL_MANIFEST_IDS:
                        if cid.startswith(f"{candidate_doc}-"):
                            chunk_doc_ids.add(candidate_doc)
            spans_diet = bool(chunk_doc_ids & DIET_IDS)
            spans_safety = bool(chunk_doc_ids & SAFETY_IDS_OIL)
            self.assertFalse(
                spans_diet and spans_safety,
                f"Section {sec['doc_id']!r} spans both diet and safety chunk_ids: {chunk_doc_ids}",
            )

    def test_6_how_many_calories_should_i_eat_to_lose_5_kg(self) -> None:
        resp = self.client.post(
            "/ask",
            json={"question": "How many calories should I eat to lose 5 kg?"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["outcome"], "out_of_scope")
        self.assertEqual(body["searched"], [])
        self.assertEqual(body["sections"], [])
        self.assertTrue(body["message"])

    def test_7_what_is_the_vitamin_c_content_of_an_orange(self) -> None:
        resp = self.client.post(
            "/ask",
            json={"question": "What is the vitamin C content of an orange?"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["outcome"], "not_in_corpus")
        self.assertEqual(body["sections"], [])
        self.assertEqual([s["doc_id"] for s in body["searched"]], list(ALL_MANIFEST_IDS))
        self.assertTrue(body["message"])
        # No numeric vitamin C claim in message
        self.assertIsNone(re.search(r"\b\d+\s*(?:mg|milligrams?|g|mcg)\b", body["message"], re.I))

    def test_8_is_this_rash_from_something_i_ate(self) -> None:
        resp = self.client.post(
            "/ask",
            json={"question": "Is this rash from something I ate?"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["outcome"], "out_of_scope")
        self.assertEqual(body["searched"], [])
        self.assertEqual(body["sections"], [])
        self.assertTrue(body["message"])

    def test_manifest_has_exactly_seven_documents_and_readme_has_cost_notes(self) -> None:
        manifest_entries = load_manifest(config.MANIFEST_PATH)
        self.assertEqual([e["doc_id"] for e in manifest_entries], list(ALL_MANIFEST_IDS))

        readme_text = (config.ROOT_DIR / "README.md").read_text(encoding="utf-8")
        for required_token in (
            "chunk_token_budget",
            "table_token_hard_cap",
            "score_floor",
            "cold-food-storage-2-2-p1",
            "eatwell-guide-105-105-p1",
            "kitchen-companion",
            "188",
        ):
            self.assertIn(required_token, readme_text)


if __name__ == "__main__":
    unittest.main()
