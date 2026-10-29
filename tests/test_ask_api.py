"""Tests for Phase 8 POST /ask API endpoint."""

from collections.abc import Sequence
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import ask as ask_module
from src.api.ask import app, handle_ask
from src.ingest.embed_index import VectorIndex
from src.schemas import AskRequest, Chunk


class TestAskAPI(unittest.TestCase):
    """Verify all Phase 8 HTTP exit checks."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_rash_question_returns_200_out_of_scope_without_calling_retriever(self) -> None:
        with patch.object(
            ask_module,
            "retrieve",
            side_effect=AssertionError("Retriever must not be called for out_of_scope questions"),
        ) as mock_retrieve:
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
            mock_retrieve.assert_not_called()

    def test_calorie_target_question_returns_200_out_of_scope_with_empty_searched(self) -> None:
        with patch.object(
            ask_module,
            "retrieve",
            side_effect=AssertionError("Retriever must not be called for out_of_scope questions"),
        ) as mock_retrieve:
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
            mock_retrieve.assert_not_called()

    def test_unknown_document_filter_returns_400_and_index_not_queried(self) -> None:
        with patch.object(
            VectorIndex,
            "query",
            side_effect=AssertionError("Index must not be queried for an unknown document_filter"),
        ) as mock_query:
            resp = self.client.post(
                "/ask",
                json={
                    "question": "What does a healthy diet include?",
                    "document_filter": "unknown-document-id",
                },
            )
            self.assertEqual(resp.status_code, 400)
            mock_query.assert_not_called()

    def test_in_scope_question_when_index_absent_returns_503_and_does_not_call_generator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_index = VectorIndex(index_dir=Path(tmp_dir) / "missing_index")
            with patch.object(
                ask_module,
                "generate_per_document",
                side_effect=AssertionError("Generator must not be called when index is missing"),
            ) as mock_gen:
                with patch.object(
                    ask_module,
                    "retrieve",
                    wraps=lambda question, document_filter=None, index=None, manifest_path=ask_module.config.MANIFEST_PATH: ask_module.retrieve.__wrapped__(
                        question=question,
                        document_filter=document_filter,
                        index=empty_index,
                        manifest_path=manifest_path,
                    )
                    if hasattr(ask_module.retrieve, "__wrapped__")
                    else None,
                ):
                    pass

                # Also verify via TestClient with VectorIndex pointed to the absent index directory
                with patch(
                    "src.retrieval.search.VectorIndex",
                    return_value=empty_index,
                ):
                    resp = self.client.post(
                        "/ask",
                        json={"question": "What does a healthy diet include?"},
                    )
                    self.assertEqual(resp.status_code, 503)
                    mock_gen.assert_not_called()

    def test_in_scope_question_under_score_floor_returns_200_not_in_corpus_with_searched_list(self) -> None:
        resp = self.client.post(
            "/ask",
            json={"question": "What is the vitamin C content of an orange?"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["outcome"], "not_in_corpus")
        self.assertEqual(body["sections"], [])
        self.assertEqual(len(body["searched"]), 7)
        self.assertTrue(body["message"])

    def test_eatwell_food_group_question_returns_200_grounded_answer_with_only_eatwell_guide(self) -> None:
        def fake_generator(
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
                            "text": "The Eatwell Guide divides the foods and drinks we consume into five main groups.",
                            "chunk_ids": [chunks[0].chunk_id],
                        }
                    ],
                }
            )

        with patch.object(
            ask_module,
            "default_chat_model",
            side_effect=fake_generator,
        ):
            resp = self.client.post(
                "/ask",
                json={"question": "How does the Eatwell Guide divide food groups?"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["outcome"], "grounded_answer")
            self.assertIsNone(body["message"])
            self.assertEqual(len(body["sections"]), 1)
            self.assertEqual(body["sections"][0]["doc_id"], "eatwell-guide")
            self.assertEqual(body["sections"][0]["document_name"], "The Eatwell Guide")
            self.assertEqual([s["doc_id"] for s in body["searched"]], ["eatwell-guide"])

    def test_chat_client_served_and_three_exit_questions_pass(self) -> None:
        # 1. GET / serves the static chat client with question input, filter, and stateless fetch('/ask')
        html_resp = self.client.get("/")
        self.assertEqual(html_resp.status_code, 200)
        self.assertIn("text/html", html_resp.headers["content-type"])
        html_text = html_resp.text
        self.assertIn('id="question-input"', html_text)
        self.assertIn('id="filter-input"', html_text)
        self.assertIn("grounded_answer", html_text)
        self.assertIn("not_in_corpus", html_text)
        self.assertIn("out_of_scope", html_text)

        # 2. Diet question -> grounded_answer with cited section(s) whose link is the dataset page
        diet_resp = self.client.post(
            "/ask",
            json={"question": "What does a healthy diet include?"},
        )
        self.assertEqual(diet_resp.status_code, 200)
        diet_body = diet_resp.json()
        self.assertEqual(diet_body["outcome"], "grounded_answer")
        self.assertGreaterEqual(len(diet_body["sections"]), 1)
        for sec in diet_body["sections"]:
            self.assertEqual(sec["source_url"], ask_module.config.source_url)
            self.assertTrue(sec["document_name"])
            self.assertTrue(sec["publisher"])
            self.assertGreater(sec["year"], 0)
            self.assertGreaterEqual(len(sec["claims"]), 1)

        # 3. "What is the vitamin C content of an orange?" -> corpus refusal naming searched documents
        vitc_resp = self.client.post(
            "/ask",
            json={"question": "What is the vitamin C content of an orange?"},
        )
        self.assertEqual(vitc_resp.status_code, 200)
        vitc_body = vitc_resp.json()
        self.assertEqual(vitc_body["outcome"], "not_in_corpus")
        self.assertEqual(len(vitc_body["searched"]), 7)
        self.assertEqual(vitc_body["sections"], [])
        self.assertTrue(vitc_body["message"])

        # 4. Rash question -> professional-referral message and no searched documents
        rash_resp = self.client.post(
            "/ask",
            json={"question": "Is this rash from something I ate?"},
        )
        self.assertEqual(rash_resp.status_code, 200)
        rash_body = rash_resp.json()
        self.assertEqual(rash_body["outcome"], "out_of_scope")
        self.assertEqual(rash_body["searched"], [])
        self.assertEqual(rash_body["sections"], [])
        self.assertIn("professional", rash_body["message"].lower())

        # Confirm the two refusal texts are different
        self.assertNotEqual(vitc_body["message"], rash_body["message"])


if __name__ == "__main__":
    unittest.main()
