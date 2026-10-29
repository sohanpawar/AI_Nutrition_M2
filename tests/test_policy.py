"""Tests for the deterministic out-of-scope policy gate (Phase 1)."""

import unittest
from unittest.mock import patch

from src.policy.gate import (
    OUT_OF_SCOPE_MESSAGE,
    check,
    make_out_of_scope_response,
)
from src.retrieval import search as retrieval_search


class TestPolicyGate(unittest.TestCase):
    """Verify deterministic scope refusals and pass-throughs for corpus questions."""

    def test_exit_table_out_of_scope_questions(self) -> None:
        out_of_scope_questions = [
            # Required Phase 1 exit check questions
            "Is this rash from something I ate?",
            "How many calories should I eat to lose 5 kg?",
            "What should I weigh?",
            # Medical advice: diagnosis, cause of symptoms, treatment, medicine, food-caused condition
            "I have hives and swelling after dinner, what medicine should I take?",
            "Can you diagnose my stomach ache and nausea?",
            "What is the best treatment for acid reflux and bloating?",
            "Did something I eat cause my diarrhea and cramps?",
            "What antibiotic dosage treats a food infection?",
            "Do I have a food allergy if my throat feels itchy?",
            # Calorie targets, deficits, and daily energy allowances
            "What is a good calorie deficit for weight loss?",
            "What should my daily calorie allowance be?",
            "How many kcal do I need to eat to gain weight?",
            "What is my personal caloric target?",
            "Should I eat 1500 calories a day to lose weight?",
            # Body-weight targets, BMI goals, what someone should weigh
            "What is my ideal body weight?",
            "What BMI goal should I aim for?",
            "How much should someone weigh at my height?",
            "How can I lose 10 lbs quickly?",
            "Am I overweight or obese?",
            "How to lose weight fast?",
        ]
        for question in out_of_scope_questions:
            with self.subTest(question=question):
                self.assertEqual(check(question), "out_of_scope")

    def test_exit_table_allowed_questions(self) -> None:
        allowed_questions = [
            # Required Phase 1 exit check questions
            "What does WHO say about free sugars?",
            "How many calories are in an orange?",
            "What temperature should my fridge be?",
            # Other corpus and nutrient lookup questions that must pass the gate
            "What does a healthy diet include?",
            "How does the Eatwell Guide divide food groups?",
            "How should I handle raw meat when cooking?",
            "What temperature should my fridge be, and how long can leftovers stay there?",
            "Which oil should I use, and is it safe to reuse?",
            "What is the vitamin C content of an orange?",
            "What do the guidelines say about salt and saturated fat?",
        ]
        for question in allowed_questions:
            with self.subTest(question=question):
                self.assertEqual(check(question), "allow")

    def test_fixed_out_of_scope_response_envelope(self) -> None:
        response = make_out_of_scope_response()
        self.assertEqual(response.outcome, "out_of_scope")
        self.assertEqual(response.searched, [])
        self.assertEqual(response.sections, [])
        self.assertEqual(response.message, OUT_OF_SCOPE_MESSAGE)
        self.assertIn("professional", response.message.lower())
        for doc_id in (
            "who-healthy-diet",
            "fao-who-healthy-diets",
            "eatwell-guide",
            "cold-food-storage",
            "fsa-chill",
            "who-five-keys",
            "kitchen-companion",
        ):
            self.assertNotIn(doc_id, response.message)

    def test_out_of_scope_never_calls_retriever(self) -> None:
        """Patch the retriever and fail if an out_of_scope question calls it."""
        with patch.object(
            retrieval_search,
            "retrieve",
            side_effect=AssertionError("Retriever must not be called for out_of_scope questions"),
            create=True,
        ) as mock_retrieve:
            for question in (
                "Is this rash from something I ate?",
                "How many calories should I eat to lose 5 kg?",
                "What should I weigh?",
            ):
                outcome = check(question)
                self.assertEqual(outcome, "out_of_scope")
                if outcome == "out_of_scope":
                    response = make_out_of_scope_response()
                    self.assertEqual(response.searched, [])
            mock_retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
