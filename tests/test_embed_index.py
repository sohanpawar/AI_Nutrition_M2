"""Tests for Phase 4 embeddings and the persisted vector index."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src import config
from src.ingest import embed_index
from src.ingest.embed_index import (
    IndexError,
    VectorIndex,
    embed_texts,
    format_chunk_for_embedding,
    load_chunks_for_doc,
)


class TestEmbeddingsAndIndex(unittest.TestCase):
    """Verify all Phase 4 exit criteria."""

    def test_format_chunk_for_embedding_uses_heading_newline_body(self) -> None:
        chunks = load_chunks_for_doc("cold-food-storage")
        chunk = chunks[0]
        formatted = format_chunk_for_embedding(chunk)
        self.assertEqual(formatted, f"{chunk.section_heading}\n{chunk.text}")

    def test_leftover_refrigerator_time_returns_cold_food_storage_in_top_results(self) -> None:
        idx = VectorIndex(index_dir=config.INDEX_DIR)
        self.assertTrue(idx.is_ready(), "Persisted index under corpus/index/ must be ready")

        results = idx.query("leftover refrigerator time", top_k=5)
        self.assertGreater(len(results), 0)
        doc_ids = [chunk.doc_id for chunk, _score in results]
        self.assertIn(
            "cold-food-storage",
            doc_ids,
            f"Expected 'cold-food-storage' in top results, got {doc_ids}",
        )

    def test_filtered_query_for_eatwell_guide_returns_no_cold_storage_chunks(self) -> None:
        idx = VectorIndex(index_dir=config.INDEX_DIR)
        results = idx.query(
            "leftover refrigerator time",
            top_k=10,
            doc_id="eatwell-guide",
        )
        self.assertGreater(len(results), 0)
        for chunk, _score in results:
            self.assertEqual(chunk.doc_id, "eatwell-guide")
            self.assertNotEqual(chunk.doc_id, "cold-food-storage")

    def test_restarting_process_and_querying_does_not_re_embed_corpus(self) -> None:
        """Instantiating a fresh VectorIndex on existing corpus/index/ only embeds the 1 query string."""
        call_sizes: list[int] = []

        def tracking_embed(texts: list[str]) -> list[list[float]]:
            call_sizes.append(len(texts))
            return embed_texts(texts)

        restarted_idx = VectorIndex(index_dir=config.INDEX_DIR, embed_fn=tracking_embed)
        self.assertTrue(restarted_idx.is_ready())
        results = restarted_idx.query("leftover refrigerator time", top_k=5)
        self.assertGreater(len(results), 0)
        # Only 1 embedding call of length 1 (the query itself); 0 chunk re-embeddings
        self.assertEqual(call_sizes, [1])

    def test_deleting_one_doc_and_reingesting_leaves_other_six_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_index_dir = Path(tmp_dir) / "index"
            shutil.copytree(config.INDEX_DIR, tmp_index_dir)

            embedded_batches: list[int] = []

            def spy_embed(texts: list[str]) -> list[list[float]]:
                embedded_batches.append(len(texts))
                return embed_texts(texts)

            idx = VectorIndex(index_dir=tmp_index_dir, embed_fn=spy_embed)
            total_before = idx.count()
            self.assertEqual(total_before, 188)

            deleted = idx.delete_document("fsa-chill")
            self.assertEqual(deleted, 5)
            self.assertEqual(idx.count("fsa-chill"), 0)
            self.assertEqual(idx.count(), total_before - 5)

            # Other six documents are still present
            for other_id in (
                "who-healthy-diet",
                "fao-who-healthy-diets",
                "eatwell-guide",
                "cold-food-storage",
                "who-five-keys",
                "kitchen-companion",
            ):
                self.assertGreater(idx.count(other_id), 0)

            reingested = idx.ingest_document("fsa-chill")
            self.assertEqual(reingested, 5)
            self.assertEqual(idx.count("fsa-chill"), 5)
            self.assertEqual(idx.count(), total_before)
            # Only the 5 chunks of fsa-chill were embedded during re-ingest
            self.assertEqual(embedded_batches, [5])

    def test_refuses_to_build_index_when_manifest_chunk_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            incomplete_parsed_dir = tmp_path / "parsed"
            tmp_index_dir = tmp_path / "index"
            shutil.copytree(config.PARSED_DIR, incomplete_parsed_dir)

            # Remove one manifest document's .chunks.jsonl file
            (incomplete_parsed_dir / "who-five-keys.chunks.jsonl").unlink()

            idx = VectorIndex(index_dir=tmp_index_dir)
            with self.assertRaises(IndexError) as ctx:
                idx.rebuild_all(parsed_dir=incomplete_parsed_dir)

            self.assertIn("who-five-keys", str(ctx.exception))
            self.assertEqual(idx.count(), 0)


if __name__ == "__main__":
    unittest.main()
