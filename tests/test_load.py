"""Tests for Phase 2 corpus download, manifest validation, and block grouping."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests
import yaml

from src import config
from src.ingest.load import (
    CorpusLoadError,
    download_blocks,
    group_and_write_documents,
    load_manifest,
    parse_blocks_file,
)
from src.schemas import Document


class TestCorpusLoader(unittest.TestCase):
    """Verify Phase 2 exit criteria for corpus/blocks.jsonl and corpus/parsed/<doc_id>.json."""

    def test_real_corpus_files_and_metadata(self) -> None:
        self.assertTrue(config.BLOCKS_PATH.exists(), "corpus/blocks.jsonl must exist")

        manifest_entries = load_manifest(config.MANIFEST_PATH)
        expected_ids = [entry["doc_id"] for entry in manifest_entries]
        self.assertEqual(len(expected_ids), 7)

        raw_blocks = parse_blocks_file(config.BLOCKS_PATH)
        total_parsed_blocks = 0

        for entry in manifest_entries:
            doc_id = entry["doc_id"]
            parsed_path = config.PARSED_DIR / f"{doc_id}.json"
            self.assertTrue(parsed_path.exists(), f"Missing parsed file: {parsed_path}")

            doc = Document.model_validate_json(parsed_path.read_text(encoding="utf-8"))
            self.assertEqual(doc.doc_id, doc_id)
            self.assertEqual(doc.document_name, entry["document_name"])
            self.assertTrue(doc.publisher.strip())
            self.assertEqual(doc.publisher, entry["publisher"])
            self.assertIsInstance(doc.year, int)
            self.assertEqual(doc.year, entry["year"])
            self.assertEqual(doc.source_url, entry["source_url"])
            self.assertNotIn("blocks.jsonl", doc.source_url)
            self.assertTrue(doc.retrieval_date.strip())
            self.assertTrue(len(doc.aliases) > 0)
            self.assertTrue(len(doc.blocks) > 0)

            ordinals = [b.ordinal for b in doc.blocks]
            self.assertEqual(ordinals, sorted(ordinals), f"Blocks in {doc_id} are not sorted by ordinal")
            total_parsed_blocks += len(doc.blocks)

        self.assertEqual(total_parsed_blocks, len(raw_blocks))

    def test_rerunning_group_step_does_not_use_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "parsed"
            with patch.object(
                requests,
                "get",
                side_effect=AssertionError("Network must not be called when grouping stored blocks.jsonl"),
            ) as mock_get:
                docs = group_and_write_documents(
                    blocks_path=config.BLOCKS_PATH,
                    manifest_path=config.MANIFEST_PATH,
                    parsed_dir=out_dir,
                )
                mock_get.assert_not_called()
                self.assertEqual(len(docs), 7)
                for doc_id in docs:
                    self.assertTrue((out_dir / f"{doc_id}.json").exists())

    def test_eighth_unknown_document_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bad_blocks = tmp_path / "blocks.jsonl"
            out_dir = tmp_path / "parsed"

            lines = config.BLOCKS_PATH.read_text(encoding="utf-8").splitlines()
            extra_block = {
                "document_id": "eighth-unknown-doc",
                "ordinal": 1,
                "type": "paragraph",
                "text": "Unapproved document block.",
                "heading_path": ["Root"],
            }
            lines.append(json.dumps(extra_block))
            bad_blocks.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(CorpusLoadError) as ctx:
                group_and_write_documents(
                    blocks_path=bad_blocks,
                    manifest_path=config.MANIFEST_PATH,
                    parsed_dir=out_dir,
                )
            self.assertIn("eighth-unknown-doc", str(ctx.exception))
            self.assertFalse(out_dir.exists())

    def test_missing_manifest_document_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            incomplete_blocks = tmp_path / "blocks.jsonl"
            out_dir = tmp_path / "parsed"

            kept_lines = [
                line
                for line in config.BLOCKS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["document_id"] != "fsa-chill"
            ]
            incomplete_blocks.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

            with self.assertRaises(CorpusLoadError) as ctx:
                group_and_write_documents(
                    blocks_path=incomplete_blocks,
                    manifest_path=config.MANIFEST_PATH,
                    parsed_dir=out_dir,
                )
            self.assertIn("fsa-chill", str(ctx.exception))
            self.assertFalse(out_dir.exists())

    def test_blank_year_in_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bad_manifest = tmp_path / "manifest.yaml"
            manifest_data = yaml.safe_load(config.MANIFEST_PATH.read_text(encoding="utf-8"))
            manifest_data["documents"][0]["year"] = ""
            bad_manifest.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

            with self.assertRaises(CorpusLoadError) as ctx:
                load_manifest(bad_manifest)
            self.assertIn("year", str(ctx.exception))

    def test_failed_download_does_not_write_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "blocks.jsonl"
            with patch.object(requests, "get", side_effect=requests.RequestException("network down")):
                with self.assertRaises(CorpusLoadError):
                    download_blocks(url=config.corpus_url, dest_path=dest)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
