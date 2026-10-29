"""Dietary Guidance RAG Chatbot configuration."""

from pathlib import Path
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT_DIR / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.yaml"
BLOCKS_PATH = CORPUS_DIR / "blocks.jsonl"
PARSED_DIR = CORPUS_DIR / "parsed"
INDEX_DIR = CORPUS_DIR / "index"

corpus_url: str = "https://raw.githubusercontent.com/saksham20189575/health-data/main/blocks.jsonl"
source_url: str = "https://github.com/saksham20189575/health-data/blob/main/blocks.jsonl"

score_floor: float = 0.22
k: int = 8
m: int = 3

chunk_token_budget: int = 500
table_token_hard_cap: int = 800

embedding_model_id: str = "text-embedding-3-small"
chat_model_id: str = "gpt-4o-mini"


class Settings(BaseModel):
    """Centralized configuration for ingest, retrieval, and generation."""

    corpus_url: str = corpus_url
    source_url: str = source_url
    score_floor: float = score_floor
    k: int = k
    m: int = m
    chunk_token_budget: int = chunk_token_budget
    table_token_hard_cap: int = table_token_hard_cap
    embedding_model_id: str = embedding_model_id
    chat_model_id: str = chat_model_id
    manifest_path: Path = MANIFEST_PATH
    blocks_path: Path = BLOCKS_PATH
    parsed_dir: Path = PARSED_DIR
    index_dir: Path = INDEX_DIR


settings = Settings()
