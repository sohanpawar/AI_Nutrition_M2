# Dietary Guidance RAG Chatbot

Prototype retrieval-augmented chatbot that answers food, nutrition, and food-safety questions exclusively from seven official public dietary guidance documents (`blocks.jsonl`).

## Corpus Documents

| `doc_id` | Document | Publisher | Blocks | Chunks |
| --- | --- | --- | --- | --- |
| `who-healthy-diet` | WHO Healthy diet, Fact sheet N°394 | World Health Organization | 92 | 10 |
| `fao-who-healthy-diets` | What are healthy diets? | FAO and WHO | 98 | 13 |
| `eatwell-guide` | The Eatwell Guide | UK Government (Public Health England) | 120 | 26 |
| `cold-food-storage` | Cold Food Storage Chart | USDA Food Safety and Inspection Service | 3 | 3 |
| `fsa-chill` | How to chill, freeze and defrost food safely | UK Food Standards Agency | 45 | 5 |
| `who-five-keys` | Five Keys to Safer Food | World Health Organization | 174 | 26 |
| `kitchen-companion` | Kitchen Companion | USDA Food Safety and Inspection Service | 469 | 105 |
| **Total** | | | **1,001** | **188** |

## Repository Structure

- `corpus/manifest.yaml`: Closed manifest of the seven guidance documents, publishers, years, official publication URLs, and document aliases.
- `corpus/parsed/`: Grouped document blocks (`<doc_id>.json`) and structure-aware chunks (`<doc_id>.chunks.jsonl`).
- `corpus/index/`: Persisted vector index (`vectors.sqlite3`).
- `src/schemas.py`: Shared Pydantic models (`Block`, `Document`, `Chunk`, `Claim`, `Section`, `SearchedDocument`, `AskRequest`, `AskResponse`).
- `src/config.py`: Central configuration for URLs, retrieval thresholds (`score_floor`, `k`, `m`), chunk budgets (`chunk_token_budget`, `table_token_hard_cap`), and model IDs.
- `src/policy/gate.py`: Deterministic scope gate refusing medical advice, calorie targets, and weight targets.
- `src/ingest/`: Offline corpus download, validation, structure-aware chunking, and embedding index build.
- `src/retrieval/`: Alias resolution (`resolve.py`) and vector search with metadata filtering (`search.py`).
- `src/answer/`: Per-document generation, citation verification, and deterministic section assembly.
- `src/api/ask.py`: `POST /ask` FastAPI route.

## Chunking Strategy and Cost Notes

### Strategy & Budgets

Fixed-size character or token windows with overlap are **not** used. Instead, `src/ingest/chunk.py` walks each document's blocks in `ordinal` order and preserves structural citation units:

- **`chunk_token_budget = 500`** (whitespace-delimited token estimate):
  - A `heading` block opens a section (`section_heading` is the last entry of `heading_path`).
  - Following `paragraph` and `list_item` blocks sharing that `heading_path` join the section until the next heading changes the path or a `table` block appears.
  - When a section exceeds `chunk_token_budget`, it splits first on `list_item` boundaries and then on `paragraph` boundaries, repeating `section_heading` on every child chunk. A `list_item` is never cut in half.
- **`table_token_hard_cap = 800`**:
  - Each `table` block stays as a single chunk with its `header` attached unless it exceeds `table_token_hard_cap`.
  - Oversized tables are split by row groups, repeating the markdown `header` on every piece and setting `table_part` (`1`, `2`, ...). No table row is ever split across pieces.

### Split Tables in the Corpus

Out of 12 tables across the 7 documents, 10 stay whole as single chunks and **2 tables exceed `table_token_hard_cap = 800` and are split by row groups**:

1. **`cold-food-storage` (block ordinal `2`, 53 rows, ~1,058 tokens, section `Cold Food Storage Chart`)**:
   - Split into **2 pieces** (`cold-food-storage-2-2-p1` with `table_part=1` and `cold-food-storage-2-2-p2` with `table_part=2`).
   - Both pieces repeat the 4-column header (`Food | Type | Refrigerator [40°F (4°C) or below] | Freezer [0°F (-18°C) or below]`), keeping every food row together with both its refrigerator and freezer times.
2. **`eatwell-guide` (block ordinal `105`, 6 rows, ~925 tokens, section `8 tips for eating well`)**:
   - Split into **2 pieces** (`eatwell-guide-105-105-p1` with `table_part=1` and `eatwell-guide-105-105-p2` with `table_part=2`).
   - Both pieces repeat the 4-column header (`Food group | What’s included? | How much? | Tips`).

All 8 tables in `kitchen-companion` and the other 2 tables in `eatwell-guide` (ordinals `71` and `81`) are under `table_token_hard_cap` and remain single chunks.

### What Structure-Aware Chunking Costs

1. **Larger chunks and neighbouring advice in a hit:** Because entire sections and list recommendations stay together up to `chunk_token_budget = 500`, chunks are larger and fewer (188 chunks across 1,001 blocks) than a fixed-window splitter would produce. A narrow question about one recommendation can retrieve neighbouring items in the same section.
2. **Wide/long tables dilute single-row embeddings until split, and splitting repeats headers:** A full 53-row storage table buries a single leftover row in one vector. Splitting `cold-food-storage` and `eatwell-guide` ordinal `105` by row groups improves row-level retrieval while keeping each row's cells intact, at the cost of repeating the table header on every piece.
3. **Document length imbalance (`kitchen-companion` crowding):** `kitchen-companion` has 469 blocks and produces **105 of the 188 total chunks (55.9%)**, whereas `cold-food-storage` has 3 chunks and `fsa-chill` has 5 chunks. Without a per-document cap `m` during retrieval, `kitchen-companion` would crowd shorter documents out of the top-`k` window.

## Retrieval Thresholds (`src/config.py`)

Tuned against the diet and food-safety acceptance questions and nutrient/out-of-corpus queries:

- **`score_floor = 0.22`**: Minimum cosine similarity required for a chunk to be treated as evidence. Below `0.22` (e.g. `"What is the vitamin C content of an orange?"` at `0.148`, `"How many calories are in an orange?"` at `0.109`, or nonsense queries at `< 0.10`), all chunks are dropped so the retriever returns an empty chunk list and a search record naming all searched documents (`not_in_corpus`).
- **`k = 8`**: Maximum total chunks returned across the corpus for a single request.
- **`m = 3`**: Maximum chunks contributed by any single `doc_id`, preventing `kitchen-companion` (105 chunks) from filling the `k = 8` window on cross-document queries.
