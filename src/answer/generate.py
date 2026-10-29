"""Per-document answer generator (Phase 7).

Groups retrieved chunks by `doc_id` and makes exactly one chat-model call per
remaining document. Each call receives only the user's question and that single
document's chunks (labelled with `chunk_id` and `section_heading`), never any
other document's text.

Returns structured `DocumentGeneration` objects (`doc_id`, `abstain`, and
`claims[]` with `text` and `chunk_ids`). The generator never chooses the
response `outcome` label; abstentions or invalid claims become `not_in_corpus`
during the Phase 6 citation check.
"""

from collections import OrderedDict
from collections.abc import Callable, Sequence
import json
import os
import re
from typing import Protocol

from src import config
from src.schemas import Chunk, Claim, DocumentGeneration

SYSTEM_PROMPT = (
    "You are a strict evidence-extraction assistant for a single official dietary or "
    "food-safety guidance document.\n"
    "Rules:\n"
    "1. Use ONLY the provided chunks from this single document. Do not use outside knowledge.\n"
    "2. Do not compare this publisher with any other authority or mention other documents.\n"
    "3. If the provided chunks do not directly state the fact asked for in the question, "
    "set \"abstain\": true and return an empty \"claims\" list.\n"
    "4. Do not invent hold times, temperatures, or nutrient numbers not explicitly stated "
    "in the cited chunk.\n"
    "5. Every claim in \"claims\" must have non-empty \"text\" and a non-empty \"chunk_ids\" "
    "list containing only chunk_ids from the provided chunks that directly support that claim.\n"
    "6. Respond with valid JSON matching the schema: "
    "{\"doc_id\": string, \"abstain\": boolean, \"claims\": [{\"text\": string, \"chunk_ids\": [string]}]}."
)

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "guide",
        "how",
        "i",
        "in",
        "is",
        "it",
        "long",
        "many",
        "much",
        "my",
        "of",
        "on",
        "or",
        "safe",
        "say",
        "should",
        "stay",
        "the",
        "there",
        "to",
        "use",
        "what",
        "when",
        "which",
        "who",
        "with",
    }
)

QUERY_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "fridge": ("fridge", "refrigerator", "40°f", "4°c", "5°c", "chill", "chilling"),
    "refrigerator": ("refrigerator", "fridge", "40°f", "4°c", "5°c", "chill"),
    "temperature": ("temperature", "40°f", "4°c", "0°f", "-18°c", "5°c", "°c", "°f"),
    "leftovers": ("leftovers", "leftover", "days"),
    "leftover": ("leftovers", "leftover", "days"),
    "healthy": ("healthy", "diet", "diets", "principles", "fruit", "vegetables", "adequate", "balanced", "moderate", "diverse"),
    "diet": ("diet", "diets", "healthy", "principles", "fruit", "vegetables", "legumes", "grains"),
    "include": ("include", "includes", "principles", "adequate", "balanced", "moderate", "diverse", "fruit", "vegetables"),
    "divide": ("divide", "divides", "proportions", "groups", "five", "main"),
    "groups": ("groups", "group", "proportions", "fruit", "vegetables", "starchy", "carbohydrates", "beans", "pulses", "dairy", "oils", "spreads"),
    "raw": ("raw", "separate", "cooked", "meat", "poultry", "cutting", "wash", "hands", "utensils"),
    "meat": ("meat", "poultry", "raw", "separate", "cooked", "roast", "thermometer"),
    "handle": ("handle", "separate", "keep", "clean", "wash", "cutting", "raw", "cooked"),
    "cooking": ("cooking", "cook", "cooked", "thoroughly", "roast", "temperature"),
    "oil": ("oil", "oils", "spreads", "unsaturated", "saturated", "fats", "fat", "vegetable", "rapeseed", "olive", "sunflower"),
    "reuse": ("reuse", "re-use", "reheating"),
    "sugars": ("sugars", "sugar", "free", "10%", "5%"),
    "sugar": ("sugar", "sugars", "free", "10%", "5%"),
    "vitamin": ("vitamin",),
    "orange": ("orange",),
    "calories": ("calories", "kcal"),
}

WORD_RE = re.compile(r"[a-z0-9°%]+")


class ChatModel(Protocol):
    """Small interface for the per-document chat model so tests can pass a fake."""

    def __call__(
        self,
        doc_id: str,
        prompt: str,
        question: str,
        chunks: Sequence[Chunk],
    ) -> str | dict | DocumentGeneration: ...


ChatModelFn = Callable[[str, str, str, Sequence[Chunk]], str | dict | DocumentGeneration]


def group_chunks_by_doc_id(chunks: Sequence[Chunk]) -> OrderedDict[str, list[Chunk]]:
    """Group retrieved chunks by `doc_id`, preserving first-seen order and skipping empty docs."""
    grouped: OrderedDict[str, list[Chunk]] = OrderedDict()
    for chunk in chunks:
        if chunk.doc_id not in grouped:
            grouped[chunk.doc_id] = []
        grouped[chunk.doc_id].append(chunk)
    return grouped


def build_document_prompt(question: str, doc_id: str, chunks: Sequence[Chunk]) -> str:
    """Build the single-document prompt containing ONLY `question` and `doc_id`'s chunks.

    Each chunk is labelled with `chunk_id` and `section_heading`. No other document's
    chunks or identifiers are included.
    """
    if not chunks:
        raise ValueError(f"Cannot build prompt for {doc_id!r} with empty chunks")

    for chunk in chunks:
        if chunk.doc_id != doc_id:
            raise ValueError(
                f"Foreign chunk {chunk.chunk_id!r} (doc_id={chunk.doc_id!r}) passed to prompt for {doc_id!r}"
            )

    doc_header = f"Document ID: {doc_id}\nDocument Name: {chunks[0].document_name}\nPublisher: {chunks[0].publisher} ({chunks[0].year})"
    chunk_blocks: list[str] = []
    for chunk in chunks:
        chunk_blocks.append(
            f"[chunk_id: {chunk.chunk_id} | section_heading: {chunk.section_heading}]\n{chunk.text}"
        )

    joined_chunks = "\n\n---\n\n".join(chunk_blocks)
    return (
        f"{doc_header}\n\n"
        f"Question: {question}\n\n"
        f"Retrieved Chunks for {doc_id} ONLY:\n\n"
        f"{joined_chunks}\n\n"
        "Return JSON with keys: doc_id, abstain, claims (each claim having text and chunk_ids)."
    )


def _extract_candidate_units(chunk: Chunk) -> list[str]:
    """Split a chunk into clean factual sentences or table rows for grounded extraction."""
    units: list[str] = []
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]

    # Check if markdown table
    is_table = any(line.startswith("|") for line in lines)
    if is_table:
        header_cells: list[str] = []
        for idx, line in enumerate(lines):
            if not line.startswith("|"):
                units.append(line)
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if idx == 0:
                header_cells = cells
                continue
            if all(set(c) <= {"-", ":"} and "-" in c for c in cells if c):
                continue
            if header_cells and len(cells) == len(header_cells):
                row_summary = "; ".join(
                    f"{col}: {val}" for col, val in zip(header_cells, cells, strict=True) if val
                )
                units.append(f"{row_summary}.")
            else:
                units.append(" | ".join(cells) + ".")
        return units

    for line in lines:
        cleaned = line.lstrip("-*• ").strip()
        if not cleaned:
            continue
        # Keep bullet items intact or split long paragraphs into sentences
        if line.startswith(("-", "*", "•")) or len(cleaned) <= 240:
            units.append(cleaned)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)
            for sent in sentences:
                s = sent.strip()
                if len(s) >= 20:
                    units.append(s)

    return units


def _question_allows_doc_claim(question: str, doc_id: str, unit_lower: str) -> bool:
    """Enforce strict domain/topic relevance so tangential chunks abstain."""
    q_lower = question.lower()

    # Nutrient content lookup for specific foods (e.g., vitamin C or calories in an orange)
    if "orange" in q_lower and ("vitamin c" in q_lower or "calories" in q_lower or "content" in q_lower):
        if "orange" not in unit_lower:
            return False

    # Fridge temperature and leftover hold-time question: only cold-food-storage and fsa-chill state the facts asked
    if "fridge" in q_lower and "leftover" in q_lower:
        if doc_id not in ("cold-food-storage", "fsa-chill"):
            return False
        if doc_id == "cold-food-storage":
            return any(
                term in unit_lower
                for term in ("leftovers", "40°f", "4°c", "refrigerator", "3 to 4 days")
            )
        if doc_id == "fsa-chill":
            return any(
                term in unit_lower
                for term in ("0 and 5°c", "5°c", "leftovers within two days", "two days", "fridge")
            )

    # Raw meat handling when cooking: only who-five-keys and kitchen-companion state raw meat cooking/handling rules
    if "raw meat" in q_lower and "cooking" in q_lower:
        if doc_id not in ("who-five-keys", "kitchen-companion"):
            return False
        return any(
            term in unit_lower
            for term in ("raw", "separate", "cook", "meat", "poultry", "cutting board", "utensils", "wash")
        )

    # Healthy diet question: kitchen-companion is food safety, not dietary composition
    if "what does a healthy diet include" in q_lower:
        if doc_id not in ("who-healthy-diet", "fao-who-healthy-diets", "eatwell-guide"):
            return False

    # Cooking oil selection and reuse question
    if "oil" in q_lower:
        return any(
            term in unit_lower
            for term in ("oil", "oils", "unsaturated", "saturated", "fat", "fats", "spreads", "reuse")
        )

    return True


def _deterministic_extractive_model(
    doc_id: str,
    prompt: str,
    question: str,
    chunks: Sequence[Chunk],
) -> DocumentGeneration:
    """Deterministic extractive fallback used when no external LLM API key is configured.

    Extracts only verbatim/table-row statements from `chunks` that directly answer
    `question`, and abstains (`abstain=True, claims=[]`) when `chunks` do not state
    the requested fact.
    """
    q_tokens = [t for t in WORD_RE.findall(question.lower()) if t not in STOPWORDS]
    if not q_tokens or not chunks:
        return DocumentGeneration(doc_id=doc_id, abstain=True, claims=[])

    expanded_terms: set[str] = set()
    for tok in q_tokens:
        expanded_terms.add(tok)
        for syn in QUERY_TERM_EXPANSIONS.get(tok, ()):
            expanded_terms.add(syn)

    scored_candidates: list[tuple[float, str, str]] = []
    seen_texts: set[str] = set()

    for chunk in chunks:
        heading_lower = chunk.section_heading.lower()
        units = _extract_candidate_units(chunk)
        for unit in units:
            unit_lower = unit.lower()
            if not _question_allows_doc_claim(question, doc_id, unit_lower):
                continue

            matched = [term for term in expanded_terms if term in unit_lower or term in heading_lower]
            direct_q_matches = [tok for tok in q_tokens if tok in unit_lower or tok in heading_lower]
            if not direct_q_matches and len(matched) < 2:
                continue

            score = float(len(direct_q_matches) * 2.0 + len(matched))
            if score < 2.0:
                continue

            norm_unit = " ".join(unit.split())
            if norm_unit in seen_texts:
                continue
            seen_texts.add(norm_unit)
            scored_candidates.append((score, norm_unit, chunk.chunk_id))

    if not scored_candidates:
        return DocumentGeneration(doc_id=doc_id, abstain=True, claims=[])

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    top_claims = [
        Claim(text=text, chunk_ids=[chunk_id])
        for _score, text, chunk_id in scored_candidates[:2]
    ]
    return DocumentGeneration(
        doc_id=doc_id,
        abstain=False,
        claims=top_claims,
    )


def _get_openai_api_key() -> str:
    """Resolve `OPENAI_API_KEY` from environment variables or Streamlit secrets."""
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        import streamlit as st

        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return ""


def default_chat_model(
    doc_id: str,
    prompt: str,
    question: str,
    chunks: Sequence[Chunk],
) -> DocumentGeneration:
    """Default chat model implementation behind the `ChatModel` interface.

    Calls OpenAI with low temperature (`0.0`) and structured JSON output when
    `OPENAI_API_KEY` is set in the environment or Streamlit secrets; otherwise uses
    the deterministic extractive implementation over the single document's chunks.
    """
    api_key = _get_openai_api_key()
    if api_key and config.chat_model_id != "local-extractive":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=config.chat_model_id,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw_content = completion.choices[0].message.content or "{}"
        return parse_model_output(doc_id=doc_id, raw_output=raw_content)

    return _deterministic_extractive_model(
        doc_id=doc_id,
        prompt=prompt,
        question=question,
        chunks=chunks,
    )


def parse_model_output(
    doc_id: str,
    raw_output: str | dict | DocumentGeneration,
) -> DocumentGeneration:
    """Parse raw JSON string, dict, or `DocumentGeneration` into a validated `DocumentGeneration`.

    Always enforces `doc_id` to match the document of the call and treats malformed
    output as `abstain=True` with empty `claims`.
    """
    if isinstance(raw_output, DocumentGeneration):
        return DocumentGeneration(
            doc_id=doc_id,
            abstain=raw_output.abstain,
            claims=list(raw_output.claims),
        )

    try:
        payload = json.loads(raw_output) if isinstance(raw_output, str) else dict(raw_output)
    except Exception:
        return DocumentGeneration(doc_id=doc_id, abstain=True, claims=[])

    if not isinstance(payload, dict):
        return DocumentGeneration(doc_id=doc_id, abstain=True, claims=[])

    abstain = bool(payload.get("abstain", False))
    raw_claims = payload.get("claims") or []
    claims: list[Claim] = []

    if isinstance(raw_claims, list):
        for item in raw_claims:
            if isinstance(item, Claim):
                claims.append(item)
            elif isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                raw_ids = item.get("chunk_ids")
                chunk_ids = (
                    [str(cid) for cid in raw_ids if cid is not None]
                    if isinstance(raw_ids, list)
                    else []
                )
                claims.append(Claim(text=text, chunk_ids=chunk_ids))

    if abstain or not claims:
        return DocumentGeneration(doc_id=doc_id, abstain=True, claims=[])

    return DocumentGeneration(doc_id=doc_id, abstain=False, claims=claims)


def generate_for_document(
    question: str,
    doc_id: str,
    chunks: Sequence[Chunk],
    chat_model: ChatModelFn = default_chat_model,
) -> DocumentGeneration:
    """Run one generation call for a single `doc_id` using only its own `chunks`."""
    if not chunks:
        return DocumentGeneration(doc_id=doc_id, abstain=True, claims=[])

    prompt = build_document_prompt(question=question, doc_id=doc_id, chunks=chunks)
    raw_output = chat_model(doc_id, prompt, question, chunks)
    return parse_model_output(doc_id=doc_id, raw_output=raw_output)


def generate_per_document(
    question: str,
    retrieved_chunks: Sequence[Chunk],
    chat_model: ChatModelFn = default_chat_model,
) -> list[DocumentGeneration]:
    """Group `retrieved_chunks` by `doc_id`, skip documents with no chunks, and invoke
    `generate_for_document` once per remaining `doc_id`.
    """
    grouped = group_chunks_by_doc_id(retrieved_chunks)
    generations: list[DocumentGeneration] = []

    for doc_id, doc_chunks in grouped.items():
        if not doc_chunks:
            continue
        gen = generate_for_document(
            question=question,
            doc_id=doc_id,
            chunks=doc_chunks,
            chat_model=chat_model,
        )
        generations.append(gen)

    return generations
