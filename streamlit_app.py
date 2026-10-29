"""Streamlit deployment entry point for the Dietary Guidance RAG Chatbot.

Runs the exact Phase 8 pipeline (`handle_ask`) over the seven official public
dietary and food-safety guidance documents:
- Auto-initializes the corpus and vector index on cold start if missing on disk
  (e.g. on a fresh Streamlit Community Cloud container).
- Provides one question input, an optional document filter, and a submit action.
- On `grounded_answer`, renders each section separately in manifest order:
  document name, publisher, year, the dataset link, and claims with their
  section headings (never collapsing sections into one paragraph).
- On `not_in_corpus`, shows `message` and the list of documents in `searched`.
- On `out_of_scope`, shows `message` only (no searched/citation list).
- Stateless requests: prior turns may be displayed in the UI, but submitting a
  new question passes only `{question, document_filter}` to `handle_ask`.
"""

from html import escape

from fastapi import HTTPException
import streamlit as st

from src import config
from src.api.ask import handle_ask
from src.ingest.chunk import chunk_all_documents
from src.ingest.embed_index import VectorIndex, build_index
from src.ingest.load import load_corpus, load_manifest
from src.schemas import AskRequest, AskResponse

DOC_ACCENT: dict[str, str] = {
    "who-healthy-diet": "#1f6b4a",
    "fao-who-healthy-diets": "#3d6b2f",
    "eatwell-guide": "#c45c26",
    "cold-food-storage": "#1d4e89",
    "fsa-chill": "#0f6e78",
    "who-five-keys": "#8a3d2a",
    "kitchen-companion": "#5c3d7a",
}

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "What does a healthy diet include?",
    "How does the Eatwell Guide divide food groups?",
    "What temperature should my fridge be?",
    "Which oil should I use, and is it safe to reuse?",
)

APP_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,680&family=Source+Sans+3:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
  }
  .stApp {
    background:
      radial-gradient(900px 420px at 100% -10%, rgba(196, 92, 38, 0.16), transparent 55%),
      radial-gradient(700px 380px at -10% 0%, rgba(31, 107, 74, 0.14), transparent 50%),
      #f6f1e7;
  }
  header[data-testid="stHeader"] { background: transparent; }
  .block-container {
    padding-top: 1.4rem;
    max-width: 860px;
  }
  .sp-hero {
    background: linear-gradient(135deg, #1c1915 0%, #2c241c 55%, #3d2a1c 100%);
    color: #fff8ef;
    border-radius: 22px;
    padding: 1.35rem 1.45rem 1.2rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 16px 40px rgba(28, 25, 21, 0.18);
  }
  .sp-kicker {
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-size: 0.72rem;
    font-weight: 700;
    color: #f0b27a;
    margin: 0 0 0.35rem;
  }
  .sp-hero h1 {
    font-family: Fraunces, Georgia, serif;
    font-size: 2.35rem;
    line-height: 1.05;
    margin: 0 0 0.45rem;
    color: #fffaf3;
  }
  .sp-hero p {
    margin: 0;
    color: #efe4d4;
    font-size: 1.02rem;
    max-width: 38rem;
  }
  .sp-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.95rem;
  }
  .sp-pill {
    border: 1px solid rgba(255, 248, 239, 0.22);
    background: rgba(255, 248, 239, 0.08);
    color: #fff8ef;
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 600;
  }
  div[data-testid="stForm"] {
    background: #fffaf3;
    border: 1px solid #e7dccb;
    border-radius: 18px;
    padding: 0.85rem 1rem 0.4rem;
    box-shadow: 0 8px 24px rgba(60, 40, 20, 0.05);
  }
  .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    background: #fffdf9 !important;
    border-radius: 12px !important;
  }
  .stButton > button, .stFormSubmitButton > button {
    background: #c45c26 !important;
    color: #fffaf3 !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    padding: 0.45rem 1.1rem !important;
  }
  .stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #a3481b !important;
    color: #fffaf3 !important;
  }
  .sp-section {
    border: 1px solid #eadfce;
    border-left: 6px solid var(--accent, #c45c26);
    background: #fffaf3;
    border-radius: 16px;
    padding: 0.95rem 1rem 0.75rem;
    margin: 0.7rem 0 0.2rem;
  }
  .sp-doc-title {
    font-family: Fraunces, Georgia, serif;
    font-size: 1.15rem;
    margin: 0;
    color: #1c1915;
  }
  .sp-meta {
    color: #6b6258;
    font-size: 0.92rem;
    margin: 0.15rem 0 0.35rem;
  }
  .sp-link a { color: #9a3412; font-weight: 600; word-break: break-all; }
  .sp-claim {
    margin: 0.45rem 0 0;
    color: #1c1915;
  }
  .sp-heading {
    color: #6b6258;
    font-style: italic;
  }
  .sp-banner {
    border-radius: 16px;
    padding: 0.95rem 1rem;
    margin-top: 0.7rem;
  }
  .sp-corpus {
    background: #fff6df;
    border: 1px solid #e8c56a;
    color: #3f3424;
  }
  .sp-scope {
    background: #fde8e4;
    border: 1px solid #e7a598;
    color: #4a241d;
  }
  .sp-q {
    font-family: Fraunces, Georgia, serif;
    font-size: 1.25rem;
    margin: 0 0 0.15rem;
    color: #1c1915;
  }
  .sp-chip {
    display: inline-block;
    background: #f3e6d4;
    color: #5c4632;
    border-radius: 999px;
    padding: 0.12rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
  }
  .sp-empty {
    margin-top: 1.1rem;
    color: #6b6258;
    font-size: 0.98rem;
  }
</style>
"""


@st.cache_resource(show_spinner="Warming the guidance counter...")
def ensure_corpus_and_index(cache_version: str = "official-urls-v2") -> VectorIndex:
    """Ensure `blocks.jsonl`, parsed chunks, and `vectors.sqlite3` exist on disk."""
    _ = cache_version
    index = VectorIndex(index_dir=config.INDEX_DIR)
    if index.is_ready():
        return index

    manifest_entries = load_manifest(config.MANIFEST_PATH)
    missing_chunks = any(
        not (config.PARSED_DIR / f"{entry['doc_id']}.chunks.jsonl").exists()
        for entry in manifest_entries
    )
    if missing_chunks:
        download_needed = not config.BLOCKS_PATH.exists()
        load_corpus(download=download_needed)
        chunk_all_documents()

    build_index(index_dir=config.INDEX_DIR)
    return index


def _accent_for(doc_id: str | None) -> str:
    if doc_id and doc_id in DOC_ACCENT:
        return DOC_ACCENT[doc_id]
    return "#c45c26"


def render_response_card(turn: dict) -> None:
    """Render one stateless Q&A turn according to Phase 9 UI rules."""
    question: str = turn["question"]
    doc_filter: str | None = turn.get("document_filter")
    response: AskResponse | None = turn.get("response")
    error_detail: str | None = turn.get("error")

    filter_chip = (
        f'<div class="sp-chip">Filtered to {escape(doc_filter)}</div>' if doc_filter else ""
    )
    parts = [
        '<div class="sp-turn">',
        filter_chip,
        f'<p class="sp-q">{escape(question)}</p>',
    ]

    if error_detail:
        parts.append(f'<div class="sp-banner sp-scope">{escape(error_detail)}</div></div>')
        st.markdown("".join(parts), unsafe_allow_html=True)
        return

    if response is None:
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)
        return

    if response.outcome == "grounded_answer":
        parts.append('<div class="sp-chip">Cited answer</div>')
        for sec in response.sections:
            accent = _accent_for(sec.doc_id)
            claims = []
            for claim in sec.claims:
                heading = (
                    f' <span class="sp-heading">({escape(claim.section_heading)})</span>'
                    if claim.section_heading
                    else ""
                )
                claims.append(f'<p class="sp-claim">{escape(claim.text)}{heading}</p>')
            parts.append(
                f'<article class="sp-section" style="--accent:{accent}">'
                f'<p class="sp-doc-title">{escape(sec.document_name)}</p>'
                f'<p class="sp-meta">{escape(sec.publisher)}, {sec.year}</p>'
                f'<p class="sp-link"><a href="{escape(sec.source_url)}" target="_blank" rel="noopener noreferrer">Official source</a></p>'
                f'<p class="sp-meta">{escape(sec.source_url)}</p>'
                f'{"".join(claims)}'
                "</article>"
            )
    elif response.outcome == "not_in_corpus":
        message = response.message or "The official dietary guidance corpus does not cover this question."
        searched_html = ""
        if response.searched:
            items = "".join(
                f"<li><strong>{escape(doc.document_name)}</strong> ({escape(doc.publisher)})</li>"
                for doc in response.searched
            )
            searched_html = f"<p><strong>Documents searched</strong></p><ul>{items}</ul>"
        parts.append(
            f'<div class="sp-banner sp-corpus"><div class="sp-chip">Not in these documents</div>'
            f"<p>{escape(message)}</p>{searched_html}</div>"
        )
    elif response.outcome == "out_of_scope":
        message = response.message or "This request is out of scope. Please consult a qualified professional."
        parts.append(
            f'<div class="sp-banner sp-scope"><div class="sp-chip">Outside this counter</div>'
            f"<p>{escape(message)}</p></div>"
        )

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _ask(index: VectorIndex, question: str, document_filter: str | None) -> None:
    """Run one independent request and store the rendered turn."""
    req = AskRequest(question=question, document_filter=document_filter)
    try:
        resp = handle_ask(req, index=index)
        turn = {
            "question": question,
            "document_filter": document_filter,
            "response": resp,
            "error": None,
        }
    except HTTPException as exc:
        turn = {
            "question": question,
            "document_filter": document_filter,
            "response": None,
            "error": f"HTTP {exc.status_code}: {exc.detail}",
        }
    st.session_state["turns"].insert(0, turn)


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Diet Factory — Cited dietary guidance",
        page_icon="🍽️",
        layout="centered",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <section class="sp-hero">
          <p class="sp-kicker">Official guidance, on the counter</p>
          <h1>Diet Factory</h1>
          <p>Ask how to eat and how to keep food safe. Answers come only from seven public guidance documents, and every claim keeps its own source.</p>
          <div class="sp-pills">
            <span class="sp-pill">Cited sections</span>
            <span class="sp-pill">No blended guidelines</span>
            <span class="sp-pill">Medical &amp; weight questions declined</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    index = ensure_corpus_and_index()
    manifest_entries = load_manifest(config.MANIFEST_PATH)
    filter_options: list[tuple[str, str | None]] = [
        ("Search all seven documents", None),
        *[
            (f"{entry['document_name']} — {entry['publisher']}", entry["doc_id"])
            for entry in manifest_entries
        ],
    ]

    if "turns" not in st.session_state:
        st.session_state["turns"] = []
    manifest_urls = {entry["doc_id"]: entry["source_url"] for entry in manifest_entries}
    for existing_turn in st.session_state["turns"]:
        existing_resp = existing_turn.get("response")
        if existing_resp and getattr(existing_resp, "sections", None):
            for sec in existing_resp.sections:
                if sec.doc_id in manifest_urls:
                    sec.source_url = manifest_urls[sec.doc_id]

    st.caption("Try a ready-made question, or write your own.")
    chip_cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUESTIONS):
        if chip_cols[idx % 2].button(example, key=f"example-{idx}", use_container_width=True):
            _ask(index, example, None)
            st.rerun()

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "What do you want to check?",
            placeholder="What does a healthy diet include?",
            height=90,
        )
        selected_label = st.selectbox(
            "Look in one document, or all of them",
            options=[label for label, _val in filter_options],
            index=0,
        )
        submitted = st.form_submit_button("Ask Diet Factory", type="primary")

    if submitted and question.strip():
        selected_filter = dict(filter_options).get(selected_label)
        _ask(index, question.strip(), selected_filter)

    if not st.session_state["turns"]:
        st.markdown(
            '<p class="sp-empty">The counter is clear. Ask about food groups, fridge temperature, leftovers, or oils — or try one of the prompts above.</p>',
            unsafe_allow_html=True,
        )

    for turn in st.session_state["turns"]:
        render_response_card(turn)


if __name__ == "__main__":
    main()
