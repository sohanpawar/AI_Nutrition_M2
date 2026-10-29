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

from fastapi import HTTPException
import streamlit as st

from src import config
from src.api.ask import handle_ask
from src.ingest.chunk import chunk_all_documents
from src.ingest.embed_index import VectorIndex, build_index
from src.ingest.load import load_corpus, load_manifest
from src.schemas import AskRequest, AskResponse


@st.cache_resource(show_spinner="Preparing official dietary guidance index...")
def ensure_corpus_and_index() -> VectorIndex:
    """Ensure `blocks.jsonl`, parsed chunks, and `vectors.sqlite3` exist on disk."""
    index = VectorIndex(index_dir=config.INDEX_DIR)
    if index.is_ready():
        return index

    # If deployed on a fresh cloud instance without pre-built artifacts, run Phases 2 -> 3 -> 4
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


def render_response_card(turn: dict) -> None:
    """Render one stateless Q&A turn according to Phase 9 UI rules."""
    question: str = turn["question"]
    doc_filter: str | None = turn.get("document_filter")
    response: AskResponse | None = turn.get("response")
    error_detail: str | None = turn.get("error")

    with st.container(border=True):
        if doc_filter:
            st.markdown(f"**Question:** {question}  \n`Filter: {doc_filter}`")
        else:
            st.markdown(f"**Question:** {question}")

        if error_detail:
            st.error(error_detail)
            return

        if response is None:
            return

        if response.outcome == "grounded_answer":
            # Render each section in order without collapsing sections into one paragraph
            for sec in response.sections:
                with st.container(border=True):
                    st.markdown(
                        f"**{sec.document_name} — {sec.publisher}, {sec.year}**"
                    )
                    st.markdown(f"[{sec.source_url}]({sec.source_url})")
                    claim_lines = [
                        f"- {claim.text} *({claim.section_heading})*"
                        if claim.section_heading
                        else f"- {claim.text}"
                        for claim in sec.claims
                    ]
                    st.markdown("\n".join(claim_lines))

        elif response.outcome == "not_in_corpus":
            # Show message and the list in searched
            st.warning(
                response.message
                or "The official dietary guidance corpus does not cover this question."
            )
            if response.searched:
                searched_items = "\n".join(
                    f"- **{doc.document_name}** ({doc.publisher})"
                    for doc in response.searched
                )
                st.markdown(f"**Documents searched:**\n{searched_items}")

        elif response.outcome == "out_of_scope":
            # Show message only; never show an empty citation list that looks like a search happened
            st.error(
                response.message
                or "This request is out of scope. Please consult a qualified professional."
            )


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Dietary Guidance RAG Chatbot",
        page_icon="🥗",
        layout="centered",
    )

    st.title("Dietary Guidance RAG Chatbot")
    st.caption(
        "Answers food, nutrition, and food-safety questions exclusively from seven "
        "official public guidance documents. Every claim is cited."
    )

    index = ensure_corpus_and_index()

    manifest_entries = load_manifest(config.MANIFEST_PATH)
    filter_options: list[tuple[str, str | None]] = [
        ("All corpus documents (no filter)", None),
        *[
            (f"{entry['document_name']} — {entry['publisher']} ({entry['doc_id']})", entry["doc_id"])
            for entry in manifest_entries
        ],
    ]

    if "turns" not in st.session_state:
        st.session_state["turns"] = []

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "Your Question",
            placeholder="e.g., What does a healthy diet include? / What temperature should my fridge be?",
            height=90,
        )
        selected_label = st.selectbox(
            "Optional Document Filter",
            options=[label for label, _val in filter_options],
            index=0,
        )
        submitted = st.form_submit_button("Ask Guidance", type="primary")

    if submitted and question.strip():
        clean_q = question.strip()
        selected_filter = dict(filter_options).get(selected_label)

        # Stateless request: only the new question and optional filter are passed to handle_ask
        req = AskRequest(question=clean_q, document_filter=selected_filter)
        try:
            resp = handle_ask(req, index=index)
            st.session_state["turns"].insert(
                0,
                {
                    "question": clean_q,
                    "document_filter": selected_filter,
                    "response": resp,
                    "error": None,
                },
            )
        except HTTPException as exc:
            st.session_state["turns"].insert(
                0,
                {
                    "question": clean_q,
                    "document_filter": selected_filter,
                    "response": None,
                    "error": f"HTTP {exc.status_code}: {exc.detail}",
                },
            )

    for turn in st.session_state["turns"]:
        render_response_card(turn)


if __name__ == "__main__":
    main()
