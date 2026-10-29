"""POST /ask endpoint (Phase 8).

Wires the request pipeline in the exact order specified by the architecture:
1. Policy gate. On a match, return `out_of_scope`, empty `sections`, empty `searched`,
   and the fixed message without embedding or reading the index.
2. Resolve `document_filter` and any alias in the question. An unknown explicit
   filter returns HTTP 400 and does not search.
3. Retrieve. If the index is missing or embedding fails, return HTTP 503 (never call
   the generator with an empty context caused by an error).
4. If no chunk remains above `score_floor`, return `not_in_corpus` and name the
   search record.
5. Generate once per document that still has chunks.
6. Run the citation check.
7. If no section remains, return `not_in_corpus`.
8. Otherwise assemble and return `grounded_answer`.

Each request is independent and stateless.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src import config
from src.answer.assemble import build_response
from src.answer.citations import filter_citations
from src.answer.generate import (
    ChatModelFn,
    default_chat_model,
    generate_per_document,
)
from src.ingest.embed_index import IndexError, VectorIndex
from src.policy import gate
from src.retrieval.resolve import InvalidDocumentFilterError, resolve_document
from src.retrieval.search import retrieve
from src.schemas import AskRequest, AskResponse

app = FastAPI(
    title="Dietary Guidance RAG Chatbot",
    description="Grounded dietary and food-safety guidance assistant over official public health documents.",
)

STATIC_INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def serve_chat_client() -> HTMLResponse:
    """Serve the single-page chat client."""
    return HTMLResponse(content=STATIC_INDEX_PATH.read_text(encoding="utf-8"))


def handle_ask(
    request: AskRequest,
    index: VectorIndex | None = None,
    chat_model: ChatModelFn = default_chat_model,
    manifest_path: Path = config.MANIFEST_PATH,
) -> AskResponse:
    """Execute the 8-stage `/ask` pipeline for a single independent request."""
    question = request.question

    # 1. Policy gate
    if gate.check(question) == "out_of_scope":
        return gate.make_out_of_scope_response()

    # 2. Resolve document_filter and any alias in the question
    try:
        resolve_document(
            question=question,
            document_filter=request.document_filter,
            manifest_path=manifest_path,
        )
    except InvalidDocumentFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 3. Retrieve (HTTP 503 if index is missing or embedding fails)
    try:
        retrieval_result = retrieve(
            question=question,
            document_filter=request.document_filter,
            index=index,
            manifest_path=manifest_path,
        )
    except InvalidDocumentFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval or embedding failure: {exc}",
        ) from exc

    # 4. If no chunk remains above score_floor, return not_in_corpus with the search record
    if not retrieval_result.chunks:
        return build_response(
            sections=[],
            searched=retrieval_result.searched,
            manifest_path=manifest_path,
        )

    # 5. Generate once per document that still has chunks
    generations = generate_per_document(
        question=question,
        retrieved_chunks=retrieval_result.chunks,
        chat_model=chat_model,
    )

    # 6. Run the citation check
    surviving_sections = filter_citations(
        generations=generations,
        retrieved_chunks=retrieval_result.chunks,
    )

    # 7 & 8. If no section remains -> not_in_corpus; otherwise assemble -> grounded_answer
    return build_response(
        sections=surviving_sections,
        searched=retrieval_result.searched,
        manifest_path=manifest_path,
    )


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """FastAPI route for `POST /ask`."""
    return handle_ask(request)
