
"""
FASTAPI RETRIEVAL API

Purpose
-------
Expose the complete retrieval pipeline through a FastAPI HTTP API.

Request flow
------------

FastAPI application startup
    |
    +--> Initialize Qdrant client
    +--> Load dense embedding model
    +--> Load sparse BM25 model
    +--> Load cross-encoder reranker
    |
    v
HTTP request
    |
    v
FastAPI endpoint
    |
    v
retrieval_service.retrieve_similar_content()
    |
    +--> Qdrant hybrid search + RRF
    |
    +--> Cross-encoder reranking
    |
    +--> Table resolution
    |
    v
JSON response


IMPORTANT
---------
The expensive ML/runtime resources are initialized ONCE during
FastAPI application startup through the lifespan handler.

Therefore:

    - Qdrant client is initialized once.
    - Dense embedding model is loaded once.
    - Sparse BM25 model is loaded once.
    - Cross-encoder reranker is loaded once.

They are NOT loaded for every API request.

If model initialization fails, FastAPI startup fails instead of
allowing the application to start in a broken state.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine.retrieval.retriever import initialize_retriever
from rag_engine.retrieval.reranker import initialize_reranker
from rag_engine.retrieval.retrieval_service import retrieve_similar_content

from rag_engine.generation.chat_service import generate_chat_response
from rag_engine.generation.chat_service import initialize_chat_service


# ==========================================================
# APPLICATION LIFESPAN
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all expensive retrieval resources once when
    the FastAPI application starts.

    Startup sequence:

        1. Qdrant client
        2. Dense embedding model
        3. Sparse BM25 model
        4. Cross-encoder reranker

    If any initialization step fails, the exception is allowed
    to propagate so that the application does not start in an
    unusable state.
    """

    print()
    print("=" * 70)
    print("STARTING RAG RETRIEVAL API")
    print("=" * 70)

    # ------------------------------------------------------
    # Initialize Qdrant + embedding models
    # ------------------------------------------------------

    print()
    print("Initializing retriever...")

    initialize_retriever()

    # ------------------------------------------------------
    # Initialize cross-encoder reranker
    # ------------------------------------------------------

    print()
    print("Initializing reranker...")

    initialize_reranker()

    print() 
    print("Initializing chat service...") 
    initialize_chat_service()

    # ------------------------------------------------------
    # Startup completed
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("RAG RETRIEVAL API STARTUP COMPLETED")
    print("=" * 70)
    print()

    try:

        yield

    finally:

        # --------------------------------------------------
        # Application shutdown
        # --------------------------------------------------

        print()
        print("=" * 70)
        print("SHUTTING DOWN RAG RETRIEVAL API")
        print("=" * 70)
        print()


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="RAG Retrieval API",
    description=(
        "Hybrid retrieval API using Qdrant, BM25, "
        "RRF, cross-encoder reranking, and table resolution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================================
# REQUEST MODEL
# ==========================================================

class RetrievalRequest(BaseModel):
    """
    Request body for the retrieval endpoint.
    """

    user_query: str = Field(
        ...,
        min_length=1,
        description="Natural-language user query.",
    )

    file_id: str = Field(
        ...,
        min_length=1,
        description="Document ID to search.",
    )

    candidate_k: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of candidates retrieved by hybrid "
            "search + RRF."
        ),
    )

    desired_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of final chunks returned after "
            "cross-encoder reranking."
        ),
    )


# ==========================================================
# RESPONSE MODEL
# ==========================================================

class RetrievalResponse(BaseModel):
    """
    Response returned by the retrieval endpoint.
    """

    query: str
    file_id: str
    candidate_k: int
    desired_k: int
    results: List[Dict[str, Any]]


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Basic API health check.

    Because FastAPI only starts after the lifespan startup
    completes successfully, reaching this endpoint means
    the application startup initialization succeeded.
    """

    return {
        "status": "healthy"
    }


# ==========================================================
# RETRIEVAL ENDPOINT
# ==========================================================

@app.post(
    "/retrieve",
    response_model=RetrievalResponse,
)
def retrieve(
    request: RetrievalRequest,
) -> RetrievalResponse:
    """
    Execute the complete retrieval pipeline.

    Pipeline:

        Qdrant Hybrid Search
                ↓
              RRF
                ↓
        Cross-Encoder Reranking
                ↓
          Table Resolution
                ↓
          JSON Response
    """

    # ------------------------------------------------------
    # Validate candidate_k / desired_k relationship
    # ------------------------------------------------------

    if request.desired_k > request.candidate_k:

        raise HTTPException(
            status_code=400,
            detail=(
                "desired_k cannot be greater than candidate_k."
            ),
        )

    # ------------------------------------------------------
    # Execute retrieval service
    # ------------------------------------------------------

    try:

        results = retrieve_similar_content(
            user_query=request.user_query,
            file_id=request.file_id,
            candidate_k=request.candidate_k,
            desired_k=request.desired_k,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except (TypeError, RuntimeError, FileNotFoundError) as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected error while executing "
                f"retrieval pipeline: {exc}"
            ),
        ) from exc

    # ------------------------------------------------------
    # Return JSON response
    # ------------------------------------------------------

    return RetrievalResponse(
        query=request.user_query,
        file_id=request.file_id,
        candidate_k=request.candidate_k,
        desired_k=request.desired_k,
        results=results,
    )




# ==========================================================
# CHAT REQUEST MODEL
# ==========================================================

class ChatRequest(BaseModel):
    """
    Request body for the chat endpoint.
    """

    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System instruction for GPT-4o-mini.",
    )

    user_prompt: str = Field(
        ...,
        min_length=1,
        description="User prompt sent to GPT-4o-mini.",
    )


# ==========================================================
# CHAT RESPONSE MODEL
# ==========================================================

class ChatResponse(BaseModel):
    """
    Response returned by the chat endpoint.
    """

    response: str


# ==========================================================
# CHAT ENDPOINT
# ==========================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Send system and user prompts to GPT-4o-mini.
    """

    try:

        response = generate_chat_response(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except (TypeError, RuntimeError) as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected error while calling "
                f"OpenAI: {exc}"
            ),
        ) from exc

    return ChatResponse(
        response=response,
    )




# ==========================================================
# LOCAL DEVELOPMENT ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

