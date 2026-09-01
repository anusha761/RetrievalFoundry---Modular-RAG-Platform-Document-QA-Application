"""
RETRIEVAL SERVICE

Purpose
-------
Provide the complete retrieval pipeline by orchestrating:

    Qdrant Hybrid Search + RRF
                |
                v
        Cross-Encoder Reranking
                |
                v
        Final ranked chunks
                |
                v
        Table Resolution
                |
                v
        Final retrieval context

This module is the service layer.

It intentionally does NOT perform:

    - FastAPI handling
    - LLM calls
    - Prompt construction
    - Answer generation

Table file resolution is delegated to table_resolver.py.

Important
---------
The original retrieved chunk is ALWAYS preserved in:

    chunk_text

If a chunk contains table references, the table-resolved version is
stored separately in:

    resolved_chunk_text

Therefore:

    chunk_text
        = original retrieved chunk

    resolved_chunk_text
        = original chunk + resolved original table(s)

This allows downstream components to choose whether they need the
original chunk or the expanded retrieval context.

Inputs
------
user_query
file_id
candidate_k
desired_k

Example
-------
candidate_k = 10
desired_k = 5

Pipeline:

    Dense search  -> top 10
    Sparse search -> top 10
             |
             v
            RRF
             |
             v
       top 10 candidates
             |
             v
       Cross-encoder
             |
             v
        top 5 chunks
             |
             v
      Table resolution
             |
             v
      Final retrieval chunks
"""

from typing import Any, Dict, List

from .retriever import hybrid_search_rrf
from .reranker import rerank_chunks
from .table_resolver import resolve_tables_for_chunks


# ==========================================================
# RETRIEVAL SERVICE FUNCTION
# ==========================================================

def retrieve_similar_content(
    user_query: str,
    file_id: str,
    candidate_k: int,
    desired_k: int,
) -> List[Dict[str, Any]]:
    """
    Execute the complete retrieval pipeline.

    Parameters
    ----------
    user_query : str
        Natural-language question from the user.

    file_id : str
        document_id of the document that should be searched.

    candidate_k : int
        Number of candidates retained after hybrid search + RRF.

        Example:
            candidate_k = 10

            Dense  -> top 10
            Sparse -> top 10
            RRF    -> top 10

    desired_k : int
        Number of final chunks returned after cross-encoder
        reranking.

        Example:
            desired_k = 5

            RRF candidates -> 10
            Reranker        -> top 5

    Returns
    -------
    List[Dict[str, Any]]
        Final reranked and table-resolved chunks.

        Each dictionary contains the original chunk metadata plus:

            "reranker_score"
            "rrf_score"
            "resolved_chunk_text"

        Important fields:

            document_name
            document_id
            chunk_id
            page_no
            page_end
            section_path
            table_ref
            chunk_text
            resolved_chunk_text
            retrieval_text
            reranker_score
            rrf_score

        The original "chunk_text" is preserved unchanged.

        "resolved_chunk_text" contains the original chunk plus
        resolved table content when table_ref is present.
    """

    # ------------------------------------------------------
    # Validate user_query
    # ------------------------------------------------------

    if not isinstance(user_query, str):
        raise TypeError(
            "user_query must be a string."
        )

    user_query = user_query.strip()

    if not user_query:
        raise ValueError(
            "user_query cannot be empty."
        )

    # ------------------------------------------------------
    # Validate file_id
    # ------------------------------------------------------

    if not isinstance(file_id, str):
        raise TypeError(
            "file_id must be a string."
        )

    file_id = file_id.strip()

    if not file_id:
        raise ValueError(
            "file_id cannot be empty."
        )

    # ------------------------------------------------------
    # Validate candidate_k
    # ------------------------------------------------------

    if not isinstance(candidate_k, int):
        raise TypeError(
            "candidate_k must be an integer."
        )

    if candidate_k <= 0:
        raise ValueError(
            "candidate_k must be greater than 0."
        )

    # ------------------------------------------------------
    # Validate desired_k
    # ------------------------------------------------------

    if not isinstance(desired_k, int):
        raise TypeError(
            "desired_k must be an integer."
        )

    if desired_k <= 0:
        raise ValueError(
            "desired_k must be greater than 0."
        )

    if desired_k > candidate_k:
        raise ValueError(
            "desired_k cannot be greater than candidate_k."
        )

    # ======================================================
    # STAGE 1 - HYBRID SEARCH + RRF
    # ======================================================

    candidate_points = hybrid_search_rrf(
        user_query=user_query,
        file_id=file_id,
        candidate_k=candidate_k,
    )

    # ------------------------------------------------------
    # No candidates found
    # ------------------------------------------------------

    if not candidate_points:
        return []

    # ======================================================
    # CONVERT QDRANT RESULTS TO SERVICE-LAYER DICTIONARIES
    # ======================================================

    candidate_chunks: List[Dict[str, Any]] = []

    for point in candidate_points:

        payload = point.payload or {}

        chunk = dict(payload)

        # Preserve the RRF score for diagnostics/debugging.
        #
        # This is useful internally, but the final answer
        # should primarily rely on reranker_score.
        chunk["rrf_score"] = float(point.score)

        candidate_chunks.append(chunk)

    # ======================================================
    # STAGE 2 - CROSS-ENCODER RERANKING
    # ======================================================

    final_chunks = rerank_chunks(
        user_query=user_query,
        candidate_chunks=candidate_chunks,
        desired_k=desired_k,
    )

        # ======================================================
    # STAGE 3 - TABLE RESOLUTION
    # ======================================================
    #
    # table_resolver.py preserves the original chunk in:
    #
    #     original_chunk_text
    #
    # and temporarily places the table-resolved content in:
    #
    #     chunk_text
    #
    # We convert that resolver output into the service-layer
    # contract:
    #
    #     chunk_text
    #         -> original retrieved chunk
    #
    #     resolved_chunk_text
    #         -> original chunk + resolved table(s)
    #
    # ======================================================

    resolved_chunks = resolve_tables_for_chunks(
        final_chunks
    )

    for chunk in resolved_chunks:

        original_text = chunk.get(
            "original_chunk_text",
            chunk.get("chunk_text", ""),
        )

        resolved_text = chunk.get(
            "chunk_text",
            original_text,
        )

        # Store the table-expanded version separately.
        chunk["resolved_chunk_text"] = resolved_text

        # Restore the original retrieved chunk.
        chunk["chunk_text"] = original_text

    return resolved_chunks

    


# ==========================================================
# TEST / MAIN
# ==========================================================

def main() -> None:
    """
    Standalone test for the complete retrieval service.

    This test verifies:

        Qdrant hybrid search
                +
        RRF
                +
        Cross-encoder reranking
                +
        Table resolution
                +
        Preservation of original chunk text
    """

    # ------------------------------------------------------
    # Test input
    # ------------------------------------------------------

    # Test - table resolution
    user_query = (
        "What is the original Table 1 containing Apple's "
        "revenue, cost of revenue, gross profit, operating expenses, "
        "operating income, and net income?"
    )

    file_id = "AAPL_10-K_Sample"

    candidate_k = 10
    desired_k = 5

    # ======================================================
    # DISPLAY INPUT
    # ======================================================

    print()
    print("=" * 70)
    print("RETRIEVAL SERVICE TEST")
    print("=" * 70)

    print(f"Query       : {user_query}")
    print(f"File ID     : {file_id}")
    print(f"Candidate K : {candidate_k}")
    print(f"Desired K   : {desired_k}")

    # ======================================================
    # RUN COMPLETE RETRIEVAL PIPELINE
    # ======================================================

    print()
    print(
        "Running hybrid search + RRF + reranking + "
        "table resolution..."
    )

    results = retrieve_similar_content(
        user_query=user_query,
        file_id=file_id,
        candidate_k=candidate_k,
        desired_k=desired_k,
    )

    # ======================================================
    # DISPLAY RESULTS
    # ======================================================

    print()
    print("=" * 70)
    print(f"FINAL RESULTS: {len(results)} chunks")
    print("=" * 70)

    if not results:

        print()
        print(
            "No matching chunks found for "
            f"file_id='{file_id}'."
        )

        print()
        print("=" * 70)
        print("RETRIEVAL SERVICE TEST COMPLETED")
        print("=" * 70)

        return

    # ------------------------------------------------------
    # Display final chunks
    # ------------------------------------------------------

    for rank, chunk in enumerate(
        results,
        start=1,
    ):

        print()
        print("-" * 70)

        print(
            f"Final Rank      : {rank}"
        )

        print(
            f"Reranker Score  : "
            f"{chunk.get('reranker_score', 0.0):.6f}"
        )

        print(
            f"RRF Score       : "
            f"{chunk.get('rrf_score', 0.0):.6f}"
        )

        print(
            f"Document ID     : "
            f"{chunk.get('document_id')}"
        )

        print(
            f"Document Name   : "
            f"{chunk.get('document_name')}"
        )

        print(
            f"Chunk ID        : "
            f"{chunk.get('chunk_id')}"
        )

        print(
            f"Page            : "
            f"{chunk.get('page_no')}"
        )

        print(
            f"Section Path    : "
            f"{chunk.get('section_path')}"
        )

        print(
            f"Table Ref       : "
            f"{chunk.get('table_ref')}"
        )

        print()
        print("Original Chunk Text:")
        print(
            chunk.get(
                "chunk_text",
                "",
            )
        )

        print()
        print("Resolved Chunk Text:")
        print(
            chunk.get(
                "resolved_chunk_text",
                "",
            )
        )

    print()
    print("=" * 70)
    print("RETRIEVAL SERVICE TEST COMPLETED")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()

