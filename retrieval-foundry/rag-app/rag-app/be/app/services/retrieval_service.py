"""Retrieval service — fetches relevant chunks from vector store."""
import time
import structlog
from typing import List, Dict
from app.interfaces.vector_interface import VectorStoreInterface, RetrievedChunk

logger = structlog.get_logger()


async def retrieve_chunks(
    vector_store: VectorStoreInterface,
    retriever_query: str,
    file_ids: List[str],
    top_k: int = 2
) -> List[Dict]:
    """Retrieve relevant chunks from vector store for each file.

    Args:
        vector_store: The vector store to query.
        retriever_query: The refined query (keywords joined as string).
        file_ids: List of file IDs to retrieve from.
        top_k: Number of chunks per file.

    Returns:
        List of chunk dicts grouped by file:
        [{"file_id": "...", "file_name": "...", "chunks": [{"page_content": "...", "page_number": N, "chunk_id": "..."}, ...]}, ...]
    """
    st = time.time()
    logger.info(f"retrieval_service.retrieving - file_count: {len(file_ids)}, query: {retriever_query}")

    all_chunks: List[RetrievedChunk] = await vector_store.retrieve(
        query=retriever_query,
        file_ids=file_ids,
        top_k=top_k
    )

    # Group chunks by file_id
    grouped: Dict[str, Dict] = {}
    for chunk in all_chunks:
        if chunk.file_id not in grouped:
            grouped[chunk.file_id] = {
                "file_id": chunk.file_id,
                "file_name": chunk.source,
                "chunks": []
            }
        grouped[chunk.file_id]["chunks"].append({
            "page_content": chunk.page_content,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "section_path": chunk.section_path
        })

    result = list(grouped.values())
    et = time.time()
    logger.info(f"Time taken for retrieval_service.retrieve_chunks: {et - st:.2f} seconds")
    logger.info(f"retrieval_service.complete - files_retrieved: {len(result)}, total_chunks: {len(all_chunks)}")
    return result
