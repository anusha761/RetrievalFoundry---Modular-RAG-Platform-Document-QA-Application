"""Real vector store adapter that calls the RAG API's /retrieve endpoint."""
import time
import httpx
import structlog
from typing import List
from app.interfaces.vector_interface import VectorStoreInterface, RetrievedChunk
from app.config import load_settings

logger = structlog.get_logger()


class RagApiVectorStore(VectorStoreInterface):
    """Calls the external RAG API /retrieve endpoint for chunk retrieval.

    The /retrieve endpoint accepts a single file_id per call.
    This adapter loops through all requested file_ids, calling once per file
    
    """

    def __init__(self, base_url: str = None):
        settings = load_settings()
        self.base_url = base_url or settings.RAG_API_BASE_URL

    async def retrieve(
        self,
        query: str,
        file_ids: List[str],
        top_k: int = 2
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks by calling /retrieve once per file.

        Args:
            query: The search query (keywords or text).
            file_ids: List of document IDs to search.
            top_k: Number of final chunks per file (desired_k).

        Returns:
            List of retrieved chunks from all files combined.
        """
        url = f"{self.base_url}/retrieve"
        all_chunks: List[RetrievedChunk] = []

        for file_id in file_ids:
            payload = {
                "user_query": query,
                "file_id": file_id,
                "candidate_k": top_k * 2,  # Retrieve more candidates for reranking
                "desired_k": top_k
            }

            st = time.time()
            logger.info(f"rag_api_vector_store.retrieving - file_id: {file_id}")
            logger.info(f"rag_api_vector_store.outbound_payload - url={url}, payload={payload}")

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload)
                    status_code = response.status_code
                    response_body = response.text
                    logger.info(
                        f"rag_api_vector_store.response_status - file_id={file_id}, "
                        f"status_code={status_code}, response_length={len(response_body)}"
                    )
                    response.raise_for_status()
                    data = response.json()

                et = time.time()
                logger.info(f"Time taken for /retrieve call (file: {file_id}): {et - st:.2f} seconds")
                logger.info(f"rag_api_vector_store.response_summary - file_id={file_id}, result_count={len(data.get('results', []))}, response_length={len(response_body)}")

                results = data.get("results", [])

                for result in results:
                    raw_section_path = result.get("section_path", [])
                    if isinstance(raw_section_path, str):
                        section_path = [part.strip() for part in raw_section_path.split(">") if part.strip()]
                    elif isinstance(raw_section_path, list):
                        section_path = [str(part).strip() for part in raw_section_path if str(part).strip()]
                    else:
                        section_path = []

                    chunk = RetrievedChunk(
                        page_content=result.get("resolved_chunk_text", result.get("chunk_text", "")),
                        source=result.get("document_name", ""),
                        file_id=result.get("document_id", file_id),
                        page_number=result.get("page_no", 0),
                        chunk_id=result.get("chunk_id", ""),
                        section_path=section_path
                    )
                    all_chunks.append(chunk)

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                body = e.response.text if e.response is not None else ""
                logger.error(
                    f"rag_api_vector_store.http_error - file_id: {file_id}, "
                    f"status: {status_code}, response_length={len(body)}, response_body={body[:1000]}"
                )
                # Continue with other files even if one fails
                continue
            except httpx.ConnectError as e:
                logger.error(f"rag_api_vector_store.connection_error - {str(e)}")
                raise RuntimeError(f"Cannot connect to RAG API at {url}. Is it running?") from e
            except Exception as e:
                logger.error(f"rag_api_vector_store.unexpected_error - file_id: {file_id}, error: {str(e)}")
                continue

        logger.info(f"rag_api_vector_store.complete - total chunks retrieved: {len(all_chunks)}")
        return all_chunks
