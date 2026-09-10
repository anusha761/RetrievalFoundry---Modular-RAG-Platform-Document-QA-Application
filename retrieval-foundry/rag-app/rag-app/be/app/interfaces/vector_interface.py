"""Abstract interface for vector store operations."""
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single retrieved chunk from the vector store."""
    page_content: str
    source: str  # filename
    file_id: str
    page_number: int
    chunk_id: str  # unique chunk identifier for citation tracking
    section_path: List[str] = Field(default_factory=list)  # hierarchical section path from document structure


class VectorStoreInterface(ABC):
    """Abstract base class for vector store providers."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        file_ids: List[str],
        top_k: int = 2
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks from the vector store.

        Args:
            query: The search query (keywords or text).
            file_ids: Filter results to only these document IDs.
            top_k: Number of chunks to retrieve per file.

        Returns:
            List of retrieved chunks with metadata.
        """
        ...
