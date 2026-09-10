"""Dependency injection — provides LLM and VectorStore instances to routes."""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import load_settings
from app.interfaces.llm_interface import LLMInterface
from app.interfaces.vector_interface import VectorStoreInterface


# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from request header."""
    settings = load_settings()
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


def get_llm() -> LLMInterface:
    """Get the configured LLM service instance."""
    settings = load_settings()
    
    if settings.LLM_PROVIDER == "rag_api":
        from app.adapters.rag_api_llm import RagApiLLM
        return RagApiLLM(base_url=settings.RAG_API_BASE_URL)
    return None  # Default to None if no valid provider is configured


def get_vector_store() -> VectorStoreInterface:
    """Get the configured vector store service instance."""
    settings = load_settings()
    
    if settings.VECTOR_STORE_PROVIDER == "rag_api":
        from app.adapters.rag_api_vector_store import RagApiVectorStore
        return RagApiVectorStore(base_url=settings.RAG_API_BASE_URL)
    return None  # Default to None if no valid provider is configured
