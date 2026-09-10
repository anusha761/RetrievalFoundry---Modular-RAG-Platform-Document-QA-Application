"""Application configuration loaded from .env and prompt modules."""
import os
from importlib import import_module
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    APP_NAME: str = "doc-rag-chat"
    API_KEY: str = "dev-api-key-12345"
    LOG_LEVEL: str = "INFO"

    # Pipeline
    BATCH_THRESHOLD: int = 2
    BATCH_SIZE: int = 2
    MAX_FILES_ALLOWED: int = 10
    CONVERSATION_HISTORY_LENGTH: int = 1
    MAX_TOKENS: int = 4096

    # LLM
    LLM_PROVIDER: str = "rag_api"
    # OPENAI_API_KEY: str = ""
    # OPENAI_MODEL: str = "gpt-4o-mini"

    # Vector Store
    VECTOR_STORE_PROVIDER: str = "rag_api"
    RAG_API_BASE_URL: str = "http://localhost:8001"

    # Cost
    INFERENCE_INPUT_COST: float = 0.000003
    INFERENCE_OUTPUT_COST: float = 0.000015

    # Domain Knowledge
    DOMAIN_KNOWLEDGE: str = "This system handles financial documents including quarterly reports, investment memoranda, and fund performance summaries. Key entities include fund names, NAV values, IRR percentages, commitment amounts, and vintage years. Documents typically follow fiscal year reporting cycles (Q1-Q4) and may reference GAAP or IFRS accounting standards."

    # MongoDB
    MONGODB_USERNAME: str = os.environ["MONGODB_USERNAME"]
    MONGODB_PASSWORD: str = os.environ["MONGODB_PASSWORD"]
    MONGODB_URI: str = "mongodb+srv://chaudhurianusha1_db_user:SYbL3fSkS0Z3HGkv@docchatdbcluster.0xeg3bw.mongodb.net"
    MONGODB_COLLECTION_CONVERSATIONS: str = "conversations"
    MONGODB_COLLECTION_RESPONSES: str = "responses"

    # Storage Provider: json | mongodb
    STORAGE_PROVIDER: str = "mongodb"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_settings() -> Settings:
    """Load settings from .env file."""
    return Settings()


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from its Python prompt module.

    Args:
        prompt_name: module name without extension
            (e.g., 'keyword_extraction_system')

    Returns:
        The prompt text constant.
    """
    prompt_module = import_module(f"app.prompts.{prompt_name}")
    try:
        return getattr(prompt_module, prompt_name.upper())
    except AttributeError as error:
        raise AttributeError(
            f"Prompt constant {prompt_name.upper()} not found in "
            f"app.prompts.{prompt_name}"
        ) from error
