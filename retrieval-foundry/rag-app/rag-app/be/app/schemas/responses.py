"""Response schemas for API endpoints."""
from typing import Any, List, Optional
from pydantic import BaseModel


class ResponseSchema(BaseModel):
    """Standard API response wrapper — mirrors original project's ResponseSchema."""
    status: bool = True
    item: Any = None
    message: str = ""


class Citation(BaseModel):
    """A single citation reference."""
    file_name: str
    file_id: str
    page_number: int
    excerpt: str
    section_path: List[str] = []


class TokenCost(BaseModel):
    """Token usage and cost tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0


class ChatResponseItem(BaseModel):
    """The item payload for a successful chat response."""
    question_id: str
    conversation_id: str
    summary: str
    citations: List[Citation] = []
    token_cost: TokenCost = TokenCost()
    file_ids: List[str] = []


class RecordPromptResponseItem(BaseModel):
    """Response from record-prompt step."""
    question_id: str
    conversation_id: str





class RetrieveResponseItem(BaseModel):
    """Response from retrieve step - chunks grouped by file."""
    retrieved_chunks: List[dict]


class ConversationItem(BaseModel):
    """A conversation record."""
    conversation_id: str
    last_question: str
    last_question_id: str
    file_ids: List[str]
    created_date: str
    status: str
