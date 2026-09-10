"""Request schemas for API endpoints."""
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr


class UserDetails(BaseModel):
    """User identity — mirrors original project's UserDetails."""
    analyst_name: str = "user"
    email_id: EmailStr = "user@example.com"


class ChatRequest(BaseModel):
    """Request body for the monolithic /chat/ask endpoint."""
    question: str = Field(..., min_length=1, description="User's question")
    file_ids: List[str] = Field(..., min_length=1, description="List of selected file IDs")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID to continue")
    created_by: UserDetails = Field(default_factory=UserDetails)


class RegenerateRequest(BaseModel):
    """Request body for /chat/regenerate endpoint."""
    conversation_id: str
    question_id: str
    edited_question: Optional[str] = Field(None, description="Optional rephrased question. If provided, regeneration uses this instead of the original.")
    created_by: UserDetails = Field(default_factory=UserDetails)


class RecordPromptRequest(BaseModel):
    """Request body for pipeline/record-prompt."""
    question: str = Field(..., min_length=1)
    file_ids: List[str] = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    created_by: UserDetails = Field(default_factory=UserDetails)





class RetrieveRequest(BaseModel):
    """Request body for pipeline/retrieve."""
    retriever_query: str = Field(..., description="user query")
    file_ids: List[str]


class SummarizeRequest(BaseModel):
    """Request body for pipeline/summarize."""
    question: str
    question_id: str
    conversation_id: str
    file_ids: List[str]
    retriever_query: str
    retrieved_chunks: List[dict] = Field(..., description="Retrieved chunks grouped by file")
    keywords_token_usage: Optional[dict] = None
    regenerate: bool = False
    created_by: UserDetails = Field(default_factory=UserDetails)
