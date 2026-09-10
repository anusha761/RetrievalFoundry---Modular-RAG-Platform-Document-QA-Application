"""Internal data models for pipeline state and storage."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.utils.helpers import generate_id, get_current_time


class PipelineState(BaseModel):
    """Typed state object that flows through the pipeline.
    
    Replaces the untyped 'attributes' dict pattern from the original project
    with a properly typed Pydantic model.
    """
    question_id: str = ""
    conversation_id: str = ""
    question: str = ""
    file_ids: List[str] = []
    previous_question: str = ""
    retriever_query: str = ""
    keywords: List[str] = []
    keywords_token_usage: Optional[dict] = None
    retrieved_chunks: List[dict] = []
    batch_summaries: List[dict] = []
    total_batch_input_tokens: int = 0
    total_batch_output_tokens: int = 0
    summary: str = ""
    citations: List[dict] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    regenerate: bool = False
    conversation_history_ids: List[str] = []  # response_ids of previous responses used as context
    created_date: str = ""
    created_by: Dict[str, str] = Field(default_factory=lambda: {"analyst_name": "user", "email_id": "user@example.com"})


class ConversationRecord(BaseModel):
    """Stored conversation record."""
    conversation_id: str = Field(default_factory=generate_id)
    last_question: str = ""
    last_question_id: str = ""
    file_ids: List[str] = []
    created_date: str = Field(default_factory=get_current_time)
    created_by: Dict[str, str] = Field(default_factory=lambda: {"analyst_name": "user", "email_id": "user@example.com"})
    status: str = "active"


class ResponseRecord(BaseModel):
    """Stored response record (one per question-answer turn).
    
    Mirrors the GaitPromptRouteResponse from the original project.
    """
    response_id: str = Field(default_factory=generate_id)
    conversation_id: str = ""
    question_id: str = ""
    prompt_question: str = ""
    # refined_keywords: List[str] = []
    summary: str = ""
    citations: List[dict] = []
    file_ids: List[str] = []
    token_cost: dict = {}
    conversation_history_ids: List[str] = []
    is_regeneration: bool = False
    created_by: Dict[str, str] = Field(default_factory=lambda: {"analyst_name": "user", "email_id": "user@example.com"})
    created_date: str = Field(default_factory=get_current_time)
    updated_date: str = Field(default_factory=get_current_time)
    prompt_status: str = "active"
    prompt_type: str = "ask_me_prompt"
