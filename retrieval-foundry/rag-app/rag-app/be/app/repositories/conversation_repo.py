"""Repository for conversation records.

Supports two storage backends:
- STORAGE_PROVIDER=json  → json_store.py (local files)
- STORAGE_PROVIDER=mongodb → mongo_store.py (MongoDB Atlas)
"""
from typing import List, Optional
from app.repositories import json_store
from app.schemas.models import ConversationRecord
from app.config import load_settings
from app.utils.helpers import get_current_time


def _use_mongo() -> bool:
    """Check if MongoDB is the configured storage provider."""
    return load_settings().STORAGE_PROVIDER == "mongodb"


def _get_collection_name() -> str:
    """Get the collection name for conversations."""
    settings = load_settings()
    return settings.MONGODB_COLLECTION_CONVERSATIONS


async def create_conversation(file_ids: List[str]) -> ConversationRecord:
    """Create a new conversation record."""
    record = ConversationRecord(file_ids=file_ids)
    data = record.model_dump()
    if _use_mongo():
        from app.repositories import mongo_store
        await mongo_store.insert_one(_get_collection_name(), data)
    else:
        json_store.insert_one("conversations", data)
    return record


async def get_conversation(conversation_id: str) -> Optional[ConversationRecord]:
    """Get a conversation by ID."""
    if _use_mongo():
        from app.repositories import mongo_store
        data = await mongo_store.find_one(_get_collection_name(), {"conversation_id": conversation_id})
    else:
        data = json_store.find_one("conversations", {"conversation_id": conversation_id})
    if data:
        return ConversationRecord(**data)
    return None


async def update_conversation_last_question(conversation_id: str, question: str, question_id: str):
    """Update the last question pointer on a conversation."""
    update_data = {
        "last_question": question,
        "last_question_id": question_id,
        "updated_date": get_current_time()
    }
    if _use_mongo():
        from app.repositories import mongo_store
        await mongo_store.update_one(
            _get_collection_name(),
            {"conversation_id": conversation_id},
            update_data
        )
    else:
        json_store.update_one("conversations", {"conversation_id": conversation_id}, update_data)


async def list_conversations(status: str = "active") -> List[ConversationRecord]:
    """List all conversations with given status."""
    if _use_mongo():
        from app.repositories import mongo_store
        records = await mongo_store.find_many(
            _get_collection_name(),
            {"status": status},
            sort_field="created_date",
            sort_desc=True
        )
    else:
        records = json_store.find_many(
            "conversations",
            {"status": status},
            sort_field="created_date",
            sort_desc=True
        )
    return [ConversationRecord(**r) for r in records]
