"""Repository for response records (prompt + answer pairs).

Supports two storage backends:
- STORAGE_PROVIDER=json  → json_store.py (local files)
- STORAGE_PROVIDER=mongodb → mongo_store.py (MongoDB Atlas)

Archived responses always use JSON file storage regardless of provider.
"""
from typing import List, Optional, Dict, Any
from app.repositories import json_store
from app.schemas.models import ResponseRecord
from app.config import load_settings
from app.utils.helpers import get_current_time

ARCHIVE_COLLECTION = "archived_responses"  # Always JSON file


def _use_mongo() -> bool:
    """Check if MongoDB is the configured storage provider."""
    return load_settings().STORAGE_PROVIDER == "mongodb"


def _get_collection_name() -> str:
    """Get the collection name for responses."""
    settings = load_settings()
    return settings.MONGODB_COLLECTION_RESPONSES


async def save_response(record: ResponseRecord) -> str:
    """Save a response record. Returns response_id."""
    data = record.model_dump()
    if _use_mongo():
        from app.repositories import mongo_store
        await mongo_store.insert_one(_get_collection_name(), data)
    else:
        json_store.insert_one("responses", data)
    return record.response_id


async def get_response(response_id: str) -> Optional[ResponseRecord]:
    """Get a response by response_id."""
    if _use_mongo():
        from app.repositories import mongo_store
        data = await mongo_store.find_one(_get_collection_name(), {"response_id": response_id})
    else:
        data = json_store.find_one("responses", {"response_id": response_id})
    if data:
        return ResponseRecord(**data)
    return None


async def get_response_by_question_id(question_id: str) -> Optional[ResponseRecord]:
    """Get a response by question_id."""
    if _use_mongo():
        from app.repositories import mongo_store
        data = await mongo_store.find_one(_get_collection_name(), {"question_id": question_id})
    else:
        data = json_store.find_one("responses", {"question_id": question_id})
    if data:
        return ResponseRecord(**data)
    return None


async def get_conversation_history(conversation_id: str, limit: int = 1) -> List[ResponseRecord]:
    """Get recent response records for a conversation (for conversation history).

    Args:
        conversation_id: The conversation to query.
        limit: Number of most recent responses to retrieve.

    Returns:
        List of response records, most recent first.
    """
    if _use_mongo():
        from app.repositories import mongo_store
        records = await mongo_store.find_many(
            _get_collection_name(),
            {"conversation_id": conversation_id},
            sort_field="created_date",
            sort_desc=True,
            limit=limit
        )
    else:
        records = json_store.find_many(
            "responses",
            {"conversation_id": conversation_id},
            sort_field="created_date",
            sort_desc=True,
            limit=limit
        )
    return [ResponseRecord(**r) for r in records]


def archive_response(response_id: str, version: int, original_data: dict) -> None:
    """Archive an existing response before overwriting it during regeneration.

    Always uses JSON file storage (not MongoDB).

    Args:
        response_id: The response to archive.
        version: Regeneration version number.
        original_data: The full original response data to archive.
    """
    archive_record = {
        "response_id": response_id,
        "regeneration_version": version,
        "archived_date": get_current_time(),
        "original_summary": original_data.get("summary", ""),
        "original_citations": original_data.get("citations", []),
        "original_token_cost": original_data.get("token_cost", {}),
        "original_prompt_question": original_data.get("prompt_question", ""),
        "conversation_id": original_data.get("conversation_id", ""),
        "question_id": original_data.get("question_id", ""),
        "created_date": original_data.get("created_date", "")
    }

    json_store.insert_one(ARCHIVE_COLLECTION, archive_record)


async def update_response(response_id: str, update_data: Dict[str, Any]) -> int:
    """Update an existing response record in-place.

    Used during regeneration to overwrite summary, citations, token_cost, etc.

    Args:
        response_id: The response to update.
        update_data: Dict of fields to overwrite.

    Returns:
        Number of records modified (0 or 1).
    """
    if _use_mongo():
        from app.repositories import mongo_store
        return await mongo_store.update_one(
            _get_collection_name(),
            {"response_id": response_id},
            update_data
        )
    else:
        return json_store.update_one("responses", {"response_id": response_id}, update_data)


def get_archive_count(response_id: str) -> int:
    """Get the number of times a response has been regenerated (archived versions count).

    Always uses JSON file storage.
    """
    archives = json_store.find_many(ARCHIVE_COLLECTION, {"response_id": response_id})
    return len(archives)
