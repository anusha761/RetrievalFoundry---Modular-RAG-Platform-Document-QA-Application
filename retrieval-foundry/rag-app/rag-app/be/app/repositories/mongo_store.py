"""MongoDB storage layer using Motor (async).

Provides generic CRUD operations backed by MongoDB.
Mirrors the interface of json_store.py so repositories can swap seamlessly.
"""
from typing import List, Optional, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import load_settings

# Module-level client (initialized once)
_client: Optional[AsyncIOMotorClient] = None
_db = None


def _get_db():
    """Get or create the MongoDB database connection."""
    global _client, _db
    if _db is None:
        settings = load_settings()
        _client = AsyncIOMotorClient(settings.MONGODB_URI)
        # Extract DB name from URI or use default
        _db = _client.get_default_database("doc_rag_chat_db")
    return _db


def _get_collection(collection_name: str):
    """Get a MongoDB collection by name."""
    db = _get_db()
    return db[collection_name]


async def insert_one(collection_name: str, record: Dict[str, Any]) -> str:
    """Insert a single record. Returns the record's business ID (not _id)."""
    collection = _get_collection(collection_name)
    await collection.insert_one(record)
    return record.get("question_id") or record.get("response_id") or record.get("conversation_id", "")


async def find_one(collection_name: str, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a single record matching all filter criteria."""
    collection = _get_collection(collection_name)
    result = await collection.find_one(filter_dict, {"_id": 0})
    return result


async def find_many(
    collection_name: str,
    filter_dict: Dict[str, Any] = None,
    sort_field: str = None,
    sort_desc: bool = True,
    limit: int = None
) -> List[Dict[str, Any]]:
    """Find records matching filter criteria."""
    collection = _get_collection(collection_name)

    query_filter = filter_dict or {}
    cursor = collection.find(query_filter, {"_id": 0})

    if sort_field:
        direction = -1 if sort_desc else 1
        cursor = cursor.sort(sort_field, direction)

    if limit:
        cursor = cursor.limit(limit)

    results = await cursor.to_list(length=limit or 1000)
    return results


async def update_one(collection_name: str, filter_dict: Dict[str, Any], update_data: Dict[str, Any]) -> int:
    """Update the first matching record. Returns number of modified records (0 or 1)."""
    collection = _get_collection(collection_name)
    result = await collection.update_one(filter_dict, {"$set": update_data})
    return result.modified_count
