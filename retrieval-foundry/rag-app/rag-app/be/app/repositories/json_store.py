"""JSON file-based storage layer.

Provides simple CRUD operations backed by JSON files.
Designed to be swappable with MongoDB, DynamoDB, etc.
"""
import json
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_file_path(collection: str) -> Path:
    """Get the JSON file path for a collection."""
    _ensure_data_dir()
    return DATA_DIR / f"{collection}.json"


def _read_collection(collection: str) -> List[Dict[str, Any]]:
    """Read all records from a collection."""
    file_path = _get_file_path(collection)
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content:
            return json.loads(content)
        return []


def _write_collection(collection: str, records: List[Dict[str, Any]]):
    """Write all records to a collection."""
    file_path = _get_file_path(collection)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def insert_one(collection: str, record: Dict[str, Any]) -> str:
    """Insert a single record. Returns the record's ID."""
    records = _read_collection(collection)
    records.append(record)
    _write_collection(collection, records)
    # Return the id field (convention: records should have a unique id field)
    return record.get("question_id") or record.get("response_id") or record.get("conversation_id", "")


def find_one(collection: str, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a single record matching all filter criteria."""
    records = _read_collection(collection)
    for record in records:
        if all(record.get(k) == v for k, v in filter_dict.items()):
            return record
    return None


def find_many(
    collection: str,
    filter_dict: Dict[str, Any] = None,
    sort_field: str = None,
    sort_desc: bool = True,
    limit: int = None
) -> List[Dict[str, Any]]:
    """Find records matching filter criteria."""
    records = _read_collection(collection)

    if filter_dict:
        records = [
            r for r in records
            if all(r.get(k) == v for k, v in filter_dict.items())
        ]

    if sort_field:
        records.sort(key=lambda r: r.get(sort_field, ""), reverse=sort_desc)

    if limit:
        records = records[:limit]

    return records


def update_one(collection: str, filter_dict: Dict[str, Any], update_data: Dict[str, Any]) -> int:
    """Update the first matching record. Returns number of modified records (0 or 1)."""
    records = _read_collection(collection)
    for i, record in enumerate(records):
        if all(record.get(k) == v for k, v in filter_dict.items()):
            records[i].update(update_data)
            _write_collection(collection, records)
            return 1
    return 0
