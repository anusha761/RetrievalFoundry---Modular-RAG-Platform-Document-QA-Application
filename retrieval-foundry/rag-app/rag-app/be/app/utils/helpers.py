"""Utility helper functions."""
import uuid
from datetime import datetime, timezone


def generate_id() -> str:
    """Generate a unique ID string."""
    return str(uuid.uuid4())


def get_current_time() -> str:
    """Get current UTC time as formatted string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
