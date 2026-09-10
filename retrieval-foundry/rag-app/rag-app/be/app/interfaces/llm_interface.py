"""Abstract interface for LLM operations."""
from abc import ABC, abstractmethod
from typing import Tuple


class LLMInterface(ABC):
    """Abstract base class for LLM service providers."""

    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        conversation_history: list = None
    ) -> Tuple[str, dict]:
        """Send a chat request to the LLM.

        Args:
            system_prompt: System instruction for the LLM.
            user_prompt: User message/query.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            conversation_history: Optional list of prior turns as message dicts,
                e.g. [{"role": "user", "content": "..."},
                      {"role": "assistant", "content": "..."}]. Defaults to None
                (treated as no history).

        Returns:
            Tuple of (response_text, token_usage_dict).
            token_usage_dict: {"prompt_tokens": int, "completion_tokens": int}
        """
        ...
