"""
Abstract base class for all tokenizers.
Every tokenizer must implement count, encode, and decode.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):

    @abstractmethod
    def count(self, text: str) -> int:
        """Count the number of tokens in text."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode a list of token IDs back into text."""
        pass

    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """
        Hard truncate text to fit within max_tokens.
        Used as a last resort when compression isn't enough.
        """
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])

    def fits_budget(self, text: str, max_tokens: int) -> bool:
        """Check if text fits within the token budget."""
        return self.count(text) <= max_tokens
    