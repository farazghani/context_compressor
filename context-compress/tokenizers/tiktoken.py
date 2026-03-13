"""
Accurate tokenizer using tiktoken.
Works for OpenAI models and is a close approximation for Claude.

Supported model shortcuts:
    "gpt-4"         -> cl100k_base encoding
    "gpt-3.5-turbo" -> cl100k_base encoding
    "claude"        -> cl100k_base encoding  (close approximation)
    "openai"        -> cl100k_base encoding
"""

from typing import List
from .base import BaseTokenizer


class TiktokenTokenizer(BaseTokenizer):

    # All modern models (GPT-4, GPT-3.5, Claude) use roughly the same
    # tokenization — cl100k_base is accurate for OpenAI and a solid
    # approximation for Anthropic.
    ENCODING_MAP = {
        "gpt-4":            "cl100k_base",
        "gpt-4-turbo":      "cl100k_base",
        "gpt-3.5-turbo":    "cl100k_base",
        "gpt-3.5":          "cl100k_base",
        "openai":           "cl100k_base",
        "claude":           "cl100k_base",
        "anthropic":        "cl100k_base",
        "claude-3":         "cl100k_base",
        "claude-3-5":       "cl100k_base",
        "default":          "cl100k_base",
    }

    def __init__(self, model: str = "default"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. "
                "Install it with: pip install tiktoken"
            )

        encoding_name = self.ENCODING_MAP.get(model.lower(), "cl100k_base")

        try:
            self.enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to cl100k_base if model-specific encoding fails
            self.enc = tiktoken.get_encoding("cl100k_base")

        self.model = model
        self.encoding_name = encoding_name

    def count(self, text: str) -> int:
        """Count tokens accurately using tiktoken."""
        if not text:
            return 0
        return len(self.enc.encode(text))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not text:
            return []
        return self.enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        if not tokens:
            return ""
        return self.enc.decode(tokens)

    def __repr__(self) -> str:
        return f"TiktokenTokenizer(model={self.model!r}, encoding={self.encoding_name!r})"