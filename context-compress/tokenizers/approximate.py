"""
Approximate tokenizer — no external dependencies.
 
Uses the ~4 chars per token heuristic which is accurate enough
for rough estimation. Use TiktokenTokenizer when you need exact counts.
 
When to use this:
- You don't have tiktoken installed
- You just want a quick estimate
- You're doing ratio-based compression (not hard token budgets)
 
When NOT to use this:
- You're doing hard token budget trimming (use TiktokenTokenizer)
- You need to know exactly whether text fits a context window
"""
 
from typing import List
from .base import BaseTokenizer
 
 
class ApproximateTokenizer(BaseTokenizer):
 
    # OpenAI / Anthropic models average ~4 characters per token.
    # This varies by content: code is ~3.5, prose is ~4.5.
    CHARS_PER_TOKEN = 4.0
 
    def __init__(self, chars_per_token: float = CHARS_PER_TOKEN):
        self.chars_per_token = chars_per_token
 
    def count(self, text: str) -> int:
        """Estimate token count from character count."""
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))
 
    def encode(self, text: str) -> List[int]:
        """
        Approximate encode — returns a list of fake token IDs whose
        length exactly matches count(). Good enough for truncation logic,
        not for real decoding.
        """
        if not text:
            return []
        n = self.count(text)
        return list(range(n))  # fake IDs: [0, 1, 2, ... n-1]
 
    def decode(self, tokens: List[int]) -> str:
        """
        Approximate decode — not meaningful for fake token IDs.
        Exists to satisfy the interface.
        """
        raise NotImplementedError(
            "ApproximateTokenizer does not support real decoding. "
            "Use TiktokenTokenizer if you need encode/decode."
        )
 
    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """
        Truncate by character count approximation.
        Overrides base to avoid calling decode().
        """
        if self.count(text) <= max_tokens:
            return text
        max_chars = int(max_tokens * self.chars_per_token)
        # Truncate at a word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated
 
    def __repr__(self) -> str:
        return f"ApproximateTokenizer(chars_per_token={self.chars_per_token})"