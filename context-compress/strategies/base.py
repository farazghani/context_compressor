"""
Abstract base class for all compression strategies.

Every strategy:
 1. Implements _compress_impl() — the actual logic.
 2. Gets timing, before/after token counts, and CompressionResult
    construction for free from the base class.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

from ..core.models import CompressionResult
from ..tokenizers.base import BaseTokenizer
from ..tokenizers.registry import get_tokenizer


class BaseStrategy(ABC):
    """
    Subclass this and implement _compress_impl().
    Call compress() — never _compress_impl() directly.
    """

    #: Override in subclass with a short unique name e.g. "extractive"
    name: str = "base"

    def __init__(self, tokenizer: Optional[BaseTokenizer] = None):
        self.tokenizer = tokenizer or get_tokenizer()

    # ── Public entry point ────────────────────────────────────────────────────

    def compress(
        self,
        text: str,
        target_ratio: float = 0.5,
        token_budget: Optional[int] = None,
    ) -> CompressionResult:
        """
        Compress *text* and return a fully-populated CompressionResult.

        Args:
            text:         Input text to compress.
            target_ratio: Fraction of original tokens to keep (0 < r ≤ 1).
                          Ignored when *token_budget* is set.
            token_budget: Hard maximum token count for the output.
                          When provided, target_ratio is derived from it.
        """
        if not text or not text.strip():
            return self._empty_result(text, target_ratio, token_budget)

        original_tokens = self.tokenizer.count(text)

        # Derive ratio from budget when budget is given
        if token_budget is not None and original_tokens > 0:
            target_ratio = min(1.0, token_budget / original_tokens)

        target_ratio = max(0.0, min(1.0, target_ratio))

        start = time.perf_counter()
        compressed_text = self._compress_impl(text, target_ratio, token_budget)
        elapsed = time.perf_counter() - start

        compressed_tokens = self.tokenizer.count(compressed_text)
        actual_ratio = (
            compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        )

        fits_budget: Optional[bool] = None
        if token_budget is not None:
            fits_budget = compressed_tokens <= token_budget

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            strategy_used=self.name,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            processing_time=elapsed,
            target_ratio=target_ratio,
            actual_ratio=actual_ratio,
            token_budget=token_budget,
            fits_budget=fits_budget,
        )

    # ── Abstract hook ─────────────────────────────────────────────────────────

    @abstractmethod
    def _compress_impl(
        self,
        text: str,
        target_ratio: float,
        token_budget: Optional[int],
    ) -> str:
        """Return compressed text. Ratio and budget have already been normalised."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _empty_result(
        self,
        text: str,
        target_ratio: float,
        token_budget: Optional[int],
    ) -> CompressionResult:
        return CompressionResult(
            original_text=text,
            compressed_text=text,
            strategy_used=self.name,
            original_tokens=0,
            compressed_tokens=0,
            processing_time=0.0,
            target_ratio=target_ratio,
            actual_ratio=1.0,
            token_budget=token_budget,
            fits_budget=(token_budget is None or 0 <= token_budget),
        )