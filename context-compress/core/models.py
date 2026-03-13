"""
Core data models for the context compressor.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
import json


@dataclass
class CompressionResult:
    """Result of a single text compression operation."""

    original_text: str
    compressed_text: str
    strategy_used: str
    original_tokens: int
    compressed_tokens: int
    processing_time: float       # in seconds
    target_ratio: float          # what you asked for  e.g. 0.5
    actual_ratio: float          # what you actually got e.g. 0.48
    token_budget: Optional[int] = None    # max tokens you were trying to fit into
    fits_budget: Optional[bool] = None    # did the compressed text fit?

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return self.original_tokens - self.compressed_tokens

    @property
    def savings_percentage(self) -> float:
        """Percentage of tokens saved."""
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "compressed_text": self.compressed_text,
            "strategy_used": self.strategy_used,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "savings_percentage": round(self.savings_percentage, 2),
            "processing_time": self.processing_time,
            "target_ratio": self.target_ratio,
            "actual_ratio": self.actual_ratio,
            "token_budget": self.token_budget,
            "fits_budget": self.fits_budget,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return (
            f"CompressionResult("
            f"strategy={self.strategy_used!r}, "
            f"tokens={self.original_tokens}→{self.compressed_tokens}, "
            f"saved={self.savings_percentage:.1f}%, "
            f"fits_budget={self.fits_budget})"
        )


@dataclass
class CacheEntry:
    """A single entry in the compression cache."""

    key: str
    result: CompressionResult
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if this entry has passed its TTL."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    def access(self) -> None:
        """Record an access hit."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class CompressionStats:
    """Running stats across all compression calls — useful for cost tracking."""

    total_compressions: int = 0
    total_tokens_processed: int = 0
    total_tokens_saved: int = 0
    total_processing_time: float = 0.0
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    @property
    def overall_savings_percentage(self) -> float:
        if self.total_tokens_processed == 0:
            return 0.0
        return (self.total_tokens_saved / self.total_tokens_processed) * 100

    def update(self, result: CompressionResult) -> None:
        """Update stats from a new compression result."""
        self.total_compressions += 1
        self.total_tokens_processed += result.original_tokens
        self.total_tokens_saved += result.tokens_saved
        self.total_processing_time += result.processing_time
        self.strategy_usage[result.strategy_used] = (
            self.strategy_usage.get(result.strategy_used, 0) + 1
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_compressions": self.total_compressions,
            "total_tokens_processed": self.total_tokens_processed,
            "total_tokens_saved": self.total_tokens_saved,
            "overall_savings_percentage": round(self.overall_savings_percentage, 2),
            "total_processing_time": round(self.total_processing_time, 3),
            "strategy_usage": self.strategy_usage,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 2),
        }