"""
Tokenizer registry.

Single place to get the right tokenizer. Nothing in the codebase
should import TiktokenTokenizer or ApproximateTokenizer directly —
always go through get_tokenizer().

Usage:
    from tokenizers.registry import get_tokenizer

    tokenizer = get_tokenizer()               # approximate (no deps)
    tokenizer = get_tokenizer("openai")       # tiktoken, cl100k_base
    tokenizer = get_tokenizer("claude")       # tiktoken, cl100k_base
    tokenizer = get_tokenizer("gpt-4")        # tiktoken, cl100k_base
    tokenizer = get_tokenizer("approximate")  # no deps fallback
"""

from .base import BaseTokenizer
from .approximate import ApproximateTokenizer


# Providers that should use tiktoken for accurate counts
TIKTOKEN_PROVIDERS = {
    "openai",
    "anthropic",
    "claude",
    "claude-3",
    "claude-3-5",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5",
    "gpt-3.5-turbo",
    "tiktoken",
    "accurate",
}


def get_tokenizer(provider: str = "approximate") -> BaseTokenizer:
    """
    Get the right tokenizer for a given provider.

    Falls back to ApproximateTokenizer if tiktoken is not installed,
    with a warning so the user knows counts may be off.

    Args:
        provider: One of "openai", "claude", "anthropic", "gpt-4",
                  "approximate", or any model name string.

    Returns:
        BaseTokenizer instance ready to use.
    """
    provider = provider.lower().strip()

    if provider in TIKTOKEN_PROVIDERS:
        try:
            from .tiktoken import TiktokenTokenizer
            return TiktokenTokenizer(model=provider)
        except Exception:
            import warnings
            warnings.warn(
                f"Could not initialize TiktokenTokenizer for provider '{provider}' "
                f"(tiktoken may not be installed or BPE file could not be downloaded). "
                f"Falling back to ApproximateTokenizer — token counts may be slightly off. "
                f"Install with: pip install tiktoken",
                RuntimeWarning,
                stacklevel=2,
            )
            return ApproximateTokenizer()

    # Default: approximate tokenizer, no deps required
    return ApproximateTokenizer()