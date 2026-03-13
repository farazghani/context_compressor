"""
Central configuration for context_compressor.
All magic numbers live here — never hardcode them elsewhere.
"""

# ── Compression defaults ──────────────────────────────────────────────────────
DEFAULT_RATIO: float = 0.5           # keep 50% of original by default
DEFAULT_TOKEN_BUDGET: int = 4096     # sensible GPT-4 / Claude budget
DEFAULT_STRATEGY: str = "extractive" # fast, no extra deps, good quality

# ── Tokenizer ─────────────────────────────────────────────────────────────────
DEFAULT_TOKENIZER: str = "approximate"
CHARS_PER_TOKEN: float = 4.0         # used by ApproximateTokenizer

# ── Extractive strategy ───────────────────────────────────────────────────────
MIN_SENTENCE_TOKENS: int = 3         # discard very short sentences
SENTENCE_DELIMITERS: str = r"(?<=[.!?])\s+"

# ── Semantic dedup ────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.85   # cosine sim above this = duplicate

# ── Abstractive strategy ──────────────────────────────────────────────────────
ABSTRACTIVE_MODEL: str = "claude-sonnet-4-20250514"
ABSTRACTIVE_PROMPT_TEMPLATE: str = (
    "Summarize the following text in at most {max_tokens} tokens. "
    "Preserve all key facts, numbers, and named entities. "
    "Return only the summary, no preamble.\n\n{text}"
)

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_MAX_SIZE: int = 256            # max entries in LRU cache
CACHE_DEFAULT_TTL: int = 3600        # seconds (1 hour); None = forever