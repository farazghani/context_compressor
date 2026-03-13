#A Python library that takes a large text, compresses it to fit a token budget, and returns the compressed text ready to send to any LLM API.


```
context_compressor/
│
├── core/
│   ├── __init__.py
│   ├── models.py              
│   ├── compressor.py          # main entry point
│   └── strategy_manager.py    # picks the right strategy
│
├── strategies/
│   ├── __init__.py
│   ├── base.py                # abstract base class
│   ├── extractive.py          # picks best sentences (TF-IDF)
│   ├── abstractive.py         # LLM-based summarization
│   ├── token_budget.py        # hard trim to fit X tokens
│   └── semantic_dedup.py      # remove near-duplicate sentences
│
├── tokenizers/
│   ├── __init__.py
│   ├── base.py                # abstract tokenizer
│   ├── tiktoken.py    # accurate counts (OpenAI/Claude)
│   ├── approximate.py         # fallback, no deps
│   └── registry.py            # get_tokenizer("openai")
│
├── cache/
│   ├── __init__.py
│   └── manager.py             # LRU + TTL cache
│
├── __init__.py                # public API
└── config.py                  # defaults (ratios, budgets, models)

```



# How Each Piece Works
```
tokenizers — accurate token counting. tiktoken for real counts, approximate as fallback. Every strategy uses this, never len(text.split()).
strategies/base — abstract class with one method compress(text, target_ratio, token_budget). Every strategy implements this. Base handles timing, token counting before/after, building CompressionResult.
strategies/extractive — split into sentences → score each with TF-IDF → keep top N sentences until ratio/budget is met. No external deps. Fastest.
strategies/token_budget — keep dropping lowest-scored sentences one at a time until token_count <= budget. The most important strategy for API use.
strategies/semantic_dedup — embed each sentence → find pairs with cosine similarity > 0.85 → drop the duplicate. Good for repetitive long docs.
strategies/abstractive — send text to an LLM with a "summarize to N tokens" prompt. Best quality, slowest, costs money.
cache/manager — SHA-256 key from hash(text) + strategy + ratio/budget. In-memory LRU with TTL. Thread-safe.
core/strategy_manager — holds all registered strategies. Auto-selects based on: text length, whether a budget or ratio was passed, user preference.
core/compressor — the only thing users import. Two methods:
pythoncompressor.compress(text, target_ratio=0.5)
compressor.compress_to_budget(text, max_tokens=4096)
config.py — one place for all defaults:
pythonDEFAULT_RATIO = 0.5
DEFAULT_TOKEN_BUDGET = 4096
DEFAULT_STRATEGY = "extractive"
CHARS_PER_TOKEN = 4.0
__init__.py — exposes only:
```
python
```
from context_compressor import Compressor

What the User Experience Looks Like at the End
pythonfrom context_compressor import Compressor

c = Compressor(tokenizer="openai")

# by ratio
result = c.compress(long_text, target_ratio=0.5)

# by token budget (the API use case)
result = c.compress_to_budget(long_text, max_tokens=4096)

print(result.compressed_text)
print(result.tokens_saved)       # 3200
print(result.savings_percentage) # 61.4%
print(result.fits_budget)        # True
```

---

## Dependencies
```
tiktoken          # accurate token counting
numpy             # TF-IDF math in extractive
sentence-transformers  # semantic dedup only
Abstractive needs no extra dep — it just calls whatever LLM API you're already using.
