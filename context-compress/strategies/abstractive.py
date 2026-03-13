"""
Abstractive compression — delegates to an LLM for best-quality summarisation.
 
The LLM call is fully pluggable: pass any callable that matches
    call_llm(prompt: str) -> str
 
If you don't pass one, the strategy tries to use the Anthropic SDK
(claude-sonnet-4-20250514) via ANTHROPIC_API_KEY in your environment.
 
Usage:
    from strategies.abstractive import AbstractiveStrategy
 
    # Use default Anthropic client
    strategy = AbstractiveStrategy()
 
    # Use your own function
    strategy = AbstractiveStrategy(llm_fn=my_openai_wrapper)
"""