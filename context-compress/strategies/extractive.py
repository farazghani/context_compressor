"""
Extractive compression — TF-IDF sentence scoring.

Algorithm:
  1. Split text into sentences.
  2. Score each sentence with TF-IDF (numpy only, no sklearn).
  3. Keep the highest-scoring sentences in original order until we
     hit target_ratio or token_budget.

No external deps beyond numpy (already a near-universal dep).
This is the fastest strategy and the default.
"""

import re
from typing import List, Optional, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from .base import BaseStrategy
from ..tokenizers.base import BaseTokenizer
from ..config import SENTENCE_DELIMITERS, MIN_SENTENCE_TOKENS


class ExtrativeStrategy(BaseStrategy):
    name = "extractive"

    def __init__(self, tokenizer: Optional[BaseTokenizer] = None):
        super().__init__(tokenizer)

    # ── Core ──────────────────────────────────────────────────────────────────

    def _compress_impl(
        self,
        text: str,
        target_ratio: float,
        token_budget: Optional[int],
    ) -> str:
        sentences = self._split_sentences(text)

        # Nothing to compress — return as-is
        if len(sentences) <= 1:
            return text

        scores = self._score_sentences(sentences)
        return self._select_sentences(sentences, scores, target_ratio, token_budget)

    # ── Sentence splitting ────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        raw = re.split(SENTENCE_DELIMITERS, text.strip())
        # Filter out very short fragments
        return [
            s.strip()
            for s in raw
            if s.strip() and self.tokenizer.count(s.strip()) >= MIN_SENTENCE_TOKENS
        ]

    # ── TF-IDF scoring ────────────────────────────────────────────────────────

    def _score_sentences(self, sentences: List[str]) -> List[float]:
        if _HAS_NUMPY:
            return self._tfidf_scores(sentences)
        return self._frequency_scores(sentences)

    def _tfidf_scores(self, sentences: List[str]) -> List[float]:
        """Simple TF-IDF without sklearn."""
        import numpy as np

        # Tokenise to words (lowercased)
        tokenised = [re.findall(r"\b\w+\b", s.lower()) for s in sentences]

        # Vocabulary
        vocab = list({w for words in tokenised for w in words})
        word_idx = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        N = len(sentences)

        if V == 0 or N == 0:
            return [1.0] * N

        # TF matrix: shape (N, V)
        tf = np.zeros((N, V), dtype=float)
        for i, words in enumerate(tokenised):
            if not words:
                continue
            for w in words:
                tf[i, word_idx[w]] += 1
            tf[i] /= len(words)

        # IDF vector
        df = np.count_nonzero(tf, axis=0).astype(float)
        idf = np.log((N + 1) / (df + 1)) + 1.0  # smoothed

        tfidf = tf * idf  # broadcast
        scores = tfidf.sum(axis=1)  # sentence score = sum of word scores

        # Normalise to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
        return scores.tolist()

    def _frequency_scores(self, sentences: List[str]) -> List[float]:
        """Pure-Python fallback: word frequency scoring."""
        from collections import Counter

        all_words = re.findall(r"\b\w+\b", " ".join(sentences).lower())
        freq = Counter(all_words)
        total = sum(freq.values()) or 1

        scores = []
        for s in sentences:
            words = re.findall(r"\b\w+\b", s.lower())
            score = sum(freq[w] / total for w in words) / (len(words) or 1)
            scores.append(score)

        max_s = max(scores) if scores else 1.0
        return [s / max_s if max_s > 0 else 1.0 for s in scores]

    # ── Sentence selection ────────────────────────────────────────────────────

    def _select_sentences(
        self,
        sentences: List[str],
        scores: List[float],
        target_ratio: float,
        token_budget: Optional[int],
    ) -> str:
        original_tokens = sum(self.tokenizer.count(s) for s in sentences)

        if token_budget is not None:
            target_tokens = token_budget
        else:
            target_tokens = max(1, int(original_tokens * target_ratio))

        # Sort by score descending, keep original order for selected set
        ranked: List[Tuple[int, float]] = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        selected_indices = set()
        accumulated = 0

        for idx, _score in ranked:
            tok = self.tokenizer.count(sentences[idx])
            if accumulated + tok <= target_tokens:
                selected_indices.add(idx)
                accumulated += tok
            if accumulated >= target_tokens:
                break

        # Preserve original order
        kept = [sentences[i] for i in sorted(selected_indices)]
        return " ".join(kept) if kept else sentences[0]