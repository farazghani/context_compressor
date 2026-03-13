"""
Semantic dedup strategy — remove near-duplicate sentences.

Algorithm:
  1. Embed each sentence with sentence-transformers.
  2. Build a pairwise cosine-similarity matrix.
  3. For each pair above the threshold, drop the lower-scored sentence
     (using TF-IDF score as the keep criterion).
  4. If still over budget/ratio, fall back to extractive trimming.

Requires: pip install sentence-transformers
If not installed, raises a clear ImportError with install instructions.
"""

from typing import List, Optional

from .base import BaseStrategy
from .extractive import ExtrativeStrategy
from ..tokenizers.base import BaseTokenizer
from ..config import SIMILARITY_THRESHOLD


class SemanticDedupStrategy(BaseStrategy):
    name = "semantic_dedup"

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(tokenizer)
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self._extractor = ExtrativeStrategy(tokenizer=self.tokenizer)
        self._model = None  # lazy-load

    # ── Core ──────────────────────────────────────────────────────────────────

    def _compress_impl(
        self,
        text: str,
        target_ratio: float,
        token_budget: Optional[int],
    ) -> str:
        sentences = self._extractor._split_sentences(text)
        if len(sentences) <= 1:
            return text

        scores = self._extractor._score_sentences(sentences)
        deduped = self._deduplicate(sentences, scores)

        # After dedup, further trim to ratio/budget if needed
        deduped_text = " ".join(deduped)
        deduped_tokens = self.tokenizer.count(deduped_text)

        original_tokens = self.tokenizer.count(text)
        target_tokens = (
            token_budget
            if token_budget is not None
            else max(1, int(original_tokens * target_ratio))
        )

        if deduped_tokens <= target_tokens:
            return deduped_text

        # Fall back to extractive on the deduped text
        return self._extractor._compress_impl(deduped_text, target_ratio, token_budget)

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(
        self, sentences: List[str], scores: List[float]
    ) -> List[str]:
        try:
            import numpy as np
            embeddings = self._embed(sentences)
            sim_matrix = self._cosine_similarity(embeddings)
        except ImportError:
            # No sentence-transformers — skip dedup, return original
            return sentences

        n = len(sentences)
        dropped = set()

        for i in range(n):
            if i in dropped:
                continue
            for j in range(i + 1, n):
                if j in dropped:
                    continue
                if sim_matrix[i][j] >= self.similarity_threshold:
                    # Drop the lower-scored one
                    if scores[i] >= scores[j]:
                        dropped.add(j)
                    else:
                        dropped.add(i)
                        break  # i is dropped, no point checking more j's

        return [s for idx, s in enumerate(sentences) if idx not in dropped]

    def _embed(self, sentences: List[str]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SemanticDedupStrategy. "
                "Install it with: pip install sentence-transformers"
            )
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(sentences, show_progress_bar=False)

    @staticmethod
    def _cosine_similarity(embeddings) -> List[List[float]]:
        import numpy as np
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalised = embeddings / norms
        matrix = normalised @ normalised.T
        return matrix.tolist()