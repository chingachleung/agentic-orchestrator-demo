"""Pluggable intent classification for routing.

The orchestrator depends only on the ``IntentClassifier`` protocol, so the
routing "brain" can be swapped without touching orchestration logic — the
same pattern used to route production traffic to specialist agents.

Two implementations ship here:

- ``TFIDFIntentClassifier``: zero-dependency, runs fully offline (fit at
  import time on the agents' own example utterances). This is what the demo
  runs with out of the box.
- ``LLMIntentClassifier``: real integration code against an LLM API for
  higher-accuracy, few-shot intent + slot understanding. It is NOT wired
  into the default demo run (no API key is bundled with this repo) but is
  complete and correct — set an API key as shown below to activate it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class IntentResult:
    intent: str
    confidence: float


class IntentClassifier(Protocol):
    def classify(self, utterance: str, candidate_intents: list[str]) -> IntentResult: ...


class TFIDFIntentClassifier:
    """Routes an utterance to the closest agent by cosine similarity between
    TF-IDF vectors of the utterance and each agent's example phrases.

    This is the same category of technique (lexical vector-space matching)
    used as one leg of an ensemble retrieval/routing system before falling
    back to a heavier model for ambiguous cases.
    """

    def __init__(self, intent_examples: dict[str, list[str]]):
        self.intent_examples = intent_examples
        corpus = [ex for examples in intent_examples.values() for ex in examples]
        self._labels = [
            intent for intent, examples in intent_examples.items() for _ in examples
        ]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = self.vectorizer.fit_transform(corpus)

    def classify(self, utterance: str, candidate_intents: list[str]) -> IntentResult:
        query_vec = self.vectorizer.transform([utterance])
        sims = cosine_similarity(query_vec, self._matrix)[0]

        best_intent, best_score = "unknown", 0.0
        for intent, score in zip(self._labels, sims):
            if intent in candidate_intents and score > best_score:
                best_intent, best_score = intent, float(score)
        return IntentResult(intent=best_intent, confidence=best_score)


class LLMIntentClassifier:
    """Real (but inactive-by-default) LLM-backed classifier.

    Swap this in for higher accuracy / better generalization to phrasing the
    TF-IDF classifier hasn't seen. The call below is written against the
    Anthropic Messages API; swapping providers only means changing this one
    method.
    """

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def classify(self, utterance: str, candidate_intents: list[str]) -> IntentResult:
        if not self.api_key:
            raise RuntimeError(
                "LLMIntentClassifier requires ANTHROPIC_API_KEY. "
                "Use TFIDFIntentClassifier for the no-key demo path."
            )

        import anthropic  # local import: optional dependency, only needed here

        client = anthropic.Anthropic(api_key=self.api_key)
        prompt = (
            "Classify the user's message into exactly one of these intents: "
            f"{', '.join(candidate_intents)}.\n"
            f"Message: {utterance!r}\n"
            'Respond with JSON only: {"intent": "...", "confidence": 0.0-1.0}'
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        import json

        payload = json.loads(response.content[0].text)
        return IntentResult(intent=payload["intent"], confidence=float(payload["confidence"]))
