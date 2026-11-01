"""High-level pipeline: text -> claims -> graph -> features -> verdict."""

from __future__ import annotations

from .extractor import extract_claims
from .features import FEATURE_NAMES, build_features, explain
from .graph import CausalGraph
from .knowledge import find_myths
from .model import load_or_train, vectorise

FAKE_THRESHOLD = 0.65
RELIABLE_THRESHOLD = 0.35
MYTH_FLOOR = 0.90


def compute_features(text: str) -> dict[str, float]:
    claims = extract_claims(text)
    graph = CausalGraph.from_claims([c for c in claims if not c.negated])
    return build_features(text, claims, graph, find_myths(claims))


class CausalNewsAnalyzer:
    """Analyse a news text and explain the result."""

    def __init__(self, model=None) -> None:
        self.model = model if model is not None else load_or_train()

    def analyze(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        claims = extract_claims(text)
        graph = CausalGraph.from_claims([c for c in claims if not c.negated])
        myths = find_myths(claims)
        features = build_features(text, claims, graph, myths)

        probability = float(
            self.model.predict_proba(vectorise([features]))[0][1])
        if myths:
            probability = max(probability, MYTH_FLOOR)

        if probability >= FAKE_THRESHOLD:
            verdict = "likely_fake"
        elif probability <= RELIABLE_THRESHOLD:
            verdict = "likely_reliable"
        else:
            verdict = "uncertain"

        return {
            "verdict": verdict,
            "fake_probability": round(probability, 4),
            "claims": [c.to_dict() for c in claims],
            "graph": graph.to_dict(),
            "myth_matches": [m.to_dict() for m in myths],
            "features": {k: round(features[k], 4) for k in FEATURE_NAMES},
            "red_flags": explain(features, claims, myths, graph),
        }
