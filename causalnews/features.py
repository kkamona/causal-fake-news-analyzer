"""Turn a text, its claims and its causal graph into a numeric feature vector."""

from __future__ import annotations

import re

from .extractor import CausalClaim
from .graph import CausalGraph, normalize_concept
from .knowledge import MythMatch
from .lexicons import COMMON_ACRONYMS, EVIDENCE_RE, SENSATIONAL_RE
from .preprocess import split_sentences, tokenize

FEATURE_NAMES = [
    "causal_density",
    "absolute_ratio",
    "hedged_ratio",
    "unattributed_ratio",
    "vague_ratio",
    "sensational_density",
    "evidence_density",
    "longest_chain",
    "has_cycle",
    "max_out_degree",
    "myth_hits",
    "correlation_ratio",
    "correlation_leap",
    "exclamation_density",
    "caps_ratio",
]

_CAPS_RE = re.compile(r"\b[A-Z]{4,}\b")


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def correlation_leaps(claims: list[CausalClaim]) -> int:
    """Count pairs where a correlation is later restated as causation.

    "A is linked to B" followed by "A causes B" (or the reverse) is the
    classic correlation-to-causation fallacy.
    """
    corr = {(normalize_concept(c.cause), normalize_concept(c.effect))
            for c in claims if c.kind == "correlational"}
    leaps = 0
    for claim in claims:
        if claim.kind != "causal" or claim.negated or claim.certainty == "hedged":
            continue
        pair = (normalize_concept(claim.cause), normalize_concept(claim.effect))
        if pair in corr or pair[::-1] in corr:
            leaps += 1
    return leaps


def build_features(text: str, claims: list[CausalClaim], graph: CausalGraph,
                   myths: list[MythMatch]) -> dict[str, float]:
    sentences = split_sentences(text)
    n_sent = max(1, len(sentences))
    n_tokens = max(1, len(tokenize(text)))

    asserted = [c for c in claims if c.kind == "causal" and not c.negated]
    correlational = [c for c in claims if c.kind == "correlational"]
    n = len(asserted)

    caps = [w for w in _CAPS_RE.findall(text) if w not in COMMON_ACRONYMS]
    return {
        "causal_density": n / n_sent,
        "absolute_ratio": _ratio(sum(c.certainty == "absolute" for c in asserted), n),
        "hedged_ratio": _ratio(sum(c.certainty == "hedged" for c in asserted), n),
        "unattributed_ratio": _ratio(sum(not c.attributed for c in asserted), n),
        "vague_ratio": _ratio(sum(c.vague_attribution for c in asserted), n),
        "sensational_density": len(SENSATIONAL_RE.findall(text)) / n_sent,
        "evidence_density": len(EVIDENCE_RE.findall(text)) / n_sent,
        "longest_chain": float(graph.longest_chain()),
        "has_cycle": 1.0 if graph.has_cycle() else 0.0,
        "max_out_degree": float(graph.max_out_degree()),
        "myth_hits": float(len(myths)),
        "correlation_ratio": _ratio(len(correlational), n + len(correlational)),
        "correlation_leap": float(min(1, correlation_leaps(claims))),
        "exclamation_density": text.count("!") / n_sent,
        "caps_ratio": len(caps) / n_tokens,
    }


def explain(features: dict[str, float], claims: list[CausalClaim],
            myths: list[MythMatch], graph: CausalGraph) -> list[str]:
    """Human-readable red flags (and reassuring signs) for the analyst."""
    flags: list[str] = []
    asserted = [c for c in claims if c.kind == "causal" and not c.negated]
    for myth in myths:
        flags.append(f"Known debunked claim: {myth.description}")
    if features["absolute_ratio"] > 0:
        flags.append("Causal claims are stated with absolute certainty "
                     f"({features['absolute_ratio']:.0%} of causal claims).")
    if asserted and features["unattributed_ratio"] >= 0.5:
        flags.append("Most causal claims cite no named, checkable source "
                     f"({features['unattributed_ratio']:.0%}).")
    if features["vague_ratio"] > 0:
        flags.append("Vague attribution is used ('experts say', 'insiders', "
                     "'leaked report').")
    if features["has_cycle"]:
        flags.append("The causal graph contains a cycle (circular reasoning).")
    for hub in graph.hubs():
        flags.append(f"'{hub}' is blamed for three or more different effects "
                     "(monocausal explanation).")
    if features["correlation_leap"]:
        flags.append("A correlation is restated as a causal relationship.")
    if features["sensational_density"] >= 1.0:
        flags.append("Emotionally charged or sensational language dominates.")
    if features["caps_ratio"] > 0.03 or features["exclamation_density"] >= 0.5:
        flags.append("Shouting style: capital letters or exclamation marks.")
    if asserted and features["hedged_ratio"] >= 0.5:
        flags.append("Reassuring: causal claims are hedged appropriately.")
    if asserted and features["unattributed_ratio"] == 0:
        flags.append("Reassuring: every causal claim has a named source.")
    if not asserted:
        flags.append("No explicit causal claims were found in the text.")
    return flags
