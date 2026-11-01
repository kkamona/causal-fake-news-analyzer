"""A small dependency-free causal graph built from extracted claims."""

from __future__ import annotations

import re
from collections import defaultdict

from .extractor import CausalClaim
from .preprocess import tokenize

_STOP_LEADING = {
    "the", "a", "an", "this", "that", "these", "those", "their", "his", "her",
    "its", "our", "your", "my", "some", "any", "all", "every", "of", "to", "in",
    "is", "are", "was", "were", "be", "been", "it", "there", "by", "for", "with",
}
_MAX_TOKENS = 6
_MERGE_THRESHOLD = 0.6


def _stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def normalize_concept(text: str) -> str:
    """Reduce a phrase to a canonical key used to identify graph nodes."""
    head = re.split(r"[;,]", text or "")[0]
    tokens = tokenize(head)
    while tokens and tokens[0] in _STOP_LEADING:
        tokens.pop(0)
    return " ".join(_stem(t) for t in tokens[:_MAX_TOKENS])


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class CausalGraph:
    """Directed graph whose nodes are concepts and edges are causal claims."""

    def __init__(self) -> None:
        self.labels: dict[str, str] = {}
        self.edges: dict[tuple[str, str], list[CausalClaim]] = defaultdict(list)

    # ------------------------------------------------------------------ build
    def _node(self, phrase: str) -> str | None:
        key = normalize_concept(phrase)
        if not key:
            return None
        if key in self.labels:
            return key
        tokens = set(key.split())
        for existing in self.labels:
            if _jaccard(tokens, set(existing.split())) >= _MERGE_THRESHOLD:
                return existing
        self.labels[key] = re.split(r"[;,]", phrase)[0].strip()
        return key

    def add_claim(self, claim: CausalClaim) -> None:
        source, target = self._node(claim.cause), self._node(claim.effect)
        if source and target and source != target:
            self.edges[(source, target)].append(claim)

    @classmethod
    def from_claims(cls, claims: list[CausalClaim]) -> CausalGraph:
        graph = cls()
        for claim in claims:
            graph.add_claim(claim)
        return graph

    # ---------------------------------------------------------------- queries
    def successors(self, key: str) -> list[str]:
        return [t for (s, t) in self.edges if s == key]

    def predecessors(self, key: str) -> list[str]:
        return [s for (s, t) in self.edges if t == key]

    @property
    def nodes(self) -> list[str]:
        used = {k for edge in self.edges for k in edge}
        return [k for k in self.labels if k in used]

    def has_cycle(self) -> bool:
        """True if the graph contains a directed cycle (circular reasoning)."""
        state: dict[str, int] = {}  # 1 = on stack, 2 = finished

        def visit(node: str) -> bool:
            state[node] = 1
            for nxt in self.successors(node):
                if state.get(nxt) == 1:
                    return True
                if nxt not in state and visit(nxt):
                    return True
            state[node] = 2
            return False

        return any(n not in state and visit(n) for n in self.nodes)

    def longest_chain(self) -> int:
        """Number of edges in the longest causal chain (cycles are ignored)."""
        memo: dict[str, int] = {}

        def depth(node: str, stack: frozenset[str]) -> int:
            if node in memo:
                return memo[node]
            best = 0
            for nxt in self.successors(node):
                if nxt not in stack:
                    best = max(best, 1 + depth(nxt, stack | {node}))
            memo[node] = best
            return best

        return max((depth(n, frozenset()) for n in self.nodes), default=0)

    def max_out_degree(self) -> int:
        counts: dict[str, int] = defaultdict(int)
        for source, _ in self.edges:
            counts[source] += 1
        return max(counts.values(), default=0)

    def hubs(self, minimum: int = 3) -> list[str]:
        """Concepts blamed for many different effects (monocausal thinking)."""
        counts: dict[str, int] = defaultdict(int)
        for source, _ in self.edges:
            counts[source] += 1
        return [self.labels[k] for k, c in counts.items() if c >= minimum]

    def to_dict(self) -> dict:
        return {
            "nodes": [self.labels[k] for k in self.nodes],
            "edges": [
                {"source": self.labels[s], "target": self.labels[t],
                 "claims": len(claims)}
                for (s, t), claims in self.edges.items()
            ],
            "longest_chain": self.longest_chain(),
            "has_cycle": self.has_cycle(),
            "hubs": self.hubs(),
        }
