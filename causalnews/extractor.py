"""Rule-based extraction of cause-and-effect (and correlation) claims.

The extractor recognises five families of linguistic patterns:

* ``because`` / ``due to`` constructions        (effect, cause order)
* passive constructions ("is caused by")        (effect, cause order)
* active causal verbs ("causes", "leads to")    (cause, effect order)
* conditionals ("if A, then B") and "A, so B"
* inter-sentential connectives ("As a result, ...", "This leads to ...")

Nested clauses ("A causes B, which leads to C") are resolved recursively so
that chains of causes become several linked claims.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .lexicons import (
    ABSOLUTE_RE,
    CORRELATION_CUES,
    HEDGE_RE,
    NEGATION_RE,
    SPECIFIC_ATTRIBUTION_RE,
    VAGUE_ATTRIBUTION_RE,
)
from .preprocess import clean_clause, split_sentences

_ACTIVE_VERBS = (
    r"caus(?:e|es|ed|ing)|lead(?:s|ing)?\s+to|led\s+to|result(?:s|ed|ing)?\s+in|"
    r"trigger(?:s|ed|ing)?|provok(?:e|es|ed|ing)|induc(?:e|es|ed|ing)|"
    r"bring(?:s|ing)?\s+about|brought\s+about|gives?\s+rise\s+to|gave\s+rise\s+to|"
    r"contribut(?:e|es|ed|ing)\s+to|(?:is|are|was|were)\s+responsible\s+for|"
    r"cures?|cured|heals?|healed|prevent(?:s|ed|ing)?|kills?|killed|"
    r"destroy(?:s|ed|ing)?|eliminat(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|"
    r"lower(?:s|ed|ing)?|weaken(?:s|ed|ing)?|worsen(?:s|ed|ing)?|"
    r"improv(?:e|es|ed|ing)|boost(?:s|ed|ing)?|harm(?:s|ed|ing)?"
)
_MODALS = (
    r"(?:(?:can|could|may|might|will|would|shall|should|must|does|do|did|has|"
    r"have|had|also|often|always|never|definitely|really|directly)\s+)*"
)
_PASSIVE_CUES = (
    r"(?:(?:is|are|was|were|been|being|be)\s+)?"
    r"(?:caused\s+by|triggered\s+by|provoked\s+by|driven\s+by|"
    r"brought\s+about\s+by|the\s+result\s+of|a\s+result\s+of|"
    r"the\s+consequence\s+of|a\s+consequence\s+of|results?\s+from|"
    r"resulted\s+from|stems?\s+from|stemmed\s+from|arises?\s+from|"
    r"due\s+to|owing\s+to|because\s+of|because|thanks\s+to|"
    r"on\s+account\s+of|as\s+a\s+result\s+of|as\s+a\s+consequence\s+of)"
)

_INITIAL_BECAUSE = re.compile(
    r"^(?P<cue>because\s+of|because|due\s+to|owing\s+to|thanks\s+to)\s+"
    r"(?P<cause>.+?),\s*(?P<effect>.+)$",
    re.IGNORECASE,
)
_PASSIVE = re.compile(
    rf"^(?P<effect>.+?),?\s+(?P<cue>{_PASSIVE_CUES})\s+(?P<cause>.+)$",
    re.IGNORECASE,
)
_ACTIVE = re.compile(
    rf"^(?P<cause>.+?)\s+(?P<cue>{_MODALS}(?:{_ACTIVE_VERBS})\b)(?!\s+by\b)\s+"
    r"(?P<effect>.+)$",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r"^(?P<cue>if)\s+(?P<cause>.+?),\s*(?:then\s+)?(?P<effect>.+)$", re.IGNORECASE
)
_SO = re.compile(r"^(?P<cause>.+?),\s+(?P<cue>so)\s+(?P<effect>.+)$", re.IGNORECASE)
_CONNECTIVE = re.compile(
    r"^(?P<cue>as a result|as a consequence|consequently|therefore|thus|hence|"
    r"for this reason|because of this|that is why|that's why|this is why|"
    r"which is why|accordingly)\b[,:]?\s*(?P<effect>.+)$",
    re.IGNORECASE,
)
_CORRELATION = re.compile(
    rf"^(?P<cause>.+?)\s+(?:(?:is|are|was|were|has been|have been)\s+)?"
    rf"(?P<cue>{CORRELATION_CUES})\s+(?P<effect>.+)$",
    re.IGNORECASE,
)

# Ordered by precedence; each entry is (pattern, pattern-name, kind, left-role).
# ``left-role`` says which role is written first in the sentence.
_PATTERNS = [
    (_INITIAL_BECAUSE, "explicit", "causal", "cause"),
    (_CONDITIONAL, "conditional", "causal", "cause"),
    (_PASSIVE, "passive", "causal", "effect"),
    (_ACTIVE, "explicit", "causal", "cause"),
    (_SO, "explicit", "causal", "cause"),
    (_CORRELATION, "correlation", "correlational", "cause"),
]
_PRONOUNS = {"this", "that", "it", "these", "those", "which"}


@dataclass(frozen=True)
class CausalClaim:
    """One extracted relation between two concepts."""

    cause: str
    effect: str
    cue: str
    pattern: str  # explicit | passive | conditional | intersentential | correlation
    kind: str  # causal | correlational
    sentence_index: int
    sentence: str
    certainty: str  # hedged | neutral | absolute
    attributed: bool
    vague_attribution: bool
    negated: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Match:
    cause: str
    effect: str
    cue: str
    pattern: str
    kind: str
    left: str  # "cause" or "effect": which argument is written first


def _try_patterns(text: str) -> _Match | None:
    for regex, pattern, kind, left in _PATTERNS:
        m = regex.match(text)
        if not m:
            continue
        cause = clean_clause(m.group("cause"))
        effect = clean_clause(m.group("effect"))
        if cause and effect:
            return _Match(cause, effect, m.group("cue").strip().lower(),
                          pattern, kind, left)
    return None


def _resolve(text: str, depth: int = 0) -> list[_Match]:
    """Find a relation in ``text`` and, recursively, in its clauses."""
    match = _try_patterns(text)
    if match is None:
        return []
    results: list[_Match] = []
    if depth < 2:
        for role in ("cause", "effect"):
            clause = getattr(match, role)
            nested = _resolve(clause, depth + 1)
            if nested:
                results.extend(nested)
                first = nested[0]
                # Replace the clause by the argument written first in it.
                setattr(match, role, first.cause if first.left == "cause"
                        else first.effect)
    results.insert(0, match)
    return results


def _certainty(sentence: str) -> str:
    if ABSOLUTE_RE.search(sentence):
        return "absolute"
    if HEDGE_RE.search(sentence):
        return "hedged"
    return "neutral"


def extract_claims(text: str) -> list[CausalClaim]:
    """Extract all cause-effect and correlation claims from ``text``."""
    sentences = split_sentences(text)
    claims: list[CausalClaim] = []
    previous_summary = ""  # effect (or text) of the previous sentence
    previous_sentence = ""

    for index, sentence in enumerate(sentences):
        window = f"{previous_sentence} {sentence}"
        attributed = bool(SPECIFIC_ATTRIBUTION_RE.search(window))
        vague = bool(VAGUE_ATTRIBUTION_RE.search(window)) and not attributed
        negated = bool(NEGATION_RE.search(sentence))
        certainty = _certainty(sentence)
        stripped = sentence.rstrip(".!?\"” ")

        matches: list[_Match] = []
        connective = _CONNECTIVE.match(stripped)
        if connective and previous_summary:
            effect = clean_clause(connective.group("effect"))
            if effect:
                matches.append(_Match(previous_summary, effect,
                                      connective.group("cue").lower(),
                                      "intersentential", "causal", "cause"))
        else:
            matches = _resolve(stripped)

        sentence_summary = clean_clause(stripped)
        for match in matches:
            if match.cause.lower() in _PRONOUNS and previous_summary:
                match.cause = previous_summary
                match.pattern = "intersentential"
            claims.append(CausalClaim(
                cause=match.cause, effect=match.effect, cue=match.cue,
                pattern=match.pattern, kind=match.kind,
                sentence_index=index, sentence=sentence, certainty=certainty,
                attributed=attributed, vague_attribution=vague,
                negated=negated,
            ))
        if matches:
            sentence_summary = matches[-1].effect if matches[0].left == "cause" \
                else matches[0].effect
        previous_summary = sentence_summary
        previous_sentence = sentence
    return claims
