"""A small knowledge base of causal claims rejected by scientific consensus.

Each entry describes a cause/effect pair that health agencies or systematic
reviews have repeatedly refuted. The knowledge base is deliberately small and
transparent: it is meant to show how background knowledge can be combined with
text analysis, not to replace professional fact-checking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extractor import CausalClaim

_REMEDY_CUES = frozenset({"cure", "cures", "cured", "heal", "heals", "kill", "kills",
                          "prevent", "prevents", "eliminate", "eliminates",
                          "destroy", "destroys"})


@dataclass(frozen=True)
class KnownMyth:
    myth_id: str
    cause: re.Pattern
    effect: re.Pattern | None
    description: str
    remedy_cue_only: bool = False


def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


_VIRUS = r"\b(?:covid\S*|coronavirus|corona virus|sars-cov-2|virus)\b"

MYTHS: list[KnownMyth] = [
    KnownMyth("5g-covid", _p(r"\b5g\b"), _p(_VIRUS + r"|\bpandemic\b"),
              "5G radio networks do not cause or spread COVID-19 (WHO)."),
    KnownMyth("vaccine-autism", _p(r"\bvaccin\w*|\bmmr\b"), _p(r"\bautis\w*"),
              "Large studies found no link between vaccines and autism."),
    KnownMyth("vaccine-microchip", _p(r"\bvaccin\w*"),
              _p(r"\b(?:microchip\w*|magnet\w*|tracking device\w*|nanochip\w*)"),
              "Vaccines do not contain microchips or magnetic tracking devices."),
    KnownMyth("vaccine-dna", _p(r"\bvaccin\w*"),
              _p(r"\b(?:infertil\w*|sterili[sz]\w*|dna|genome)\b"),
              "Authorised vaccines do not alter DNA or cause infertility."),
    KnownMyth("bleach-cure", _p(r"\b(?:bleach|disinfectant|chlorine dioxide)\b"),
              _p(_VIRUS + r"|\bcancer\b|\bautis\w*"),
              "Drinking bleach or disinfectants is dangerous and cures nothing.",
              remedy_cue_only=True),
    KnownMyth("food-cure", _p(r"\b(?:garlic|lemons?(?: juice)?|baking soda|"
                              r"onions?|ginger|hot water|colloidal silver|"
                              r"essential oils?)\b"),
              _p(_VIRUS + r"|\bcancer\b|\bdiabetes\b"),
              "No household food or remedy cures cancer, diabetes or COVID-19.",
              remedy_cue_only=True),
    KnownMyth("heat-covid", _p(r"\b(?:hot|warm) weather\b|\bsummer heat\b|"
                               r"\bhigh temperatures?\b|\bsunshine\b"),
              _p(_VIRUS), "Hot weather does not eliminate the coronavirus.",
              remedy_cue_only=True),
    KnownMyth("chemtrails", _p(r"\bchemtrails?\b"), None,
              "'Chemtrails' are condensation trails; no mind-control or "
              "poisoning programme has been shown."),
    KnownMyth("microwave-cancer", _p(r"\bmicrowave (?:ovens?)?\b"),
              _p(r"\bcancer\b"),
              "Microwave ovens do not cause cancer when used as intended."),
    KnownMyth("gmo-cancer", _p(r"\bgmos?\b|genetically modified"),
              _p(r"\bcancer\b"),
              "Major reviews found no evidence that approved GM foods cause "
              "cancer."),
]


@dataclass(frozen=True)
class MythMatch:
    myth_id: str
    description: str
    claim: CausalClaim

    def to_dict(self) -> dict:
        return {"id": self.myth_id, "description": self.description,
                "cause": self.claim.cause, "effect": self.claim.effect,
                "sentence": self.claim.sentence}


def find_myths(claims: list[CausalClaim]) -> list[MythMatch]:
    """Return claims that assert a known debunked causal relationship.

    Negated claims ("there is no evidence that 5G causes COVID-19") are
    ignored, because they correctly *deny* the myth.
    """
    found: list[MythMatch] = []
    seen: set[tuple[str, int]] = set()
    for claim in claims:
        if claim.kind != "causal" or claim.negated:
            continue
        cue_words = set(claim.cue.split())
        for myth in MYTHS:
            if myth.remedy_cue_only and not (cue_words & _REMEDY_CUES):
                continue
            if not myth.cause.search(claim.cause):
                continue
            if myth.effect is not None and not myth.effect.search(claim.effect):
                continue
            key = (myth.myth_id, claim.sentence_index)
            if key not in seen:
                seen.add(key)
                found.append(MythMatch(myth.myth_id, myth.description, claim))
    return found
