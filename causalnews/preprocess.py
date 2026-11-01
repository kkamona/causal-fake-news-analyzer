"""Text pre-processing: sentence splitting, tokenisation, clause cleaning."""

import re

_ABBREVIATIONS = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "U.S.", "U.K.", "e.g.",
                  "i.e.", "vs.", "St.", "No."]
_DOT = "\u2024"  # one-dot leader, used as a placeholder for protected dots
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9])")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")

_REPORTING_FRAME = re.compile(
    r"^(?:\w+(?:[\s,'’-]+\w+){0,5}?)\s+"
    r"(?:said|says|say|claimed|claims|claim|confirmed|confirm|confirms|found|"
    r"reported|reports|noted|stated|warned|warn|warns|suggest|suggests|"
    r"suggested|showed|shows|show|prove|proves|proved|observed)\s+"
    r"(?:that\s+)?",
    re.IGNORECASE,
)
_EXISTENTIAL_FRAME = re.compile(
    r"^(?:there\s+(?:is|are|was|were)\s+(?:no|little|some|any)\s+"
    r"(?:\w+\s+)?(?:evidence|proof|sign|signs)\s+(?:that\s+)?)",
    re.IGNORECASE,
)
_TRAILING_ATTRIBUTION = re.compile(
    r",\s*(?:[\w\s'’-]{1,60}?)\s+(?:said|reported|says|noted|stated|added)\b.*$",
    re.IGNORECASE,
)
_TRAILING_ACCORDING = re.compile(r"(?<=\w)[,\s]+according to\b.*$", re.IGNORECASE)
_LEADING_ACCORDING = re.compile(r"^according to [^,]+,\s*", re.IGNORECASE)
_TRAILING_RELATIVE = re.compile(r"[,\s]+(?:which|that|who|whom)$", re.IGNORECASE)
_LEADING_JUNK = re.compile(r"^(?:that|and|but|then|so|also)\s+", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences (abbreviation-aware)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", _DOT))
    parts = _SENT_SPLIT.split(text)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens."""
    return _TOKEN_RE.findall((text or "").lower())


def clean_clause(text: str) -> str:
    """Remove framing words, attributions and punctuation around a clause."""
    text = (text or "").strip()
    text = text.strip(" \t\"“”'’.,;:!?-–—")
    text = _TRAILING_ATTRIBUTION.sub("", text)
    text = _TRAILING_ACCORDING.sub("", text)
    previous = None
    while previous != text:
        previous = text
        text = _LEADING_ACCORDING.sub("", text)
        text = _EXISTENTIAL_FRAME.sub("", text)
        text = _REPORTING_FRAME.sub("", text)
        text = _TRAILING_RELATIVE.sub("", text)
        text = _LEADING_JUNK.sub("", text)
        text = text.strip(" \t\"“”'’.,;:!?-–—")
    return re.sub(r"\s+", " ", text)
