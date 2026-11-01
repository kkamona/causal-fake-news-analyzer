"""Lexicons used by the extractor and the feature builder.

All patterns are case-insensitive unless noted otherwise.
"""

import re

# Words that weaken a claim ("may cause", "could lead to").
HEDGE_RE = re.compile(
    r"\b(?:may|might|could|can|possibly|perhaps|probably|likely|unlikely|"
    r"suggests?|suggested|appears?|seems?|potentially|tends?|partly|partially|"
    r"reportedly|allegedly|expected|estimated|should)\b",
    re.IGNORECASE,
)

# Words that express unjustified certainty.
ABSOLUTE_RE = re.compile(
    r"\b(?:always|never|definitely|certainly|undoubtedly|inevitably|guaranteed|"
    r"guarantee|proves?|proven|proof|obviously|absolutely|completely|entirely|"
    r"instantly|every single|everyone|nobody|no one|without (?:a|any) doubt|"
    r"the only (?:reason|cause)|single cause|nothing else)\b|100\s?%",
    re.IGNORECASE,
)

# Markers of verifiable evidence (sources, studies, figures).
EVIDENCE_RE = re.compile(
    r"\b(?:according to|study|studies|researchers?|survey|published|journal|"
    r"university|ministry|agency|statistics|analysis|experiment|trial|"
    r"peer-reviewed|observatory|institute|found that|reported|percent|per cent)\b"
    r"|\d+(?:\.\d+)?\s?%",
    re.IGNORECASE,
)

# Attribution to a named, checkable source (case-sensitive on the name part).
SPECIFIC_ATTRIBUTION_RE = re.compile(
    r"(?:[Aa]ccording to|[Rr]eported by|[Pp]ublished in|[Ff]ound by|"
    r"[Ss]tated by|[Cc]onfirmed by|[Rr]esearchers (?:at|from)|"
    r"[Ss]cientists at|[Aa]nalysis in|[Ss]urvey (?:of|by))\s+"
    r"(?:the\s+|a\s+|an\s+)?[A-Z0-9][\w&.\-]*"
)

# Vague, uncheckable attribution typical for misinformation.
VAGUE_ATTRIBUTION_RE = re.compile(
    r"\b(?:(?:some|many|several|most)\s+)?"
    r"(?:experts|scientists|doctors|sources|insiders|people|they|studies)\s+"
    r"(?:say|said|claim|claimed|show|showed|prove|proved|confirm|confirmed|warn|"
    r"admit|admitted)\b"
    r"|\baccording to (?:some|many|anonymous|unnamed|experts|sources|insiders|"
    r"a source|experts who)\b|\bwho cannot be named\b|\bleaked\b|\bhidden study\b"
    r"|\bsecret study\b",
    re.IGNORECASE,
)

# Emotive / sensational vocabulary.
SENSATIONAL_RE = re.compile(
    r"\b(?:shocking|bombshell|exposed?|secret|secretly|miracle|banned?|"
    r"cover-?up|covering it up|hoax|outrage|terrifying|deadly|wake up|"
    r"mainstream media|elites?|globalists?|urgent|unbelievable|stunning|"
    r"stunned|horrifying|evil|scam|conspiracy|brainwash\w*|mind control|"
    r"poison\w*|lies|lying|truth|big pharma|furious|sheeple|breaking|"
    r"share (?:this|now|it)|before it (?:is|gets) (?:deleted|banned)|"
    r"don'?t want you to know|do not want you to know|hide|hides|hiding|"
    r"hidden|silenced)\b",
    re.IGNORECASE,
)

# Phrases denying a causal link ("no evidence that X causes Y").
NEGATION_RE = re.compile(
    r"\bno (?:scientific )?(?:evidence|proof|link|sign)\b"
    r"|\bnot (?:cause|caused|lead|led|result|trigger)"
    r"|\b(?:does|do|did|can|could|will|would|has|have|had|is|are|was|were)"
    r"(?:n't| not)\s+(?:\w+\s+)?(?:cause|lead|result|trigger|spread|prevent|"
    r"cure|kill|reduce)"
    r"|\bmyths?\b|\bdebunk\w*|\bdisprov\w*|\bunfounded\b|\bbaseless\b"
    r"|\bfalse claims?\b",
    re.IGNORECASE,
)

# Cue words for correlation (not causation).
CORRELATION_CUES = (
    r"linked\s+to|associated\s+with|correlated\s+with|tied\s+to|"
    r"connected\s+to|related\s+to"
)

# Acronyms that are normal in news and must not count as "shouting".
COMMON_ACRONYMS = frozenset(
    {"COVID", "NASA", "NATO", "OPEC", "UNICEF", "UNESCO", "FIFA", "UEFA",
     "AIDS", "OECD", "ASEAN", "WHO", "IEA"}
)
