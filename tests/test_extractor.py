import pytest

from causalnews.extractor import extract_claims


def one(text):
    claims = extract_claims(text)
    assert len(claims) >= 1, text
    return claims[0]


@pytest.mark.parametrize("text, cause, effect", [
    ("Heavy rain caused the river to flood.", "rain", "flood"),
    ("The flooding was caused by heavy rainfall.", "rainfall", "flooding"),
    ("Prices rose because demand increased.", "demand", "prices"),
    ("Because of a fault, trains were delayed.", "fault", "delayed"),
    ("If the policy continues, unemployment will rise.", "policy", "unemployment"),
    ("The new drug reduced hospital stays by 15 percent.", "drug", "hospital"),
])
def test_basic_patterns(text, cause, effect):
    claim = one(text)
    assert cause in claim.cause.lower()
    assert effect in claim.effect.lower()
    assert claim.kind == "causal"


def test_intersentential_connective():
    claims = extract_claims(
        "The factory closed. As a result, hundreds of workers lost their jobs.")
    assert len(claims) == 1
    assert claims[0].pattern == "intersentential"
    assert "factory" in claims[0].cause
    assert "workers" in claims[0].effect


def test_anaphoric_this():
    claims = extract_claims("Smoking harms lungs. This leads to higher costs.")
    patterns = [c.pattern for c in claims]
    assert "intersentential" in patterns


def test_chain_is_split_into_two_claims():
    claims = extract_claims("Inflation causes unemployment, which leads to crime.")
    pairs = [(c.cause, c.effect) for c in claims]
    assert ("Inflation", "unemployment") in pairs
    assert ("unemployment", "crime") in pairs


def test_correlation_is_not_causation():
    claim = one("Coffee consumption is linked to longer life.")
    assert claim.kind == "correlational"


def test_certainty_levels():
    assert one("Smoking may cause cancer.").certainty == "hedged"
    assert one("Smoking definitely causes cancer.").certainty == "absolute"
    assert one("Smoking causes cancer.").certainty == "neutral"


def test_attribution():
    attributed = one("According to the World Health Organization, smoking "
                     "causes cancer.")
    assert attributed.attributed
    vague = one("Experts say smoking causes cancer.")
    assert vague.vague_attribution and not vague.attributed


def test_negation_is_flagged():
    claim = one("There is no evidence that 5G causes COVID-19.")
    assert claim.negated


def test_no_claims_in_plain_text():
    assert extract_claims("The museum opens on Monday. Tickets cost 8 dollars.") == []
