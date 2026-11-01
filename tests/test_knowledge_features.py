from causalnews.extractor import extract_claims
from causalnews.features import (
    FEATURE_NAMES,
    build_features,
    correlation_leaps,
    explain,
)
from causalnews.graph import CausalGraph
from causalnews.knowledge import find_myths


def test_myth_detected():
    myths = find_myths(extract_claims("5G towers cause coronavirus."))
    assert [m.myth_id for m in myths] == ["5g-covid"]


def test_negated_myth_is_not_flagged():
    claims = extract_claims("There is no evidence that 5G causes COVID-19.")
    assert find_myths(claims) == []


def test_remedy_myth_needs_remedy_verb():
    assert find_myths(extract_claims("Garlic cures cancer."))
    assert not find_myths(extract_claims("Garlic causes bad breath."))


def test_correlation_leap():
    text = "Screen time is linked to obesity. Screen time causes obesity."
    assert correlation_leaps(extract_claims(text)) == 1


def test_features_complete_and_bounded():
    text = "SHOCKING! Sugar always causes cancer, experts say. Wake up!"
    claims = extract_claims(text)
    graph = CausalGraph.from_claims(claims)
    feats = build_features(text, claims, graph, find_myths(claims))
    assert list(feats) == FEATURE_NAMES
    assert feats["absolute_ratio"] == 1.0
    assert feats["exclamation_density"] > 0
    assert all(v >= 0 for v in feats.values())


def test_features_without_claims_are_zero_for_ratios():
    feats = build_features("The museum opens.", [], CausalGraph(), [])
    assert feats["causal_density"] == 0 and feats["unattributed_ratio"] == 0


def test_explain_mentions_myth():
    claims = extract_claims("Vaccines cause autism.")
    graph = CausalGraph.from_claims(claims)
    myths = find_myths(claims)
    feats = build_features("Vaccines cause autism.", claims, graph, myths)
    assert any("debunked" in f for f in explain(feats, claims, myths, graph))
