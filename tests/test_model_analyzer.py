import pytest

from causalnews.model import cross_validate, load_dataset, load_or_train, train


def test_dataset_is_balanced():
    texts, labels = load_dataset()
    assert len(texts) == len(labels) == 48
    assert sum(labels) == 24


def test_cross_validation_quality():
    metrics = cross_validate()
    assert metrics["accuracy"] >= 0.85
    assert metrics["recall"] >= 0.80


def test_model_roundtrip(tmp_path):
    path = tmp_path / "m.joblib"
    train(path)
    assert path.exists()
    assert load_or_train(path) is not None


def test_fake_text_scores_higher_than_real(analyzer):
    fake = analyzer.analyze(
        "Tap water is secretly causing diabetes, and doctors hide this because "
        "they profit from medicine. Experts say it always leads to blindness. "
        "Share this now!")
    real = analyzer.analyze(
        "The ministry of transport said road accidents fell by 12 percent, "
        "according to figures published on Monday. Heavy snow caused delays on "
        "two highways, the agency reported.")
    assert fake["verdict"] == "likely_fake"
    assert real["verdict"] == "likely_reliable"
    assert fake["fake_probability"] > real["fake_probability"]


def test_myth_raises_probability(analyzer):
    result = analyzer.analyze("Vaccines cause autism in children.")
    assert result["fake_probability"] >= 0.9
    assert result["myth_matches"]


def test_debunking_article_is_not_flagged(analyzer):
    result = analyzer.analyze(
        "The World Health Organization said there is no evidence that 5G "
        "networks cause COVID-19. The virus spreads through droplets.")
    assert not result["myth_matches"]
    assert result["verdict"] != "likely_fake"


def test_result_structure(analyzer):
    result = analyzer.analyze("Rain causes floods, according to the agency.")
    assert {"verdict", "fake_probability", "claims", "graph", "features",
            "myth_matches", "red_flags"} <= set(result)
    assert 0.0 <= result["fake_probability"] <= 1.0


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_text_rejected(analyzer, bad):
    with pytest.raises(ValueError):
        analyzer.analyze(bad)
