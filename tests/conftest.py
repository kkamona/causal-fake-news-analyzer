import pytest

from causalnews.analyzer import CausalNewsAnalyzer


@pytest.fixture(scope="session")
def analyzer(tmp_path_factory):
    """An analyzer with a model trained on the bundled corpus."""
    from causalnews.model import train

    path = tmp_path_factory.mktemp("model") / "model.joblib"
    return CausalNewsAnalyzer(model=train(path))
