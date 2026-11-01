# Causal Fake News Analyzer

An intelligent system that analyses news texts for signs of misinformation by
**identifying cause-and-effect relationships**. The project accompanies
Assignment 5 (DevOps and CI/CD for research projects).

![CI](https://github.com/YOUR_USERNAME/causal-fake-news-analyzer/actions/workflows/ci.yml/badge.svg)

## How it works

1. **Extraction** (`extractor.py`): rule-based patterns find causal claims
   ("X causes Y", "Y because of X", "if X then Y", "As a result, ...") and
   correlational claims ("X is linked to Y"). Each claim is tagged as hedged,
   neutral or absolute, attributed to a named source or not, and negated or not.
2. **Causal graph** (`graph.py`): claims become a directed graph. Chains, cycles
   (circular reasoning) and "hubs" (one cause blamed for everything) are detected.
3. **Knowledge base** (`knowledge.py`): claims that match well-known debunked
   causal myths (e.g. "5G causes COVID-19") are flagged; denials are ignored.
4. **Features and classifier** (`features.py`, `model.py`): 15 numeric features
   feed a logistic-regression model.
5. **Explanation** (`analyzer.py`): the verdict comes with human-readable red flags.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
python -m causalnews evaluate
python -m causalnews analyze --text "Vaccines definitely cause autism. Share this now!"
python -m causalnews serve            # API on http://127.0.0.1:8000/docs
```

## Docker

```bash
docker build -t causalnews .
docker run -p 8000:8000 causalnews
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
     -d '{"text": "5G towers cause coronavirus and the media is hiding it!"}'
```

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | What it does |
|-----|--------------|
| `test` | ruff lint, pytest with coverage (Python 3.11 and 3.12), model cross-validation |
| `docker` | builds the image, smoke-tests the running container, optionally pushes to Docker Hub |
| `deploy-minikube` | (main branch) starts Minikube, deploys `k8s/` manifests, smoke-tests the service |

To enable the optional Docker Hub push, add repository secrets `DOCKERHUB_USERNAME`
and `DOCKERHUB_TOKEN` and a repository variable `PUSH_TO_DOCKERHUB=true`.
A `Jenkinsfile` with the equivalent pipeline is included for Jenkins users.

## Data and limitations

`causalnews/data/sample_articles.csv` contains 48 short articles (24 fake, 24 real)
**written for this assignment** to demonstrate the pipeline. Cross-validation
scores on such a small, hand-made corpus are optimistic and must not be read as
real-world accuracy. The extractor is pattern-based and misses implicit
causality. For serious use, evaluate on public datasets (e.g. LIAR, FakeNewsNet)
and combine the system with human fact-checking.
