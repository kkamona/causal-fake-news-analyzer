"""REST API (FastAPI) exposing the analyzer."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .analyzer import CausalNewsAnalyzer

MAX_TEXT_LENGTH = 20_000

app = FastAPI(
    title="Causal Fake News Analyzer",
    version=__version__,
    description="Detects signs of misinformation by analysing cause-and-effect "
                "claims in news texts.",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=20, max_length=MAX_TEXT_LENGTH,
                      description="News article or post to analyse.")


@lru_cache(maxsize=1)
def get_analyzer() -> CausalNewsAnalyzer:
    return CausalNewsAnalyzer()


@app.get("/health")
def health() -> dict:
    get_analyzer()
    return {"status": "ok", "version": __version__}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    return get_analyzer().analyze(request.text)
