"""
Integration tests for the prediction API.
Runs against the FastAPI app using httpx.AsyncClient — no trained models required.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def metrics_payload():
    return {
        "loc": 245, "lloc": 198, "num_methods": 12, "num_attributes": 8,
        "cyclomatic_complexity": 18, "halstead_volume": 4200,
        "halstead_effort": 95000, "halstead_difficulty": 22.6,
        "cbo": 12, "wmc": 24, "rfc": 45, "lcom": 0.73, "dit": 3, "noc": 1,
        "num_authors": 4, "code_churn": 312, "prior_defects": 2, "module_age_days": 540,
        "fan_in": 6, "fan_out": 9, "instability": 0.6,
    }


@pytest.fixture
def code_payload():
    return {
        "source_code": (
            "class DataProcessor:\n"
            "    def __init__(self):\n"
            "        self.data = []\n"
            "        self.results = []\n"
            "    def process(self, items):\n"
            "        for i in range(len(items)):\n"
            "            for j in range(len(items)):\n"
            "                if items[i] > items[j]:\n"
            "                    self.results.append(items[i])\n"
            "        return self.results\n"
        ),
        "language": "python",
    }


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_metrics_returns_200(metrics_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/predict/metrics", json=metrics_payload)
    assert r.status_code == 200
    body = r.json()
    assert "bug_probability" in body
    assert body["prediction"] in ("BUGGY", "CLEAN")
    assert body["severity"] in ("HIGH", "MEDIUM", "LOW", "NONE")
    assert isinstance(body["top_features"], list)


@pytest.mark.asyncio
async def test_predict_metrics_high_risk_module(metrics_payload):
    """A module with high CBO, WMC, and prior defects should be flagged."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/predict/metrics", json=metrics_payload)
    body = r.json()
    assert body["bug_probability"] > 0.5
    assert body["prediction"] == "BUGGY"


@pytest.mark.asyncio
async def test_predict_metrics_clean_module():
    clean_payload = {
        "loc": 30, "lloc": 25, "num_methods": 2, "num_attributes": 1,
        "cyclomatic_complexity": 2, "halstead_volume": 120,
        "halstead_effort": 800, "halstead_difficulty": 3.0,
        "cbo": 1, "wmc": 2, "rfc": 4, "lcom": 0.1, "dit": 1, "noc": 0,
        "num_authors": 1, "code_churn": 20, "prior_defects": 0, "module_age_days": 90,
        "fan_in": 2, "fan_out": 1, "instability": 0.3,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/predict/metrics", json=clean_payload)
    body = r.json()
    assert body["prediction"] == "CLEAN"


@pytest.mark.asyncio
async def test_predict_code_returns_200(code_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/predict/code", json=code_payload)
    assert r.status_code == 200
    body = r.json()
    assert "bug_probability" in body
    assert isinstance(body["code_smells"], list)
    assert "review_suggestion" in body


@pytest.mark.asyncio
async def test_predict_code_invalid_language(code_payload):
    code_payload["language"] = "cobol"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/predict/code", json=code_payload)
    assert r.status_code in (422, 500)
