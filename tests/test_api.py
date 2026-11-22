"""Basic API tests using FastAPI TestClient (in-process, no running server)."""

import os
import tempfile

os.environ.setdefault("CHROMA_PATH", tempfile.mkdtemp())
os.environ.setdefault("LLM_PROVIDER", "mock")

import config

config.settings.chroma_path = os.environ["CHROMA_PATH"]
config.settings.collection_name = "test_documents"

from main import app  # noqa: E402
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "documents" in body


def test_ingest_and_query(client):
    r = client.post("/ingest/text", json={"text": "The sky is blue. Water is wet.", "source": "facts.txt"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

    r = client.post("/query", json={"question": "What color is the sky?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert len(body["sources"]) >= 1


def test_query_empty_store_grounded_false(client):
    client.post("/reset")
    r = client.post("/query", json={"question": "Anything?"})
    assert r.status_code == 200
    assert r.json()["grounded"] is False


def test_ingest_empty_text_rejected(client):
    r = client.post("/ingest/text", json={"text": "   "})
    assert r.status_code == 400
