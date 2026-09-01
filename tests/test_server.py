"""Unit and integration tests for FastAPI presentation server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_serve_ui_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "HH-FaceChain" in response.text
    assert "Verification Console" in response.text


def test_api_sample_images(client):
    response = client.get("/api/sample-images")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 2
    filenames = [s["filename"] for s in data["samples"]]
    assert "scan1.jpg" in filenames
    assert "scan2.jpg" in filenames


def test_api_chain_status_local(client):
    response = client.get("/api/chain-status?network=local")
    assert response.status_code == 200
    data = response.json()
    assert data["network"] == "local"
    assert "status" in data
    assert data["status"]["integrity_valid"] is True


def test_api_run_missing_input_error(client):
    response = client.post("/api/run", data={"network": "local", "tolerance": "0.35"})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "No image file or sample provided" in data["error"]


def test_api_run_sample_offline(client):
    response = client.post(
        "/api/run",
        data={
            "sample_id": "scan1.jpg",
            "network": "local",
            "tolerance": "0.35",
            "offline_demo": "true",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["matched"] is True
    assert data["accuracy_pct"] > 80.0
    assert "anchor" in data
    assert data["anchor"]["network"] == "local"
    assert len(data["anchor"]["content_hash"]) == 64


def test_api_verify_and_tamper(client):
    # Ensure a record exists first via offline run
    client.post(
        "/api/run",
        data={
            "sample_id": "scan1.jpg",
            "network": "local",
            "tolerance": "0.35",
            "offline_demo": "true",
        },
    )

    # Test verify
    v_resp = client.post("/api/verify", data={"network": "local"})
    assert v_resp.status_code == 200
    v_data = v_resp.json()
    assert "verified" in v_data

    # Test tamper
    t_resp = client.post("/api/tamper", data={"network": "local"})
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert t_data["tamper_detected"] is True
    assert t_data["original_hash"] != t_data["tampered_hash"]
