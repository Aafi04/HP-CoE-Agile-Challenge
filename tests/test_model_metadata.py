#!/usr/bin/env python3
"""Validate model identity metadata endpoints for deployment safety."""

import requests

API_URL = "http://127.0.0.1:8000"


def main() -> None:
    health = requests.get(f"{API_URL}/health", timeout=30)
    model_info = requests.get(f"{API_URL}/model-info", timeout=30)

    assert health.status_code == 200, f"/health failed: {health.status_code}"
    assert model_info.status_code == 200, f"/model-info failed: {model_info.status_code}"

    health_data = health.json()
    model_data = model_info.json()

    for key in ["status", "device", "model", "model_path", "model_version", "model_hash", "expected_hash_set"]:
        assert key in health_data, f"Missing health key: {key}"

    for key in [
        "model_path",
        "model_type",
        "model_version",
        "model_hash",
        "expected_model_hash",
        "hash_matches_expected",
        "device",
    ]:
        assert key in model_data, f"Missing model-info key: {key}"

    assert health_data["status"] == "ok"
    assert isinstance(model_data["model_hash"], str) and len(model_data["model_hash"]) == 64
    assert isinstance(model_data["hash_matches_expected"], bool)

    print("Model metadata endpoint tests passed")


if __name__ == "__main__":
    main()
