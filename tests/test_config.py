import json
import pytest
from rag.config import load_model_config
from rag.errors import ConfigurationError


def test_load_and_hide_secret(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"generation": {"provider": "openai_compatible", "base_url_env": "TEST_URL", "api_key_env": "TEST_KEY", "model_env": "TEST_MODEL"}}))
    monkeypatch.setenv("TEST_URL", "https://example.test/v1")
    monkeypatch.setenv("TEST_KEY", "secret-value")
    monkeypatch.setenv("TEST_MODEL", "model-a")
    config = load_model_config(path, "generation")
    assert config.model == "model-a"
    assert "secret-value" not in repr(config)
    monkeypatch.delenv("TEST_KEY")
    with pytest.raises(ConfigurationError, match="TEST_KEY"):
        load_model_config(path, "generation")
