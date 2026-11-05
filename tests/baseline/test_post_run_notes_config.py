"""DR-ITER-062 configurable post-run notes."""

from __future__ import annotations

import json

import pytest

from deepresearch_core.baseline.post_run_notes_config import (
    coerce_post_run_notes_config,
    default_post_run_notes_config,
)
from deepresearch_core.config import (
    load_post_run_notes_config,
    load_provider_config,
    save_post_run_notes_config,
)


def test_post_run_notes_default_is_all_modes_off():
    assert default_post_run_notes_config().to_dict() == {"enabled_modes": []}
    assert coerce_post_run_notes_config({}).enabled_modes == ()


def test_post_run_notes_validates_modes():
    cfg = coerce_post_run_notes_config({"enabled_modes": ["deep", "council", "deep"]})
    assert cfg.enabled_modes == ("deep", "council")
    with pytest.raises(ValueError):
        coerce_post_run_notes_config({"enabled_modes": ["bogus"]})


def test_post_run_notes_config_roundtrip_preserves_other_blocks(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"provider": {"brave_api_key": "b"}}), encoding="utf-8")
    cfg = coerce_post_run_notes_config({"enabled_modes": ["fast", "deep"]})

    save_post_run_notes_config(path, cfg)

    assert load_post_run_notes_config(path).enabled_modes == ("fast", "deep")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["provider"]["brave_api_key"]) == {"secret_ref"}
    assert load_provider_config(path).brave_api_key == "b"
