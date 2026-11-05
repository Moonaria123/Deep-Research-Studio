import json
import subprocess
import sys
from pathlib import Path

from deepresearch_core.baseline.budget import Mode, mode_budget_presets, preset_for


ROOT = Path(__file__).resolve().parents[2]
GENERATED_PATH = ROOT / "apps" / "web" / "src" / "lib" / "modeBudgetPresets.generated.json"


def test_mode_budget_presets_exports_preview_fields_only():
    presets = mode_budget_presets()

    assert set(presets) == {mode.value for mode in Mode}
    assert presets["fast"] == {
        "max_searches": 1,
        "max_output_tokens": 700,
        "timeout_s": 30,
        "top_k": 6,
    }
    assert set(presets["deep"]) == {"max_searches", "max_output_tokens", "timeout_s", "top_k"}
    assert "max_think_tokens" not in presets["deep"]
    assert presets["council"]["max_output_tokens"] == preset_for(Mode.COUNCIL).max_output_tokens


def test_mode_budget_generated_json_matches_backend():
    generated = json.loads(GENERATED_PATH.read_text(encoding="utf-8"))
    assert generated == mode_budget_presets()


def test_export_mode_presets_check_detects_drift(tmp_path):
    stale = tmp_path / "modeBudgetPresets.generated.json"
    stale.write_text(json.dumps({"fast": {"max_searches": 999}}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_mode_presets.py",
            "--check",
            "--output",
            str(stale),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    assert "out of date" in result.stderr
