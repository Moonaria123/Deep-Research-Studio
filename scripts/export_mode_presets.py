"""Export backend mode budget presets for the web preview UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "apps" / "web" / "src" / "lib" / "modeBudgetPresets.generated.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch_core.baseline.budget import mode_budget_presets  # noqa: E402


def _render() -> str:
    return json.dumps(mode_budget_presets(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the output file is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="path to generated JSON")
    args = parser.parse_args()

    expected = _render()
    output = args.output.resolve()

    if args.check:
        actual = output.read_text(encoding="utf-8") if output.exists() else ""
        if actual != expected:
            print(f"{output} is out of date; run scripts/export_mode_presets.py", file=sys.stderr)
            return 1
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
