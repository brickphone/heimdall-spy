"""Check that a remote machine reproduces the Mac's locator before a full evaluation.

    python -m eval.pod_check record data/auto_regions/pod_check_mac.json   # on the Mac
    python -m eval.pod_check compare data/auto_regions/pod_check_mac.json  # on the pod

The first 5 dev photos (manifest order) go through ``GeoLocator.locate_bytes`` with the
whole photo only. The pass rule was fixed before any pod run (2026-11-11): for all 5
photos the top-1 reference and the confidence state are identical, the top-5 sets share
at least 4 IDs, and the similarities of shared IDs differ by at most 1e-3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.auto_regions import build
from eval.gsv_eval import load_sets

N = 5
MIN_SHARED = 4
SIMILARITY_TOL = 1e-3


def results() -> list[dict[str, Any]]:
    locator = build("whole")
    out = []
    for q in load_sets()["dev"][:N]:
        r = locator.locate_bytes(Path(q["path"]).read_bytes())
        out.append({"image_id": q["image_id"], "state": r["confidence_state"],
                    "top5": [c["image_id"] for c in r["candidates"]],
                    "similarity": {c["image_id"]: c["similarity"] for c in r["candidates"]},
                    "device": locator.versions()["model"]["device"]})
    return out


def compare(mac: list[dict[str, Any]], here: list[dict[str, Any]]) -> list[str]:
    """Problems found; empty means pass."""
    problems = []
    for a, b in zip(mac, here, strict=True):
        name = a["image_id"]
        if a["image_id"] != b["image_id"]:
            problems.append(f"{name}: different photo order")
            continue
        if a["top5"][0] != b["top5"][0]:
            problems.append(f"{name}: top-1 differs")
        if a["state"] != b["state"]:
            problems.append(f"{name}: state {a['state']} vs {b['state']}")
        shared = set(a["top5"]) & set(b["top5"])
        if len(shared) < MIN_SHARED:
            problems.append(f"{name}: top-5 share {len(shared)} < {MIN_SHARED} IDs")
        worst = max(abs(a["similarity"][i] - b["similarity"][i]) for i in shared) if shared else 1
        if worst > SIMILARITY_TOL:
            problems.append(f"{name}: similarity differs by {worst:.2e} > {SIMILARITY_TOL}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("record", "compare"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    here = results()
    if args.command == "record":
        args.path.parent.mkdir(parents=True, exist_ok=True)
        args.path.write_text(json.dumps(here, indent=1))
        print(f"recorded {len(here)} photos on {here[0]['device']}")
        return
    problems = compare(json.loads(args.path.read_text()), here)
    shared = [len(set(a["top5"]) & set(b["top5"]))
              for a, b in zip(json.loads(args.path.read_text()), here, strict=True)]
    print(json.dumps({"pass": not problems, "shared_top5": shared, "problems": problems},
                     indent=1))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
