"""Environment and artifact check.

Imports what the running app needs, reports hardware state, and — the part that
actually saves time — confirms the pipeline's artifacts exist. A missing
`stage2_detector.joblib` surfaces here rather than as a 503 on the first
analyze request.

    .venv/bin/python -m scripts.bootstrap
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# What the serving path imports. `lime` and `captum` are deliberately absent —
# they belonged to the archived sentiment pipeline and are no longer used.
PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "torch",
    "transformers",
    "datasets",   # only needed to build a packet
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "bs4",        # app.preprocess
    "shap",       # required: the detector service builds a TreeExplainer
    "joblib",     # loads the detector bundle
    "numba",      # shap's tree path uses it when present
]

# (path, what builds it, what breaks without it)
ARTIFACTS = [
    ("models/stage2_detector.joblib", "make stage2", "the app cannot start"),
    ("models/stage2_reference.json", "make stage2",
     "reason codes cannot report percentiles"),
    ("data/gen_pairs/*.jsonl", "make ingest-gen",
     "`make stage2` has nothing to train on"),
    ("data/hitl_pool.json", "make stage4", "the HITL page 503s"),
    ("data/examples.json", "make stage4", "the Examples page 503s"),
    ("frontend/dist/index.html", "make frontend", "the UI is not served"),
]


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    failed = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  ok   {name:14s} {getattr(mod, '__version__', '?')}")
        except ImportError as exc:
            failed.append(name)
            print(f"  FAIL {name:14s} {exc}")

    try:
        import torch

        print(f"\n  MPS available: {torch.backends.mps.is_available()}")
        print(f"  CPU threads:   {torch.get_num_threads()}")
    except Exception:  # noqa: BLE001
        pass

    print("\nArtifacts")
    missing = []
    for pattern, builder, consequence in ARTIFACTS:
        hits = list(Path(".").glob(pattern))
        if hits:
            print(f"  ok   {pattern:30s} ({len(hits)} file(s))")
        else:
            missing.append((pattern, builder, consequence))
            print(f"  MISS {pattern:30s} -> run `{builder}`; without it {consequence}")

    if failed:
        print(f"\nMissing packages: {', '.join(failed)}")
    if missing:
        print(f"Missing artifacts: {len(missing)} (see above)")

    # Only a broken interpreter is fatal here; a missing artifact is a state the
    # app reports through /api/health anyway.
    hard = {"torch", "transformers", "sklearn", "shap", "joblib"}
    if hard & set(failed):
        print("\nBootstrap FAILED — core packages missing.")
        raise SystemExit(1)
    print("\nBootstrap OK." + ("" if not missing else "  (artifacts pending)"))


if __name__ == "__main__":
    main()
