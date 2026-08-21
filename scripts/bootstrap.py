"""Environment smoke check: import the key packages and report hardware state.

    .venv/bin/python -m scripts.bootstrap
"""
from __future__ import annotations

import importlib
import sys

PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "lime",
    "captum",
    "bs4",
    "shap",  # optional
    "numba",  # optional
]


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    failed = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✓ {name:14s} {version}")
        except ImportError as exc:
            failed.append(name)
            print(f"  ✗ {name:14s} (import failed: {exc})")

    try:
        import torch

        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  CPU threads: {torch.get_num_threads()}")
    except Exception:  # noqa: BLE001
        pass

    if failed:
        print(f"\nMissing/optional-uninstalled: {', '.join(failed)}")
        raise SystemExit(1 if "torch" in failed or "transformers" in failed else 0)
    print("\nBootstrap OK.")


if __name__ == "__main__":
    main()
