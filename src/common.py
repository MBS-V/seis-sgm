"""Paths and small helpers shared by all scripts (same code on Windows and Colab)."""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or pathlib.Path("/content").exists()
_drive = pathlib.Path("/content/drive/MyDrive")
# Persistent storage: Google Drive on Colab (survives session resets), local folder otherwise.
WORK = _drive / "seis_sgm" if (IN_COLAB and _drive.exists()) else ROOT / "_work"
DATA, CKPT, RUNS, RESULTS = (WORK / d for d in ("data", "ckpt", "runs", "results"))


def ensure_dirs():
    for p in (DATA, CKPT, RUNS, RESULTS):
        p.mkdir(parents=True, exist_ok=True)


def use_vendor(name="IPS-RNA"):
    """Put a vendored authors' repo first on sys.path so `import models` etc. resolve to it."""
    p = str(VENDOR / name)
    if p not in sys.path:
        sys.path.insert(0, p)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
