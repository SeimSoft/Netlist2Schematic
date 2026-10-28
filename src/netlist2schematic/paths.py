"""Location of the training database and model weights."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """``$N2S_DATA`` if set, otherwise ``netlist2schematic/data``."""
    override = os.environ.get("N2S_DATA")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data"


def db_path() -> Path:
    return data_dir() / "training.sqlite"


def model_path() -> Path:
    return data_dir() / "policy.npz"
