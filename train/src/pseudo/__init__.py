"""Token-classification pseudonymizer: data encoding, training, decoding, prediction."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
# PSEUDO_SILVER points a process at another dataset folder: scripts/launch.py trains on several datasets at the same time
SILVER = Path(os.environ.get("PSEUDO_SILVER") or DATA / "silver")
TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
L2I = {l: i for i, l in enumerate(LABELS)}
