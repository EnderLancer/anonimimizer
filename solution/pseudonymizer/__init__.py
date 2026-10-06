"""Псевдонімізатор персональних даних в українських текстах (нейромережа + regex + CRF)."""
from .pipeline import Pseudonymizer, build_annotations

__all__ = ["Pseudonymizer", "build_annotations"]
