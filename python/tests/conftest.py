"""Make the flat modules in python/ importable (``import compare``, ``import ts_masks``, ...)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
