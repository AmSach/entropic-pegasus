from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper' / 'main.tex'
OUT_DIR = ROOT / 'build'


def main():