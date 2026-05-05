from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = Path('/home/workspace/Projects/amsach.github.io')


def main():
    print(SITE / 'papers' / 'middleout-lattice-paper' / 'index.html')
