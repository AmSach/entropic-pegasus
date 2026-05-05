from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def main():
    csv_path = REPORT_DIR / "benchmark.csv"
    out = REPORT_DIR / "comparison.png"
    rows = list(csv.DictReader(csv_path.open()))
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    draw.text((40, 30), "MiddleOut Lattice benchmark", fill="#e2e8f0", font=title_font)
    y = 90
    for row in rows:
        line = f"{row['file']} | {row['codec']} | ratio {row['ratio']}x | roundtrip {row['roundtrip_ok']}"
        draw.text((40, y), line, fill="#cbd5e1", font=font)
        y += 28

    bar_top = 350
    bar_left = 80
    bar_area_w = width - 160
    bar_area_h = 450
    ratios = [float(r["ratio"]) for r in rows]
    max_ratio = max(ratios) if ratios else 1
    bar_w = bar_area_w / max(1, len(rows)) * 0.7
    gap = bar_area_w / max(1, len(rows)) * 0.3
    for i, row in enumerate(rows):
        ratio = float(row["ratio"])
        x = bar_left + i * (bar_w + gap)
        bh = bar_area_h * ratio / max_ratio
        y0 = bar_top + (bar_area_h - bh)
        draw.rectangle([x, y0, x + bar_w, bar_top + bar_area_h], fill="#38bdf8")
        draw.text((x, y0 - 20), f"{ratio:.2f}x", fill="#f8fafc", font=font)
        draw.text((x, bar_top + bar_area_h + 10), f"{row['codec']}", fill="#94a3b8", font=font)

    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
