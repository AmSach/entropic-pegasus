from __future__ import annotations

import csv
from pathlib import Path


def _svg_bar_chart(rows):
    labels = [f"{r['file']} / {r['codec']}" for r in rows]
    ratios = [float(r["ratio"]) for r in rows]
    width = 1400
    height = 760
    margin = 70
    chart_w = width - margin * 2
    chart_h = height - margin * 2
    max_ratio = max(ratios) if ratios else 1
    bar_w = chart_w / max(1, len(rows)) * 0.72
    gap = chart_w / max(1, len(rows)) * 0.28
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="#0f172a"/>')
    parts.append(f'<text x="{margin}" y="40" fill="#e2e8f0" font-size="28" font-family="Inter,Arial,sans-serif">MiddleOut Lattice benchmark</text>')
    for i, (label, ratio) in enumerate(zip(labels, ratios)):
        x = margin + i * (bar_w + gap) + gap * 0.5
        bar_h = chart_h * (ratio / max_ratio)
        y = height - margin - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="8" fill="#38bdf8"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" fill="#f8fafc" font-size="14" font-family="monospace">{ratio:.2f}x</text>')
        parts.append(f'<text x="{x:.1f}" y="{height - margin + 18}" fill="#cbd5e1" font-size="11" font-family="monospace" transform="rotate(35 {x:.1f},{height - margin + 18})">{label}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    report_dir = Path("reports")
    csv_path = report_dir / "benchmark.csv"
    out_md = report_dir / "REPORT.md"
    out_svg = report_dir / "comparison.svg"
    if not csv_path.exists():
        raise SystemExit("benchmark.csv not found; run scripts/benchmark_model.py first")
    rows = list(csv.DictReader(csv_path.open()))
    best = max(rows, key=lambda r: float(r["ratio"]))
    out_md.write_text(
        "# Benchmark report\n\n"
        f"- best file: {best['file']}\n"
        f"- best codec: {best['codec']}\n"
        f"- original bytes: {best['original_bytes']}\n"
        f"- compressed bytes: {best['compressed_bytes']}\n"
        f"- ratio: {best['ratio']}\n"
        f"- roundtrip ok: {best['roundtrip_ok']}\n\n"
        "## Full comparison\n\n"
        + "\n".join(f"- {r['file']} / {r['codec']}: {r['ratio']}x" for r in rows)
        + "\n"
    )
    out_svg.write_text(_svg_bar_chart(rows))
    print(out_md)
    print(out_svg)


if __name__ == "__main__":
    main()
