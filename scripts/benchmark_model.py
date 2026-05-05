from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from middleout_lattice.codec import CODECS, _encode_payload, _pack, compress_bytes, decompress_bytes

DEFAULT_FILES = ["config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]


def fetch(repo_id: str, filename: str) -> bytes:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=1"
    req = urllib.request.Request(url, headers={"User-Agent": "middleout-lattice/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--out-dir", default="artifacts/qwen05")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for filename in args.files:
        data = fetch(args.repo_id, filename)
        digest = hashlib.sha256(data).hexdigest()
        for codec in CODECS:
            blob = _pack(codec, digest, len(data), _encode_payload(codec, data))
            rows.append({
                "file": filename,
                "codec": codec,
                "original_bytes": len(data),
                "compressed_bytes": len(blob),
                "ratio": round(len(data) / max(1, len(blob)), 4),
                "roundtrip_ok": decompress_bytes(blob) == data,
            })
        best = compress_bytes(data)
        rows.append({
            "file": filename,
            "codec": best.codec,
            "original_bytes": len(data),
            "compressed_bytes": len(best.compressed_bytes),
            "ratio": round(len(data) / max(1, len(best.compressed_bytes)), 4),
            "roundtrip_ok": decompress_bytes(best.compressed_bytes) == data,
        })

    rows.sort(key=lambda row: (row["file"], row["compressed_bytes"]))
    csv_path = out_dir / "benchmark.csv"
    json_path = out_dir / "benchmark.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
