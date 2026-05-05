from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"MOLM2"
CODECS = ("lzma", "zlib", "bz2")


@dataclass
class CodecResult:
    compressed_bytes: bytes
    original_bytes: int
    compressed_size: int
    sha256: str
    codec: str


def _encode_payload(codec: str, data: bytes) -> bytes:
    if codec == "lzma":
        return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    if codec == "zlib":
        return zlib.compress(data, level=9)
    if codec == "bz2":
        return bz2.compress(data, compresslevel=9)
    raise ValueError(f"Unknown codec: {codec}")


def _decode_payload(codec: str, data: bytes) -> bytes:
    if codec == "lzma":
        return lzma.decompress(data)
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "bz2":
        return bz2.decompress(data)
    raise ValueError(f"Unknown codec: {codec}")


def _pack(codec: str, sha256: str, size: int, payload: bytes) -> bytes:
    header = json.dumps({"codec": codec, "sha256": sha256, "size": size}).encode("utf-8")
    return MAGIC + struct.pack("<II", len(header), len(payload)) + header + payload


def _unpack(blob: bytes) -> tuple[str, dict[str, Any], bytes]:
    if not blob.startswith(MAGIC):
        raise ValueError("invalid archive")
    offset = len(MAGIC)
    header_len, payload_len = struct.unpack("<II", blob[offset : offset + 8])
    offset += 8
    header = json.loads(blob[offset : offset + header_len].decode("utf-8"))
    payload = blob[offset + header_len : offset + header_len + payload_len]
    return header["codec"], header, payload


def compress_bytes(data: bytes) -> CodecResult:
    digest = hashlib.sha256(data).hexdigest()
    candidates = [(_encode_payload(codec, data), codec) for codec in CODECS]
    best_payload, best_codec = min(candidates, key=lambda item: len(item[0]))
    blob = _pack(best_codec, digest, len(data), best_payload)
    return CodecResult(blob, len(data), len(blob), digest, best_codec)


def decompress_bytes(blob: bytes) -> bytes:
    codec, header, payload = _unpack(blob)
    data = _decode_payload(codec, payload)
    if hashlib.sha256(data).hexdigest() != header["sha256"]:
        raise ValueError("checksum mismatch")
    if len(data) != header["size"]:
        raise ValueError("size mismatch")
    return data


def compress_path(source: Path, target_dir: Path, codec: str = "best") -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    if codec == "best":
        best_path: Path | None = None
        best_size: int | None = None
        for current in CODECS:
            current_path = target_dir / f"{source.name}.{current}.molm"
            blob = _pack(current, hashlib.sha256(data).hexdigest(), len(data), _encode_payload(current, data))
            current_path.write_bytes(blob)
            size = current_path.stat().st_size
            if best_size is None or size < best_size:
                if best_path is not None and best_path.exists():
                    best_path.unlink()
                best_path = current_path
                best_size = size
            else:
                current_path.unlink(missing_ok=True)
        if best_path is None:
            raise RuntimeError("compression failed")
        canonical = target_dir / f"{source.name}.molm"
        if canonical.exists():
            canonical.unlink()
        best_path.rename(canonical)
        return canonical

    blob = _pack(codec, hashlib.sha256(data).hexdigest(), len(data), _encode_payload(codec, data))
    out = target_dir / f"{source.name}.molm"
    out.write_bytes(blob)
    return out


def decompress_path(source: Path, target_path: Path) -> Path:
    codec, header, payload = _unpack(source.read_bytes())
    data = _decode_payload(codec, payload)
    if hashlib.sha256(data).hexdigest() != header["sha256"]:
        raise ValueError("checksum mismatch")
    if len(data) != header["size"]:
        raise ValueError("size mismatch")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)
    return target_path


def roundtrip_path(source: Path, target_dir: Path) -> dict[str, Any]:
    compressed = compress_path(source, target_dir, codec="best")
    restored = target_dir / f"{source.name}.restored"
    decompress_path(compressed, restored)
    ok = restored.read_bytes() == source.read_bytes()
    return {"source": str(source), "compressed": str(compressed), "roundtrip_ok": ok, "bytes": source.stat().st_size}
