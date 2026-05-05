from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .codec import compress_bytes, decompress_bytes
from .mosaic import MosaicMeta, mosaic_candidates, mosaic_decode

MAGIC = b"MTSH1"


@dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: list[int]
    data_offsets: list[int]
    mode: str
    codec: str
    transform: str
    transform_meta: dict[str, Any] | None
    raw_size: int
    stored_size: int
    sha256: str


@dataclass
class TensorShardManifest:
    version: int
    original_header_b64: str
    original_size: int
    shard_sha256: str
    tensors: list[TensorEntry]


@dataclass
class TensorArray:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_safetensors_blob(tensors: Iterable[TensorArray], metadata: dict[str, Any] | None = None) -> bytes:
    tensors = list(tensors)
    header: dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    body = bytearray()
    for tensor in tensors:
        start = offset
        end = start + len(tensor.data)
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, end],
        }
        body.extend(tensor.data)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(header_bytes).to_bytes(8, "little") + header_bytes + bytes(body)


def parse_safetensors_blob(blob: bytes) -> tuple[dict[str, Any], bytes, bytes, list[TensorEntry]]:
    if len(blob) < 8:
        raise ValueError("invalid safetensors blob")
    header_len = int.from_bytes(blob[:8], "little")
    header_start = 8
    header_end = header_start + header_len
    header_bytes = blob[header_start:header_end]
    header = json.loads(header_bytes.decode("utf-8"))
    body = blob[header_end:]
    entries: list[TensorEntry] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        data_offsets = [int(x) for x in info["data_offsets"]]
        entries.append(
            TensorEntry(
                name=name,
                dtype=str(info["dtype"]),
                shape=[int(x) for x in info["shape"]],
                data_offsets=data_offsets,
                mode="raw",
                codec="raw",
                transform="none",
                transform_meta=None,
                raw_size=data_offsets[1] - data_offsets[0],
                stored_size=data_offsets[1] - data_offsets[0],
                sha256=_sha256(body[data_offsets[0] : data_offsets[1]]),
            )
        )
    entries.sort(key=lambda item: item.data_offsets[0])
    return header, header_bytes, body, entries


@dataclass
class TensorEncoding:
    mode: str
    codec: str
    transform: str
    transform_meta: dict[str, Any] | None
    payload: bytes
    raw_size: int
    stored_size: int
    sha256: str


def _best_tensor_encoding(data: bytes) -> TensorEncoding:
    candidates: list[TensorEncoding] = [
        TensorEncoding(
            mode="raw",
            codec="raw",
            transform="none",
            transform_meta=None,
            payload=data,
            raw_size=len(data),
            stored_size=len(data),
            sha256=_sha256(data),
        )
    ]

    generic = compress_bytes(data)
    candidates.append(
        TensorEncoding(
            mode="generic",
            codec=generic.codec,
            transform="none",
            transform_meta=None,
            payload=generic.compressed_bytes,
            raw_size=len(data),
            stored_size=len(generic.compressed_bytes),
            sha256=generic.sha256,
        )
    )

    for transformed, meta in mosaic_candidates(data):
        transformed_best = compress_bytes(transformed)
        candidates.append(
            TensorEncoding(
                mode="mosaic",
                codec=transformed_best.codec,
                transform="mosaic",
                transform_meta=asdict(meta),
                payload=transformed_best.compressed_bytes,
                raw_size=len(data),
                stored_size=len(transformed_best.compressed_bytes),
                sha256=transformed_best.sha256,
            )
        )

    return min(candidates, key=lambda item: item.stored_size)


def compress_safetensors_blob(blob: bytes) -> bytes:
    header, header_bytes, body, entries = parse_safetensors_blob(blob)
    payload_parts: list[bytes] = []
    manifest_entries: list[TensorEntry] = []

    for entry in entries:
        raw = body[entry.data_offsets[0] : entry.data_offsets[1]]
        chosen = _best_tensor_encoding(raw)
        payload_parts.append(chosen.payload)
        manifest_entries.append(
            TensorEntry(
                name=entry.name,
                dtype=entry.dtype,
                shape=entry.shape,
                data_offsets=[entry.data_offsets[0], entry.data_offsets[1]],
                mode=chosen.mode,
                codec=chosen.codec,
                transform=chosen.transform,
                transform_meta=chosen.transform_meta,
                raw_size=chosen.raw_size,
                stored_size=len(chosen.payload),
                sha256=chosen.sha256,
            )
        )

    manifest = TensorShardManifest(
        version=1,
        original_header_b64=base64.b64encode(header_bytes).decode("ascii"),
        original_size=len(blob),
        shard_sha256=_sha256(blob),
        tensors=manifest_entries,
    )
    manifest_bytes = json.dumps(
        {
            "version": manifest.version,
            "original_header_b64": manifest.original_header_b64,
            "original_size": manifest.original_size,
            "shard_sha256": manifest.shard_sha256,
            "tensors": [
                {
                    "name": tensor.name,
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": tensor.data_offsets,
                    "mode": tensor.mode,
                    "codec": tensor.codec,
                    "transform": tensor.transform,
                    "transform_meta": tensor.transform_meta,
                    "raw_size": tensor.raw_size,
                    "stored_size": tensor.stored_size,
                    "sha256": tensor.sha256,
                }
                for tensor in manifest.tensors
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return MAGIC + len(manifest_bytes).to_bytes(8, "little") + manifest_bytes + b"".join(payload_parts)


def decompress_safetensors_blob(blob: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("invalid tensor shard archive")
    manifest_len = int.from_bytes(blob[len(MAGIC) : len(MAGIC) + 8], "little")
    manifest_start = len(MAGIC) + 8
    manifest_end = manifest_start + manifest_len
    manifest = json.loads(blob[manifest_start:manifest_end].decode("utf-8"))
    payload = memoryview(blob[manifest_end:])
    header_bytes = base64.b64decode(manifest["original_header_b64"])
    cursor = 0
    body = bytearray()

    for tensor in manifest["tensors"]:
        stored_size = int(tensor["stored_size"])
        chunk = bytes(payload[cursor : cursor + stored_size])
        cursor += stored_size

        if tensor["mode"] == "raw":
            raw = chunk
        else:
            raw = decompress_bytes(chunk)
            if tensor["mode"] == "mosaic":
                raw = mosaic_decode(raw, tensor["transform_meta"])

        if len(raw) != int(tensor["raw_size"]):
            raise ValueError(f"tensor size mismatch for {tensor['name']}")
        if _sha256(raw) != tensor["sha256"]:
            raise ValueError(f"tensor checksum mismatch for {tensor['name']}")
        body.extend(raw)

    rebuilt = len(header_bytes).to_bytes(8, "little") + header_bytes + bytes(body)
    if _sha256(rebuilt) != manifest["shard_sha256"]:
        raise ValueError("shard checksum mismatch")
    return rebuilt


def compress_safetensors_file(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = compress_safetensors_blob(source.read_bytes())
    out = target_dir / f"{source.name}.mshard"
    out.write_bytes(archive)
    return out


def decompress_safetensors_file(source: Path, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(decompress_safetensors_blob(source.read_bytes()))
    return target_path


def benchmark_tensor_strategies(blob: bytes) -> list[dict[str, Any]]:
    header, header_bytes, body, entries = parse_safetensors_blob(blob)
    del header, header_bytes
    rows: list[dict[str, Any]] = []
    raw_size = len(blob)

    rows.append({"strategy": "raw", "original_bytes": raw_size, "compressed_bytes": raw_size, "ratio": 1.0, "roundtrip_ok": True})

    generic = compress_bytes(blob)
    rows.append({
        "strategy": f"whole-file:{generic.codec}",
        "original_bytes": raw_size,
        "compressed_bytes": len(generic.compressed_bytes),
        "ratio": round(raw_size / max(1, len(generic.compressed_bytes)), 4),
        "roundtrip_ok": decompress_bytes(generic.compressed_bytes) == blob,
    })

    per_tensor_raw = 0
    per_tensor_generic = 0
    per_tensor_mosaic = 0
    mosaic_ok = True
    generic_ok = True

    for entry in entries:
        raw = body[entry.data_offsets[0] : entry.data_offsets[1]]
        per_tensor_raw += len(raw)

        generic_best = compress_bytes(raw)
        per_tensor_generic += len(generic_best.compressed_bytes)
        generic_ok = generic_ok and decompress_bytes(generic_best.compressed_bytes) == raw

        chosen = _best_tensor_encoding(raw)
        per_tensor_mosaic += chosen.stored_size
        if chosen.mode == "mosaic":
            restored = mosaic_decode(decompress_bytes(chosen.payload), chosen.transform_meta or {})
        elif chosen.mode == "generic":
            restored = decompress_bytes(chosen.payload)
        else:
            restored = chosen.payload
        mosaic_ok = mosaic_ok and restored == raw

    rows.append({"strategy": "per-tensor generic", "original_bytes": per_tensor_raw, "compressed_bytes": per_tensor_generic, "ratio": round(per_tensor_raw / max(1, per_tensor_generic), 4), "roundtrip_ok": generic_ok})
    rows.append({"strategy": "per-tensor mosaic", "original_bytes": per_tensor_raw, "compressed_bytes": per_tensor_mosaic, "ratio": round(per_tensor_raw / max(1, per_tensor_mosaic), 4), "roundtrip_ok": mosaic_ok})
    return rows
