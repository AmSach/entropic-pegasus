from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .codec import CODECS, _decode_payload, _encode_payload
from .mosaic import mosaic_candidates, mosaic_decode, MosaicMeta

MAGIC = b"MOLA2"


@dataclass
class BlockMeta:
    index: int
    codec: str
    raw_size: int
    stored_size: int
    sha256: str
    mode: str = "codec"
    mosaic: dict[str, Any] | None = None


@dataclass
class FileRecord:
    path: str
    mode: str
    original_bytes: int
    stored_bytes: int
    sha256: str
    block_size: int
    storage_path: str
    blocks: list[BlockMeta]


@dataclass
class ArchiveManifest:
    version: int
    root: str
    files: list[FileRecord]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _best_block_encoding(data: bytes) -> tuple[str, bytes, str, dict[str, Any] | None]:
    candidates: list[tuple[str, bytes, str, dict[str, Any] | None]] = [("raw", data, "raw", None)]
    for codec in CODECS:
        encoded = _encode_payload(codec, data)
        candidates.append((codec, encoded, "codec", None))
    for encoded, meta in mosaic_candidates(data):
        if len(encoded) < len(data):
            candidates.append(("mosaic", encoded, "mosaic", {
                "word_size": meta.word_size,
                "plane_order": meta.plane_order,
                "residual": meta.residual,
                "pad_len": meta.pad_len,
                "original_size": meta.original_size,
                "word_count": meta.word_count,
            }))
    codec, payload, mode, mosaic = min(candidates, key=lambda item: len(item[1]))
    return codec, payload, mode, mosaic


def _pack_archive(blocks: list[dict[str, Any]]) -> bytes:
    serializable = []
    payload = bytearray()
    for block in blocks:
        serializable.append({k: v for k, v in block.items() if k != "_payload"})
        payload.extend(block["_payload"])
    header = json.dumps({"blocks": serializable}, separators=(",", ":")).encode("utf-8")
    return MAGIC + len(header).to_bytes(8, "little") + header + bytes(payload)


def _unpack_archive(blob: bytes) -> tuple[list[dict[str, Any]], memoryview]:
    if not blob.startswith(MAGIC):
        raise ValueError("invalid archive")
    header_len = int.from_bytes(blob[len(MAGIC) : len(MAGIC) + 8], "little")
    start = len(MAGIC) + 8
    header = json.loads(blob[start : start + header_len].decode("utf-8"))
    payload = memoryview(blob[start + header_len :])
    return header["blocks"], payload


def compress_file_bytes(data: bytes, block_size: int = 1 << 20) -> tuple[bytes, FileRecord]:
    if not data:
        record = FileRecord(
            path="",
            mode="raw",
            original_bytes=0,
            stored_bytes=0,
            sha256=_sha256(data),
            block_size=block_size,
            storage_path="",
            blocks=[],
        )
        return data, record

    raw_size = len(data)
    blocks: list[dict[str, Any]] = []

    for index, offset in enumerate(range(0, raw_size, block_size)):
        chunk = data[offset : offset + block_size]
        codec, payload, mode, mosaic = _best_block_encoding(chunk)
        blocks.append(
            {
                "index": index,
                "codec": codec,
                "mode": mode,
                "mosaic": mosaic,
                "raw_size": len(chunk),
                "stored_size": len(payload),
                "sha256": _sha256(chunk),
                "_payload": payload,
            }
        )

    candidate = _pack_archive([dict(block) for block in blocks])
    if len(candidate) >= raw_size:
        record = FileRecord(
            path="",
            mode="raw",
            original_bytes=raw_size,
            stored_bytes=raw_size,
            sha256=_sha256(data),
            block_size=block_size,
            storage_path="",
            blocks=[],
        )
        return data, record

    archive_blocks = [{k: v for k, v in block.items() if k != "_payload"} for block in blocks]
    archive = _pack_archive(blocks)
    record = FileRecord(
        path="",
        mode="archive",
        original_bytes=raw_size,
        stored_bytes=len(archive),
        sha256=_sha256(data),
        block_size=block_size,
        storage_path="",
        blocks=[BlockMeta(**block) for block in archive_blocks],
    )
    return archive, record


def decompress_file_bytes(blob: bytes, record: FileRecord) -> bytes:
    if record.mode == "raw":
        return blob

    blocks, payload = _unpack_archive(blob)
    cursor = 0
    parts: list[bytes] = []
    for block in blocks:
        stored_size = int(block["stored_size"])
        chunk = bytes(payload[cursor : cursor + stored_size])
        cursor += stored_size
        mode = block.get("mode", "codec")
        if mode == "raw":
            part = chunk
        elif mode == "mosaic":
            part = mosaic_decode(chunk, block["mosaic"])
        else:
            part = _decode_payload(block["codec"], chunk)
        if len(part) != int(block["raw_size"]):
            raise ValueError("size mismatch")
        if _sha256(part) != block["sha256"]:
            raise ValueError("checksum mismatch")
        parts.append(part)
    data = b"".join(parts)
    if _sha256(data) != record.sha256:
        raise ValueError("file checksum mismatch")
    return data


def compress_directory(source_dir: Path, target_dir: Path, block_size: int = 1 << 20) -> Path:
    source_dir = source_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = ArchiveManifest(version=2, root=str(source_dir), files=[])

    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(source_dir)
        data = source.read_bytes()
        archive, record = compress_file_bytes(data, block_size=block_size)
        record.path = str(rel)
        if record.mode == "raw":
            dest = target_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            record.storage_path = str(rel)
        else:
            dest = target_dir / f"{rel}.molm"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive)
            record.storage_path = f"{rel}.molm"
        manifest.files.append(record)

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "version": manifest.version,
        "root": manifest.root,
        "files": [
            {
                "path": rec.path,
                "mode": rec.mode,
                "original_bytes": rec.original_bytes,
                "stored_bytes": rec.stored_bytes,
                "sha256": rec.sha256,
                "block_size": rec.block_size,
                "storage_path": rec.storage_path,
                "blocks": [asdict(b) for b in rec.blocks],
            }
            for rec in manifest.files
        ],
    }, indent=2))
    return manifest_path


def decompress_directory(manifest_path: Path, target_dir: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        stored = base_dir / entry["storage_path"]
        if entry["mode"] == "raw":
            data = stored.read_bytes()
        else:
            data = decompress_file_bytes(stored.read_bytes(), FileRecord(
                path=entry["path"],
                mode=entry["mode"],
                original_bytes=entry["original_bytes"],
                stored_bytes=entry["stored_bytes"],
                sha256=entry["sha256"],
                block_size=entry["block_size"],
                storage_path=entry["storage_path"],
                blocks=[BlockMeta(**b) for b in entry["blocks"]],
            ))
        out = target_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        if _sha256(data) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {rel}")
    return target_dir
