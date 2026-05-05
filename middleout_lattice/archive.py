from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .codec import CODECS, _decode_payload, _encode_payload
from .mosaic import MosaicMeta, mosaic_candidates, mosaic_decode

MAGIC = b"MOLA2"


@dataclass
class BlockMeta:
    index: int
    codec: str
    raw_size: int
    stored_size: int
    sha256: str
    transform: str = "none"
    transform_meta: dict[str, Any] | None = None


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


@dataclass
class Candidate:
    codec: str
    payload: bytes
    transform: str
    transform_meta: dict[str, Any] | None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate_streams(data: bytes) -> list[Candidate]:
    candidates: list[Candidate] = [Candidate(codec="raw", payload=data, transform="none", transform_meta=None)]
    for codec in CODECS:
        candidates.append(Candidate(codec=codec, payload=_encode_payload(codec, data), transform="none", transform_meta=None))

    for transformed, meta in mosaic_candidates(data):
        meta_dict = asdict(meta)
        for codec in CODECS:
            candidates.append(
                Candidate(
                    codec=f"mosaic:{codec}",
                    payload=_encode_payload(codec, transformed),
                    transform="mosaic",
                    transform_meta=meta_dict,
                )
            )
    return candidates


def _best_block_encoding(data: bytes) -> Candidate:
    return min(_candidate_streams(data), key=lambda item: len(item.payload))


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


def _decode_candidate(chunk: bytes, block: dict[str, Any]) -> bytes:
    codec = str(block["codec"])
    transform = str(block.get("transform", "none"))
    transform_meta = block.get("transform_meta")
    if codec == "raw":
        return chunk
    if codec.startswith("mosaic:"):
        inner_codec = codec.split(":", 1)[1]
        transformed = _decode_payload(inner_codec, chunk)
        if transform_meta is None:
            raise ValueError("missing mosaic metadata")
        return mosaic_decode(transformed, transform_meta)
    return _decode_payload(codec, chunk)


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
        candidate = _best_block_encoding(chunk)
        blocks.append(
            {
                "index": index,
                "codec": candidate.codec,
                "raw_size": len(chunk),
                "stored_size": len(candidate.payload),
                "sha256": _sha256(chunk),
                "transform": candidate.transform,
                "transform_meta": candidate.transform_meta,
                "_payload": candidate.payload,
            }
        )

    archive = _pack_archive(blocks)
    if len(archive) >= raw_size:
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

    record = FileRecord(
        path="",
        mode="archive",
        original_bytes=raw_size,
        stored_bytes=len(archive),
        sha256=_sha256(data),
        block_size=block_size,
        storage_path="",
        blocks=[
            BlockMeta(
                index=int(block["index"]),
                codec=str(block["codec"]),
                raw_size=int(block["raw_size"]),
                stored_size=int(block["stored_size"]),
                sha256=str(block["sha256"]),
                transform=str(block.get("transform", "none")),
                transform_meta=block.get("transform_meta"),
            )
            for block in blocks
        ],
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
        part = _decode_candidate(chunk, block)
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
    manifest = ArchiveManifest(version=1, root=str(source_dir), files=[])

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
    manifest_path.write_text(
        json.dumps(
            {
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
            },
            indent=2,
        )
    )
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
            data = decompress_file_bytes(
                stored.read_bytes(),
                FileRecord(
                    path=entry["path"],
                    mode=entry["mode"],
                    original_bytes=entry["original_bytes"],
                    stored_bytes=entry["stored_bytes"],
                    sha256=entry["sha256"],
                    block_size=entry["block_size"],
                    storage_path=entry["storage_path"],
                    blocks=[BlockMeta(**b) for b in entry["blocks"]],
                ),
            )
        out = target_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        if _sha256(data) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {rel}")
    return target_dir
