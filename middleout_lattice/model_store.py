from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .archive import ArchiveManifest, FileRecord, BlockMeta, compress_directory, decompress_directory, decompress_file_bytes


@dataclass
class CompressedModelStore:
    manifest_path: Path
    storage_root: Path
    source_root: Path

    @classmethod
    def from_source(cls, source_dir: Path, target_dir: Path, block_size: int = 1 << 20) -> "CompressedModelStore":
        manifest_path = compress_directory(source_dir, target_dir, block_size=block_size)
        return cls.from_manifest(manifest_path)

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "CompressedModelStore":
        manifest = json.loads(manifest_path.read_text())
        return cls(
            manifest_path=manifest_path,
            storage_root=manifest_path.parent,
            source_root=Path(manifest["root"]),
        )

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

    def files(self) -> list[Path]:
        return [Path(entry["path"]) for entry in self.manifest["files"]]

    def file_record(self, rel_path: str | Path) -> dict[str, Any]:
        rel = str(Path(rel_path))
        for entry in self.manifest["files"]:
            if entry["path"] == rel:
                return entry
        raise FileNotFoundError(rel)

    def read_bytes(self, rel_path: str | Path) -> bytes:
        entry = self.file_record(rel_path)
        stored = self.storage_root / entry["storage_path"]
        if entry["mode"] == "raw":
            return stored.read_bytes()
        return decompress_file_bytes(
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

    def materialize(self, target_dir: Path, paths: Sequence[str | Path] | None = None) -> Path:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if paths is None:
            decompress_directory(self.manifest_path, target_dir)
            return target_dir
        for rel_path in paths:
            rel = Path(rel_path)
            out = target_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self.read_bytes(rel))
        return target_dir

    @contextmanager
    def open_materialized(self, paths: Sequence[str | Path] | None = None) -> Iterator[Path]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="middleout-lattice-"))
        try:
            yield self.materialize(tmp_dir, paths=paths)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def summary(self) -> dict[str, Any]:
        manifest = self.manifest
        files = manifest["files"]
        original_bytes = sum(int(entry["original_bytes"]) for entry in files)
        stored_bytes = sum(int(entry["stored_bytes"]) for entry in files)
        raw_files = sum(1 for entry in files if entry["mode"] == "raw")
        return {
            "files": len(files),
            "raw_files": raw_files,
            "compressed_files": len(files) - raw_files,
            "original_bytes": original_bytes,
            "stored_bytes": stored_bytes,
            "ratio": round(original_bytes / max(1, stored_bytes), 4),
            "source_root": str(self.source_root),
        }
