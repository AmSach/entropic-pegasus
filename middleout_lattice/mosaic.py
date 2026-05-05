from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class MosaicMeta:
    word_size: int
    plane_order: list[int]
    residual: str
    pad_len: int
    original_size: int
    word_count: int


def _xor_prefix(data: bytes) -> bytes:
    out = bytearray()
    prev = 0
    for value in data:
        residual = value ^ prev
        out.append(residual)
        prev = value
    return bytes(out)


def _xor_restore(data: bytes) -> bytes:
    out = bytearray()
    prev = 0
    for value in data:
        original = value ^ prev
        out.append(original)
        prev = original
    return bytes(out)


def mosaic_encode(
    data: bytes,
    word_size: int,
    plane_order: Iterable[int] | None = None,
    residual: str = "xor",
) -> tuple[bytes, MosaicMeta]:
    if word_size < 2:
        raise ValueError("word_size must be at least 2")
    if plane_order is None:
        plane_order = list(reversed(range(word_size)))
    plane_order = list(plane_order)
    if sorted(plane_order) != list(range(word_size)):
        raise ValueError("plane_order must be a permutation of word positions")
    if residual not in {"xor", "none"}:
        raise ValueError("residual must be 'xor' or 'none'")

    pad_len = (-len(data)) % word_size
    padded = data + (b"\x00" * pad_len)
    word_count = len(padded) // word_size
    words = [padded[i * word_size : (i + 1) * word_size] for i in range(word_count)]

    stream = bytearray()
    for plane_index in plane_order:
        plane = bytes(word[plane_index] for word in words)
        if residual == "xor":
            plane = _xor_prefix(plane)
        stream.extend(plane)

    meta = MosaicMeta(
        word_size=word_size,
        plane_order=plane_order,
        residual=residual,
        pad_len=pad_len,
        original_size=len(data),
        word_count=word_count,
    )
    return bytes(stream), meta


def mosaic_decode(stream: bytes, meta: MosaicMeta | dict[str, Any]) -> bytes:
    if isinstance(meta, dict):
        meta = MosaicMeta(
            word_size=int(meta["word_size"]),
            plane_order=[int(x) for x in meta["plane_order"]],
            residual=str(meta["residual"]),
            pad_len=int(meta["pad_len"]),
            original_size=int(meta["original_size"]),
            word_count=int(meta["word_count"]),
        )

    if meta.word_size < 2:
        raise ValueError("word_size must be at least 2")
    if len(stream) != meta.word_count * meta.word_size:
        raise ValueError("stream length does not match mosaic metadata")

    plane_len = meta.word_count
    offset = 0
    planes: dict[int, bytes] = {}
    for plane_index in meta.plane_order:
        plane = bytes(stream[offset : offset + plane_len])
        offset += plane_len
        if meta.residual == "xor":
            plane = _xor_restore(plane)
        planes[plane_index] = plane

    words = [bytearray(meta.word_size) for _ in range(meta.word_count)]
    for plane_index, plane in planes.items():
        for i, value in enumerate(plane):
            words[i][plane_index] = value

    padded = b"".join(bytes(word) for word in words)
    return padded[: meta.original_size]


def mosaic_candidates(data: bytes) -> list[tuple[bytes, MosaicMeta]]:
    candidates: list[tuple[bytes, MosaicMeta]] = []
    for word_size in (2, 4, 8):
        if len(data) < word_size * 2:
            continue
        for plane_order in (list(range(word_size)), list(reversed(range(word_size)))):
            for residual in ("none", "xor"):
                encoded, meta = mosaic_encode(data, word_size=word_size, plane_order=plane_order, residual=residual)
                candidates.append((encoded, meta))
    return candidates
