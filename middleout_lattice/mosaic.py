from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class MosaicMeta:
    word_size: int
    plane_order: list[int]
    order_mode: str
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


def _delta_prefix(data: bytes) -> bytes:
    out = bytearray()
    prev = 0
    for value in data:
        out.append((value - prev) & 0xFF)
        prev = value
    return bytes(out)


def _delta_restore(data: bytes) -> bytes:
    out = bytearray()
    prev = 0
    for value in data:
        original = (prev + value) & 0xFF
        out.append(original)
        prev = original
    return bytes(out)


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for value in data:
        counts[value] += 1
    total = len(data)
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _build_words(data: bytes, word_size: int) -> tuple[bytes, list[bytes], int]:
    pad_len = (-len(data)) % word_size
    padded = data + (b"\x00" * pad_len)
    word_count = len(padded) // word_size
    words = [padded[i * word_size : (i + 1) * word_size] for i in range(word_count)]
    return padded, words, pad_len


def _resolve_plane_order(words: list[bytes], word_size: int, order_mode: str) -> list[int]:
    if order_mode == "natural":
        return list(range(word_size))
    if order_mode == "reversed":
        return list(reversed(range(word_size)))

    plane_scores: list[tuple[float, int]] = []
    for plane_index in range(word_size):
        plane = bytes(word[plane_index] for word in words)
        plane_scores.append((_entropy(plane), plane_index))

    if order_mode == "entropy":
        plane_scores.sort(key=lambda item: (item[0], item[1]))
    elif order_mode == "reverse_entropy":
        plane_scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
    else:
        raise ValueError("order_mode must be one of natural, reversed, entropy, reverse_entropy")
    return [plane_index for _, plane_index in plane_scores]


def _apply_residual(data: bytes, residual: str) -> bytes:
    if residual == "none":
        return data
    if residual == "xor":
        return _xor_prefix(data)
    if residual == "delta":
        return _delta_prefix(data)
    raise ValueError("residual must be 'none', 'xor', or 'delta'")


def _restore_residual(data: bytes, residual: str) -> bytes:
    if residual == "none":
        return data
    if residual == "xor":
        return _xor_restore(data)
    if residual == "delta":
        return _delta_restore(data)
    raise ValueError("residual must be 'none', 'xor', or 'delta'")


def mosaic_encode(
    data: bytes,
    word_size: int,
    plane_order: Iterable[int] | None = None,
    residual: str = "xor",
    order_mode: str = "entropy",
) -> tuple[bytes, MosaicMeta]:
    if word_size < 2:
        raise ValueError("word_size must be at least 2")
    if residual not in {"none", "xor", "delta"}:
        raise ValueError("residual must be 'none', 'xor', or 'delta'")

    padded, words, pad_len = _build_words(data, word_size)
    if plane_order is None:
        plane_order = _resolve_plane_order(words, word_size, order_mode)
    else:
        plane_order = list(plane_order)
        if sorted(plane_order) != list(range(word_size)):
            raise ValueError("plane_order must be a permutation of word positions")
        order_mode = "custom"

    stream = bytearray()
    for plane_index in plane_order:
        plane = bytes(word[plane_index] for word in words)
        stream.extend(_apply_residual(plane, residual))

    meta = MosaicMeta(
        word_size=word_size,
        plane_order=plane_order,
        order_mode=order_mode,
        residual=residual,
        pad_len=pad_len,
        original_size=len(data),
        word_count=len(words),
    )
    return bytes(stream), meta


def mosaic_decode(stream: bytes, meta: MosaicMeta | dict[str, Any]) -> bytes:
    if isinstance(meta, dict):
        meta = MosaicMeta(
            word_size=int(meta["word_size"]),
            plane_order=[int(x) for x in meta["plane_order"]],
            order_mode=str(meta.get("order_mode", "custom")),
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
        planes[plane_index] = _restore_residual(plane, meta.residual)

    words = [bytearray(meta.word_size) for _ in range(meta.word_count)]
    for plane_index, plane in planes.items():
        for i, value in enumerate(plane):
            words[i][plane_index] = value

    padded = b"".join(bytes(word) for word in words)
    return padded[: meta.original_size]


def mosaic_candidates(data: bytes) -> list[tuple[bytes, MosaicMeta]]:
    candidates: list[tuple[bytes, MosaicMeta]] = []
    if len(data) < 4:
        return candidates

    for word_size in (2, 4, 8, 16):
        if len(data) < word_size * 2:
            continue
        for order_mode in ("natural", "reversed", "entropy", "reverse_entropy"):
            for residual in ("none", "xor", "delta"):
                encoded, meta = mosaic_encode(
                    data,
                    word_size=word_size,
                    plane_order=None,
                    residual=residual,
                    order_mode=order_mode,
                )
                candidates.append((encoded, meta))
    return candidates
