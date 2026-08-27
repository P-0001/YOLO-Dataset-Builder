from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from .utils import Progress, map_progress

_PHASH_SIZE = 8
_PHASH_DCT = 32
_NUM_BLOCKS = 8
_BLOCK_BITS = 8
_BLOCK_MASK = (1 << _BLOCK_BITS) - 1


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _dct_basis(size: int, keep: int) -> tuple[tuple[float, ...], ...]:
    """Pre-compute the first ``keep`` rows of the orthonormal DCT-II basis."""
    basis: list[tuple[float, ...]] = []
    for row in range(keep):
        scale = math.sqrt(1.0 / size) if row == 0 else math.sqrt(2.0 / size)
        basis.append(
            tuple(
                scale * math.cos(math.pi * row * (2 * col + 1) / (2 * size))
                for col in range(size)
            )
        )
    return tuple(basis)


def _perceptual_hash(path: Path) -> tuple[int, tuple[int, int, int]]:
    """Compute a 64-bit DCT-based pHash plus the mean RGB colour of the image.

    pHash captures frequency structure rather than raw brightness, making it
    far less prone to false-positive matches than the average hash it
    replaces.  The mean-RGB value is retained as a secondary guard so that
    uniformly coloured but structurally different images are not collapsed.
    """
    with Image.open(path) as image:
        gray = image.resize((_PHASH_DCT, _PHASH_DCT)).convert("L")
        small_rgb = image.resize((8, 8)).convert("RGB")
    pixels = list(gray.getdata())
    rgb_pixels = list(small_rgb.getdata())

    basis = _dct_basis(_PHASH_DCT, _PHASH_SIZE)

    # 2-D DCT, keeping only the top-left _PHASH_SIZE x _PHASH_SIZE block.
    # D[0:k, 0:k] = basis @ image @ basis.T
    temp: list[float] = [0.0] * (_PHASH_SIZE * _PHASH_DCT)
    for row in range(_PHASH_SIZE):
        basis_row = basis[row]
        for col in range(_PHASH_DCT):
            total = 0.0
            for inner in range(_PHASH_DCT):
                total += basis_row[inner] * pixels[inner * _PHASH_DCT + col]
            temp[row * _PHASH_DCT + col] = total

    coeffs: list[float] = [0.0] * (_PHASH_SIZE * _PHASH_SIZE)
    for row in range(_PHASH_SIZE):
        for col in range(_PHASH_SIZE):
            basis_col = basis[col]
            total = 0.0
            for inner in range(_PHASH_DCT):
                total += temp[row * _PHASH_DCT + inner] * basis_col[inner]
            coeffs[row * _PHASH_SIZE + col] = total

    # Exclude the DC term (index 0) when computing the median so that overall
    # brightness does not skew the threshold.
    sorted_coeffs = sorted(coeffs[1:])
    median = (sorted_coeffs[31] + sorted_coeffs[32]) / 2
    phash = sum(1 << index for index, value in enumerate(coeffs) if value > median)

    average_rgb = tuple(
        round(sum(pixel[channel] for pixel in rgb_pixels) / len(rgb_pixels))
        for channel in range(3)
    )
    return phash, average_rgb


def _is_near_duplicate(
    first: tuple[int, tuple[int, int, int]],
    second: tuple[int, tuple[int, int, int]],
    threshold: int,
) -> bool:
    # The mean-colour guard avoids treating unrelated uniformly coloured images
    # as duplicates: their pHash values are otherwise identical.
    return (first[0] ^ second[0]).bit_count() <= threshold and max(
        abs(a - b) for a, b in zip(first[1], second[1])
    ) <= 16


def find_duplicates(
    records: list[dict[str, Any]],
    threshold: int,
    workers: int = 0,
    progress: Progress | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep the first sorted record in each exact/near duplicate group."""
    valid = [record for record in records if not record["error"]]
    max_workers = workers or min(32, max(1, (len(valid) // 100) + 1))
    paths = [record["path"] for record in valid]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        exact_hashes = map_progress(
            executor, _hash_file, paths, progress, "Hashing files"
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for record, digest in zip(valid, exact_hashes):
        groups.setdefault(digest, []).append(record)
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, str]] = []
    for group in groups.values():
        keeper = next(
            (item for item in group if item["path"].with_suffix(".txt").exists()),
            group[0],
        )
        kept.append(keeper)
        for duplicate in group:
            if duplicate is keeper:
                continue
            duplicates.append(
                {
                    "duplicate": str(duplicate["path"]),
                    "kept": str(keeper["path"]),
                    "kind": "exact",
                }
            )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        perceptual = map_progress(
            executor,
            _perceptual_hash,
            [record["path"] for record in kept],
            progress,
            "Comparing images",
        )
    final: list[dict[str, Any]] = []
    final_hashes: list[tuple[int, tuple[int, int, int]]] = []
    # If two 64-bit hashes differ in at most T bits, at least one of their
    # eight 8-bit blocks is identical (pigeonhole: T <= 7 < 8 blocks).  Indexing
    # those blocks avoids O(n^2) scans while guaranteeing no missed matches for
    # any supported threshold.
    buckets: dict[tuple[int, int], list[int]] = {}
    for record, image_hash in zip(kept, perceptual):
        candidate_indexes: set[int] = set()
        for block in range(_NUM_BLOCKS):
            key = (block, (image_hash[0] >> (block * _BLOCK_BITS)) & _BLOCK_MASK)
            candidate_indexes.update(buckets.get(key, []))
        match = next(
            (
                index
                for index in candidate_indexes
                if _is_near_duplicate(image_hash, final_hashes[index], threshold)
            ),
            None,
        )
        if match is None:
            index = len(final)
            final.append(record)
            final_hashes.append(image_hash)
            for block in range(_NUM_BLOCKS):
                key = (block, (image_hash[0] >> (block * _BLOCK_BITS)) & _BLOCK_MASK)
                buckets.setdefault(key, []).append(index)
        else:
            old = final[match]
            if (
                record["path"].with_suffix(".txt").exists()
                and not old["path"].with_suffix(".txt").exists()
            ):
                final[match] = record
                duplicates.append(
                    {
                        "duplicate": str(old["path"]),
                        "kept": str(record["path"]),
                        "kind": "visual",
                    }
                )
            else:
                duplicates.append(
                    {
                        "duplicate": str(record["path"]),
                        "kept": str(old["path"]),
                        "kind": "visual",
                    }
                )
    return final, duplicates
