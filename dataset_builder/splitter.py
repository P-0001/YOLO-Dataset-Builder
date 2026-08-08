from __future__ import annotations

import random
from typing import Any


def split_records(records: list[dict[str, Any]], splits: dict[str, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    total = len(ordered)
    train_end = round(total * float(splits["train"]))
    val_end = train_end + round(total * float(splits["val"]))
    return {"train": ordered[:train_end], "val": ordered[train_end:val_end], "test": ordered[val_end:]}
