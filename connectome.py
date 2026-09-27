"""Load and rewire the small MaleCNS circuit used by this experiment."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_circuit():
    return pd.read_csv(ROOT / "data/neurons.csv"), pd.read_csv(ROOT / "data/edges.csv")


def rewire(edges, neurons, seed):
    """Swap directed edges within type pairs while preserving in-degree."""
    rng = np.random.default_rng(seed)
    result = edges.copy()
    source = result.source.to_numpy().copy()
    target = result.target.to_numpy().copy()
    type_ids = neurons.type_index.to_numpy()
    groups = {}
    for index, (src, dst) in enumerate(zip(source, target)):
        if src != dst:
            groups.setdefault((type_ids[src], type_ids[dst]), []).append(index)
    occupied = set(zip(source, target))
    swaps = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        for _ in range(20 * len(group)):
            i, j = rng.choice(group, 2, replace=False)
            a, b, c, d = source[i], target[i], source[j], target[j]
            if a == c or b == d or a == d or c == b or (a, d) in occupied or (c, b) in occupied:
                continue
            occupied.remove((a, b))
            occupied.remove((c, d))
            occupied.update(((a, d), (c, b)))
            target[i], target[j] = d, b
            swaps += 1
    result["target"] = target
    result["body_post"] = neurons.bodyId.to_numpy()[target]
    assert len(occupied) == len(result)
    assert np.array_equal(np.bincount(target, minlength=len(neurons)),
                          np.bincount(edges.target, minlength=len(neurons)))
    assert np.array_equal(type_ids[target], type_ids[edges.target.to_numpy()])
    changed = 1 - len(occupied & set(zip(edges.source, edges.target))) / len(result)
    return result, {"accepted_swaps": swaps, "fraction_edges_changed": changed}
