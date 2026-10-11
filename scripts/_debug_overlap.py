"""Debug why the cross-split overlap is 50% rather than 100%."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import pandas as pd  # noqa: E402

from test_unsw_benchmark import make_partition  # noqa: E402
from cyberllm.detection.unsw import row_hashes  # noqa: E402

train = make_partition(n_per_class=40, seed=1)
test = make_partition(n_per_class=20, seed=1)
dropped = {"attack_cat", "id", "label"}
keep = [c for c in train.columns if c not in dropped]
print("keep:", keep)

# Which columns actually differ between matching (class, i) rows?
for i in (0, 1, 5):
    for ci in (0, 1):
        a = train[(train["attack_cat"] == train["attack_cat"].unique()[ci])
                  & (train["id"] == ci * 40 + i + 1)]
        b = test[(test["attack_cat"] == test["attack_cat"].unique()[ci])
                 & (test["id"] == ci * 20 + i + 1)]
        if a.empty or b.empty:
            print(f"missing row i={i} ci={ci}")
            continue
        diffs = [c for c in keep if str(a.iloc[0][c]) != str(b.iloc[0][c])]
        print(f"i={i} ci={ci} diffs={diffs}")

th = pd.Index(pd.unique(row_hashes(train, keep)))
overlap = th.isin(row_hashes(test, keep))
print("overlap:", int(overlap.sum()), "/", len(test))
