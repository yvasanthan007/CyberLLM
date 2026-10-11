"""Time each benchmark stage so a slow step is attributable, not guessed."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from cyberllm.detection.unsw import (  # noqa: E402
    audit_partition, audit_split, conversation_key, row_hashes,
)

base = Path(r"d:\CyberLLM\data\raw\unsw_nb15")


def timed(label, function, *args, **kwargs):
    started = time.monotonic()
    value = function(*args, **kwargs)
    print(f"{label:<38} {time.monotonic() - started:8.2f}s")
    return value


frames = {}
for role, name in (("train", "UNSW_NB15_training-set.csv"),
                   ("test", "UNSW_NB15_testing-set.csv")):
    frame, audit = timed(f"audit_partition[{role}]", audit_partition, base / name)
    print(f"    rows={audit.rows} dup_all={audit.duplicate_rows} "
          f"dup_feat={audit.duplicate_feature_rows} bom={audit.has_bom}")
    frames[role] = frame

target = "attack_cat"
dropped = ["attack_cat", "id", "label"]
features = [c for c in frames["train"].columns if c not in dropped]

timed("row_hashes(pool)", row_hashes,
      pd.concat([frames["train"], frames["test"]], ignore_index=True), features)
timed("audit_split(publisher)", audit_split, frames["train"], frames["test"],
      target_column=target, dropped=dropped)
timed("conversation_key(pool)", conversation_key,
      pd.concat([frames["train"], frames["test"]], ignore_index=True),
      target, dropped)
print("done")
