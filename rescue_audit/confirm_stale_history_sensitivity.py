"""Post-hoc D1 sensitivity: remove the two legacy one-state-late C2 columns.

This does not replace or modify the preregistered result.  It checks whether
the old->fresh transfer result was inflated by a known feature-definition shift:
old a3/b3 columns C2[8:10] are one state late, while freshA was collected after
the history-alignment repair.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import confirm_fresh as C
from rlib import rdata as RD


def clean_cheap(d):
    return np.concatenate([d["C1"], d["C2"][:, :8], d["C2"][:, 10:], d["C3"]],
                          axis=1).astype(np.float32)


def main():
    old = RD.load_labels(["a3", "b3"])
    fresh = RD.load_labels(["freshA"])
    sp = RD.doc_splits(old, seed=0)
    RD.check_split_disjoint(old, sp)

    original = RD.cheap_block
    RD.cheap_block = clean_cheap
    try:
        idx = np.arange(len(fresh[C.TARGET]))
        blocks = C.build_blocks(old, sp["train"], sp["val"], fresh, idx,
                                C.LAYER, C.PCA_D)
        y = {"train": old[C.TARGET][sp["train"]],
             "val": old[C.TARGET][sp["val"]],
             "test": fresh[C.TARGET]}
        sid = {"train": old["state_id"][sp["train"]],
               "val": old["state_id"][sp["val"]],
               "test": fresh["state_id"]}
        fam = C.fit_family(blocks, y, sid, probe_seeds=3, epochs=400)
        rep = C.evaluate(fam, y["test"], sid["test"], fresh["doc_id"],
                         n_perm=10000, n_boot=2000, tag="D1_no_stale_C2")
    finally:
        RD.cheap_block = original

    out = os.path.join(HERE, "results",
                       "CONFIRMATORY_freshA_posthoc_no_stale_C2.json")
    payload = {
        "status": "post-hoc sensitivity; does not replace preregistered D1",
        "removed_columns": ["C2.flip_count", "C2.persistence"],
        "reason": "legacy a3/b3 values are shifted one state; freshA values are fixed",
        "result": rep,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(out)


if __name__ == "__main__":
    main()
