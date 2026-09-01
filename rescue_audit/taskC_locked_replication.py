"""Locked replication of the positive taskA/taskB task-utility signal.

The configuration was documented before taskC was complete: Nemotron layer 12,
train-only PCA-32 candidate hidden state, pairwise linear objective, A_task.
We repeat it on three fixed document splits and add objective-matched cheap and
cheap+hidden controls.  No layer, feature family, or test result is selected.
"""
import argparse
import json
import os

import numpy as np

from rlib import metrics as M
from rlib import probes2 as P
from rlib import rdata as RD
from rlib import screen as SC


def score(y, pred, sid):
    _, groups = M.group_slices(sid)
    return {
        "within_r2": M.within_state_r2(y, pred, sid, groups),
        "concordance": M.concordance(y, pred, sid, groups),
        **M.topk_metrics(y, pred, sid, groups),
    }


def pairwise_linear(prep, y, sid, keys, seeds, epochs):
    feats = {key: prep["pca"][key] for key in keys}
    runner = P._Runner(
        {k: feats[k]["train"] for k in keys}, y["train"], sid["train"],
        {k: feats[k]["val"] for k in keys}, y["val"], sid["val"],
        {k: feats[k]["test"] for k in keys}, loss_kind="pairwise")
    dims = [feats[k]["train"].shape[1] for k in keys]
    selector = P.make_selector("within_r2", sid["val"])
    return runner.run(lambda: P.MLPProbe(dims, (), keys=tuple(keys)),
                      selector, seeds=seeds, epochs=epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taskC")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "results", "taskC_locked_replication.json"))
    args = ap.parse_args()

    d = RD.load_labels([args.tag])
    rep = {"status": "locked replication of a pre-taskC configuration",
           "config": vars(args), "results": []}
    for split_seed in range(args.splits):
        sp = RD.doc_splits(d, split_seed)
        RD.check_split_disjoint(d, sp)
        prep = SC.prepare(d, sp, args.layer, pca_dim=args.pca_dim,
                          groups=RD.state_groups(d["state_id"])[1])
        y = SC.split_targets(d, sp, "A_task")
        sid = SC.split_sid(d, sp)
        models = {
            "pair_hidden": pairwise_linear(
                prep, y, sid, ("hi",), (0, 1, 2), args.epochs),
            "pair_cheap": pairwise_linear(
                prep, y, sid, ("cheap",), (0, 1, 2), args.epochs),
            "pair_cheap_hidden": pairwise_linear(
                prep, y, sid, ("cheap", "hi"), (0, 1, 2), args.epochs),
        }
        row = {"split_seed": split_seed, "n_test_docs": int(
            len(np.unique(d["doc_id"][sp["test"]]))), "models": {}}
        for name, model in models.items():
            row["models"][name] = {
                "metrics": score(y["test"], model["pred_test"], sid["test"]),
                "val_score": model["val_score"], "hp": model["hp"]}
        for name in ("pair_hidden", "pair_cheap_hidden"):
            row[f"delta_{name}_vs_pair_cheap"] = {
                k: row["models"][name]["metrics"][k]
                - row["models"]["pair_cheap"]["metrics"][k]
                for k in ("within_r2", "concordance", "top1")}
        rep["results"].append(row)
        print(split_seed, row[f"delta_pair_hidden_vs_pair_cheap"],
              row[f"delta_pair_cheap_hidden_vs_pair_cheap"], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
