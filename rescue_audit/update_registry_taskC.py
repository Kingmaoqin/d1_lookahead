"""Idempotently append Task-C exploratory/replication results to registry."""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "EXPERIMENT_REGISTRY.csv")
FIELDS = ["arm", "target", "split_seed", "criterion", "layer", "probe",
          "hg_kind", "r2", "within_r2", "concordance", "top1",
          "regret_norm_mean", "kendall_tau", "val_score", "n_params",
          "date", "phase", "output_path"]


def metric_row(arm, seed, layer, probe, metrics, phase, path, criterion):
    return {"arm": arm, "target": "A_task", "split_seed": seed,
            "criterion": criterion, "layer": layer, "probe": probe,
            "hg_kind": "hg", "r2": metrics.get("r2", ""),
            "within_r2": metrics.get("within_r2", ""),
            "concordance": metrics.get("pairwise_concordance",
                                        metrics.get("concordance", "")),
            "top1": metrics.get("top1_accuracy", metrics.get("top1", "")),
            "regret_norm_mean": metrics.get("normalized_regret",
                                              metrics.get("regret_norm_mean", "")),
            "kendall_tau": metrics.get("kendall_tau_state_mean", ""),
            "val_score": "", "n_params": "", "date": "2026-09-01",
            "phase": phase, "output_path": path}


def main():
    rows = []
    rel = "rescue_audit/results/screen_taskC_all_quick/report.json"
    d = json.load(open(os.path.join(os.path.dirname(HERE), rel)))
    for x in d["results"]:
        rows.append(metric_row("Nemotron_taskC", x["seed"], x["layer"],
                               x["probe"], x["metrics"],
                               "taskC_exploratory", rel, "broad_quick"))
    for tag, arm in (("taskD", "Nemotron_taskD_GSM8K"),
                     ("taskE_svamp", "Nemotron_taskE_SVAMP")):
        rel = f"rescue_audit/results/screen_{tag}_all_quick/report.json"
        full = os.path.join(os.path.dirname(HERE), rel)
        if not os.path.exists(full):
            continue
        d = json.load(open(full))
        for x in d["results"]:
            row = metric_row(arm, x["seed"], x["layer"], x["probe"],
                             x["metrics"], f"{tag}_exploratory", rel,
                             "broad_quick")
            row["date"] = "2026-09-02"
            rows.append(row)
    rel = "rescue_audit/results/taskC_locked_replication.json"
    d = json.load(open(os.path.join(os.path.dirname(HERE), rel)))
    for x in d["results"]:
        for name, model in x["models"].items():
            rows.append(metric_row("Nemotron_taskC", x["split_seed"], 12,
                                   name, model["metrics"],
                                   "taskC_locked_replication", rel,
                                   "pairwise_matched"))
    for fn in ("taskC_crossfit_positive_exact.json",
               "taskC_crossfit_positive_exact_seedmatch.json"):
        rel = "rescue_audit/results/" + fn
        d = json.load(open(os.path.join(os.path.dirname(HERE), rel)))
        for name, metrics in d["overall"].items():
            rows.append(metric_row("Nemotron_taskC", "OOF5", "val_selected",
                                   name, metrics, "taskC_posthoc_crossfit", rel,
                                   "fivefold_oof"))
    for fn, arm, phase in (
            ("taskCD_crossfit_confirmatory.json", "Nemotron_taskCD_GSM8K",
             "taskCD_posthoc_crossfit"),
            ("taskE_svamp_crossfit.json", "Nemotron_taskE_SVAMP",
             "taskE_posthoc_crossfit")):
        rel = "rescue_audit/results/" + fn
        full = os.path.join(os.path.dirname(HERE), rel)
        if not os.path.exists(full):
            continue
        d = json.load(open(full))
        for name, metrics in d["overall"].items():
            row = metric_row(arm, "OOF5", "val_selected", name, metrics,
                             phase, rel, "fivefold_oof")
            row["date"] = "2026-09-02"
            rows.append(row)
    old = list(csv.DictReader(open(REG)))
    keys = {(r["arm"], r["split_seed"], r["probe"], r["output_path"])
            for r in old}
    add = [r for r in rows if (str(r["arm"]), str(r["split_seed"]),
                               str(r["probe"]), str(r["output_path"])) not in keys]
    with open(REG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writerows(add)
    print("appended", len(add), "rows")


if __name__ == "__main__":
    main()
