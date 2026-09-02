"""Export auditable per-shard cards and compact trend tables for Task C/D/E."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("taskC", "taskD", "taskE_svamp")
SCALARS = ("A_task", "A_task_sem", "Q_reward", "V_reward", "A_pertok",
           "A_sem", "V_pertok", "logp_action", "mask_ratio")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def npz_schema(path: Path):
    """Read embedded NPY headers without inflating multi-megabyte arrays."""
    ans = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as f:
                version = np.lib.format.read_magic(f)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(f)
            ans[Path(member).stem] = {"shape": list(shape), "dtype": str(dtype)}
    return ans


def states(doc, step):
    return len(set(zip(map(int, doc), map(int, step))))


def moments(a):
    x = np.asarray(a, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max())}


def add_group(store, key, z):
    x = {name: np.asarray(z[name], dtype=np.float64) for name in SCALARS}
    x["doc_id"] = np.asarray(z["doc_id"])
    x["step"] = np.asarray(z["step"])
    for name, value in x.items():
        store[key][name].append(value)


def group_row(tag, kind, value, chunks):
    a = {k: np.concatenate(v) for k, v in chunks.items()}
    y = a["A_task"]
    return {
        "tag": tag, "group": kind, "value": value,
        "n_rows": len(y), "n_docs": len(np.unique(a["doc_id"])),
        "n_states": states(a["doc_id"], a["step"]),
        "A_task_mean": float(y.mean()), "A_task_std": float(y.std()),
        "A_task_nonzero_frac": float((np.abs(y) > 1e-12).mean()),
        "A_task_positive_frac": float((y > 0).mean()),
        "A_task_negative_frac": float((y < 0).mean()),
        **{f"{k}_mean": float(a[k].mean()) for k in
           ("Q_reward", "V_reward", "A_pertok", "mask_ratio")},
    }


def main():
    out = ROOT / "rescue_audit" / "results" / "batch_exports"
    cards = out / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    catalog, trends, dataset_summary = [], [], {}

    for tag in TAGS:
        source = ROOT / "data" / f"labels_{tag}"
        files = sorted(source.glob("shard*.npz"))
        grouped = defaultdict(lambda: defaultdict(list))
        totals = defaultdict(lambda: defaultdict(list))
        tag_cards = cards / tag
        tag_cards.mkdir(parents=True, exist_ok=True)
        all_docs, total_rows, total_states = set(), 0, 0

        for path in files:
            with np.load(path, allow_pickle=True) as z:
                doc, step = z["doc_id"], z["step"]
                ndoc, nstate, nrow = len(np.unique(doc)), states(doc, step), len(doc)
                all_docs.update(map(int, np.unique(doc)))
                total_rows += nrow; total_states += nstate
                card = {
                    "tag": tag, "batch": path.stem,
                    "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                    "sha256": digest(path), "n_rows": nrow, "n_docs": ndoc,
                    "n_states": nstate, "doc_id_min": int(doc.min()),
                    "doc_id_max": int(doc.max()),
                    "prompt_row_min": int(z["prompt_row"].min()),
                    "prompt_row_max": int(z["prompt_row"].max()),
                    "prompt_stratum_counts": {
                        str(int(k)): int(v) for k, v in zip(
                            *np.unique(z["prompt_stratum"], return_counts=True))},
                    "arrays": npz_schema(path),
                    "statistics": {k: moments(z[k]) for k in SCALARS},
                }
                with (tag_cards / f"{path.stem}.json").open("w") as f:
                    json.dump(card, f, indent=2, sort_keys=True)
                catalog.append({
                    "tag": tag, "batch": path.stem, "path": card["path"],
                    "bytes": card["bytes"], "sha256": card["sha256"],
                    "n_docs": ndoc, "n_states": nstate, "n_rows": nrow,
                    "doc_id_min": card["doc_id_min"], "doc_id_max": card["doc_id_max"],
                    **{f"{k}_{q}": card["statistics"][k][q]
                       for k in SCALARS for q in ("mean", "std")},
                })
                for s in np.unique(step):
                    keep = step == s
                    add_group(grouped, ("step", int(s)), {k: z[k][keep] for k in
                              (*SCALARS, "doc_id", "step")})
                for s in np.unique(z["prompt_stratum"]):
                    keep = z["prompt_stratum"] == s
                    add_group(grouped, ("prompt_stratum", int(s)), {k: z[k][keep] for k in
                              (*SCALARS, "doc_id", "step")})
                add_group(totals, "all", {k: z[k] for k in (*SCALARS, "doc_id", "step")})

        for (kind, value), chunks in sorted(grouped.items()):
            trends.append(group_row(tag, kind, value, chunks))
        total = group_row(tag, "dataset", "all", totals["all"])
        dataset_summary[tag] = {
            **total, "n_shards": len(files), "n_docs": len(all_docs),
            "n_states": total_states, "n_rows": total_rows,
            "total_bytes": int(sum(p.stat().st_size for p in files)),
            "doc_id_min": min(all_docs), "doc_id_max": max(all_docs),
        }

    def write_csv(path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    write_csv(out / "batch_catalog.csv", catalog)
    write_csv(out / "trend_summary.csv", trends)
    with (out / "dataset_summary.json").open("w") as f:
        json.dump(dataset_summary, f, indent=2, sort_keys=True)
    print(json.dumps({"cards": len(catalog), "datasets": dataset_summary,
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
