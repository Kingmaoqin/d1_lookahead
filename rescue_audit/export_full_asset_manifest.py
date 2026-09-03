"""Create a checksum inventory and lightweight cards for all experiment assets."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rescue_audit" / "results" / "full_inventory"
ROOTS = ("data", "results", "logs", "rescue_audit/results",
         "rescue_audit/logs", "rescue_audit/sealed_taskD_primary")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def npy_header(stream):
    version = np.lib.format.read_magic(stream)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = reader(stream)
    return {"shape": list(shape), "dtype": str(dtype),
            "fortran_order": bool(fortran)}


def schema(path: Path):
    if path.suffix == ".npy":
        with path.open("rb") as f:
            return {path.stem: npy_header(f)}
    ans = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if member.endswith(".npy"):
                with archive.open(member) as f:
                    ans[Path(member).stem] = npy_header(f)
    return ans


def scalar_statistics(path: Path, sch):
    if path.suffix != ".npz":
        return {}
    ans = {}
    with np.load(path, allow_pickle=True) as z:
        for name, meta in sch.items():
            if len(meta["shape"]) > 2 or np.prod(meta["shape"], dtype=np.int64) > 1_000_000:
                continue
            a = z[name]
            if not np.issubdtype(a.dtype, np.number) or a.size == 0:
                continue
            x = np.asarray(a, dtype=np.float64)
            finite = np.isfinite(x)
            ans[name] = {
                "finite_fraction": float(finite.mean()),
                "mean": float(x[finite].mean()) if finite.any() else None,
                "std": float(x[finite].std()) if finite.any() else None,
                "min": float(x[finite].min()) if finite.any() else None,
                "max": float(x[finite].max()) if finite.any() else None,
            }
    return ans


def category(rel: str) -> str:
    if rel.startswith("data/labels_"):
        return "raw_label_batch"
    if rel.startswith("rescue_audit/sealed_taskD_primary/"):
        return "sealed_probe_model"
    if rel.startswith("rescue_audit/results/") or rel.startswith("results/"):
        return "derived_result"
    if rel.startswith("logs/") or rel.startswith("rescue_audit/logs/"):
        return "run_log"
    return "data_asset"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    card_root = OUT / "cards"
    card_root.mkdir(parents=True, exist_ok=True)
    files = set()
    for root in ROOTS:
        base = ROOT / root
        if base.exists():
            files.update(p for p in base.rglob("*") if p.is_file())
    files = sorted(p for p in files if OUT not in p.parents and
                   "__pycache__" not in p.parts and p.suffix != ".pyc")
    rows = []
    for path in files:
        rel = str(path.relative_to(ROOT))
        row = {"path": rel, "category": category(rel), "bytes": path.stat().st_size,
               "sha256": sha256(path), "suffix": path.suffix.lower(),
               "git_lfs": path.suffix.lower() in (".npz", ".npy")}
        rows.append(row)
        if path.suffix.lower() in (".npz", ".npy"):
            sch = schema(path)
            card = {**row, "arrays": sch,
                    "scalar_statistics": scalar_statistics(path, sch)}
            target = card_root / Path(rel).with_suffix(".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as f:
                json.dump(card, f, indent=2, sort_keys=True)
    with (OUT / "asset_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    summary = {
        "n_assets": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
        "by_category": {k: {"files": sum(r["category"] == k for r in rows),
                            "bytes": sum(r["bytes"] for r in rows if r["category"] == k)}
                        for k in sorted({r["category"] for r in rows})},
        "by_suffix": dict(Counter(r["suffix"] or "[none]" for r in rows)),
        "lfs_objects": sum(r["git_lfs"] for r in rows),
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
