"""
INTERIM KILL GATE -- mechanically evaluate the pre-registered thresholds in
docs/PREREGISTRATION.md against the Experiment 1 / 2 reports.

Out-of-sample incremental prediction is the decision rule. Conditional mutual
information is deliberately NOT used as the gate (estimator-sensitive).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

G1_MIN_DELTA_R2 = 0.010
G2_MIN_CONCORD = 0.020
G3_MIN_ACC = 0.55
G3_MIN_GAP = 0.03


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "A_pertok"
    suf = sys.argv[2] if len(sys.argv) > 2 else ""
    e1 = json.load(open(os.path.join(ROOT, "results", "exp1" + suf,
                                     "exp1_report.json")))
    e2p = os.path.join(ROOT, "results", "exp2" + suf, "exp2_report.json")
    e2 = json.load(open(e2p)) if os.path.exists(e2p) else {}

    t = e1[target]
    dr2_fixed = t["deltas"]["r2"]
    # Prefer refit inference when it exists.  Historical reports without it
    # remain readable, but new gates cannot silently fall back to a narrower
    # fixed-model test bootstrap.
    dr2 = e1.get(f"refit_boot_{target}", {}).get("r2", dr2_fixed)
    dcc = t["deltas"]["concordance"]
    ceil = e1.get(f"noise_ceiling_{target}", float("nan"))

    print(f"=== INTERIM KILL GATE  (target {target}) ===")
    print(f"label noise ceiling on R2: {ceil:.3f}")

    g1 = (dr2["mean"] >= G1_MIN_DELTA_R2) and dr2["ci_excludes_zero"]
    print(f"\nG1 incremental prediction")
    print(f"   Delta_R2 = {dr2['mean']:+.4f}  (>= {G1_MIN_DELTA_R2})   "
          f"95% CI [{dr2['ci_lo']:+.4f}, {dr2['ci_hi']:+.4f}] "
          f"{'excludes 0' if dr2['ci_excludes_zero'] else 'INCLUDES 0'}")
    print(f"   as a fraction of the noise ceiling: "
          f"{dr2['mean']/ceil if ceil else float('nan'):+.3f}")
    print(f"   -> {'PASS' if g1 else 'FAIL'}")

    per = dcc["per_seed"]
    g2 = (dcc["mean"] >= G2_MIN_CONCORD) and (
        all(x > 0 for x in per) or all(x < 0 for x in per))
    print(f"\nG2 candidate ranking")
    print(f"   Delta_concordance = {dcc['mean']:+.4f} (>= {G2_MIN_CONCORD}), "
          f"per-seed {[round(x,4) for x in per]}")
    print(f"   -> {'PASS' if g2 else 'FAIL'}")

    print(f"\nG3 matched candidates")
    if not e2:
        g3 = False
        print("   exp2 report missing -> FAIL")
    else:
        acc = e2.get("_primary_seed0", {}).get(
            "hidden_accuracy", e2.get("HIDDEN linear probe", {}).get("mean", float("nan")))
        gap = e2.get("_gap_hidden_minus_cheap", {})
        g3 = (acc >= G3_MIN_ACC and gap.get("mean", -1) >= G3_MIN_GAP
              and gap.get("ci_excludes_zero", False))
        print(f"   hidden matched-pair accuracy = {acc:.4f} (>= {G3_MIN_ACC})")
        print(f"   gap over cheap controls = {gap.get('mean', float('nan')):+.4f} "
              f"(>= {G3_MIN_GAP})  CI [{gap.get('ci_lo', float('nan')):+.4f}, "
              f"{gap.get('ci_hi', float('nan')):+.4f}] "
              f"{'excludes 0' if gap.get('ci_excludes_zero') else 'INCLUDES 0'}")
        print(f"   n pairs per seed: {e2.get('_n_pairs_per_seed')}")
        print(f"   -> {'PASS' if g3 else 'FAIL'}")

    heat = e1.get("heatmap", {})
    import numpy as np
    hm = np.array(heat.get("delta_r2", [[]]), dtype=float)
    if hm.size and np.isfinite(hm).any():
        # Use the layer chosen on VALIDATION, not the per-bin max over 13
        # layers: a max over 13 noisy cells is almost always positive, so the
        # old criterion was selection-biased and nearly free to pass.
        layers = e1.get(target, {}).get("best_layers", [])
        L = int(round(float(np.median(layers)))) if layers else int(hm.shape[0] // 2)
        L = min(max(L, 0), hm.shape[0] - 1)
        per_bin = hm[L]
        frac = float(np.mean(per_bin[np.isfinite(per_bin)] > 0))
        print(f"   (evaluated at the validation-selected layer {L}, "
              f"not the per-bin max over layers)")
    else:
        frac = float("nan")
        L = -1
    nat = t["natural_stratum"]
    nat_delta = nat["r2_cheap_hidden"] - nat["r2_cheap"]
    g4 = (frac >= 0.5) and (nat_delta > 0)
    print(f"\nG4 robustness")
    print(f"   timestep bins with Delta_R2 > 0 at the best layer: {frac:.2f} "
          f"(>= 0.50)")
    print(f"   natural (non-oversampled) stratum Delta_R2 = {nat_delta:+.4f} "
          f"(n={nat['n']})")
    print(f"   -> {'PASS' if g4 else 'FAIL'}")

    allp = g1 and g2 and g3 and g4
    print(f"\n{'='*58}")
    print("VERDICT:", "GATE PASSED -- proceed to Phase 0B"
          if allp else "GATE FAILED -- KILL the direction")
    print(f"{'='*58}")
    out = {"target": target, "G1": g1, "G2": g2, "G3": g3, "G4": g4,
           "passed": allp, "delta_r2": dr2, "delta_concordance": dcc,
           "noise_ceiling": ceil, "bins_positive_frac": frac,
           "natural_stratum_delta_r2": nat_delta}
    with open(os.path.join(ROOT, "results",
                           f"kill_gate_{target}{suf}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return 0 if allp else 1


if __name__ == "__main__":
    sys.exit(main())
