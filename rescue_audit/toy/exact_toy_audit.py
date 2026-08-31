"""Exact finite-state audit for Q, V, A, CRN, RB, and horizon accounting.

The environment has three masked positions and a binary vocabulary.  Its
conditional token probabilities depend on the complete visible state, so a
forced first action changes later values.  The state space is small enough to
enumerate every future position order and token outcome exactly.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

MASK = -1


def probs(state, pos):
    """P(token=0/1 | state,pos), deliberately relational and state dependent."""
    visible = sum((i + 1) * (2 * x - 1) for i, x in enumerate(state) if x != MASK)
    logit1 = 0.45 * (pos - 1) + 0.55 * visible + 0.35 * (state[(pos + 1) % 3] == 1)
    p1 = 1.0 / (1.0 + math.exp(-logit1))
    return np.array([1.0 - p1, p1], dtype=np.float64)


def rb_score(state, pos):
    p = probs(state, pos)
    return float(np.sum(p * np.log(p)))


def enumerate_policy(state, n_commit):
    """Enumerate uniform remaining-position permutations and token draws."""
    masked = [i for i, x in enumerate(state) if x == MASK]
    h = min(n_commit, len(masked))
    out = []
    for order in itertools.permutations(masked, h):
        p_order = 1.0 / math.perm(len(masked), h)

        def rec(cur, t, prob, mc_sum, rb_sum, trace):
            if t == h:
                out.append((prob * p_order, mc_sum, rb_sum, tuple(cur), tuple(trace)))
                return
            pos = order[t]
            p = probs(cur, pos)
            r = rb_score(cur, pos)
            for tok in (0, 1):
                nxt = list(cur); nxt[pos] = tok
                rec(nxt, t + 1, prob * p[tok], mc_sum + math.log(p[tok]),
                    rb_sum + r, trace + [(pos, tok)])

        rec(list(state), 0, 1.0, 0.0, 0.0, [])
    return out


def exact_v(state, n_commit):
    paths = enumerate_policy(state, n_commit)
    return {
        "mc": sum(w * mc for w, mc, _, _, _ in paths),
        "rb": sum(w * rb for w, _, rb, _, _ in paths),
        "weight": sum(w for w, *_ in paths),
    }


def exact_q(state, action, horizon):
    pos, tok = action
    p = probs(state, pos)
    forced = math.log(p[tok])
    nxt = list(state); nxt[pos] = tok
    fut = exact_v(tuple(nxt), horizon)
    return {"mc": forced + fut["mc"], "rb": forced + fut["rb"],
            "forced": forced}


def exact_advantage(state, action, horizon):
    q = exact_q(state, action, horizon)
    v = exact_v(state, horizon + 1)
    # Expected first V score by direct enumeration of the first commit.
    v_first = exact_v(state, 1)["rb"]
    return {
        "Q": q["rb"] / (horizon + 1),
        "V": v["rb"] / (horizon + 1),
        "A_full": (q["rb"] - v["rb"]) / (horizon + 1),
        "A_future": ((q["rb"] - q["forced"]) - (v["rb"] - v_first)) / horizon,
        "V_first": v_first,
    }


def sample_policy(state, n_commit, rng, order=None, uniforms=None):
    cur = list(state)
    masked = [i for i, x in enumerate(cur) if x == MASK]
    if order is None:
        order = list(rng.permutation(masked))
    mc = rb = first_rb = 0.0
    trace = []
    for t, pos in enumerate(order[:n_commit]):
        p = probs(cur, pos)
        r = rb_score(cur, pos)
        u = rng.random() if uniforms is None else uniforms[pos]
        tok = int(u >= p[0])
        mc += math.log(p[tok]); rb += r
        if t == 0:
            first_rb = r
        cur[pos] = tok; trace.append((pos, tok))
    return {"mc": mc, "rb": rb, "first_rb": first_rb,
            "state": tuple(cur), "trace": trace}


def monte_carlo_advantage(state, action, horizon, n=30_000, seed=7, crn=True):
    vals = np.empty(n)
    for k in range(n):
        rng = np.random.default_rng(seed + k)
        # Counter-style per-position random utilities are shared by branches.
        order_keys = rng.random(len(state))
        token_u = rng.random(len(state))
        v_order = sorted([i for i, x in enumerate(state) if x == MASK],
                         key=lambda i: order_keys[i], reverse=True)
        v = sample_policy(state, horizon + 1, rng, v_order, token_u)
        pos, tok = action
        q0 = list(state); q0[pos] = tok
        q_order = [i for i in v_order if i != pos]
        q = sample_policy(tuple(q0), horizon, rng, q_order,
                          token_u if crn else None)
        vals[k] = (math.log(probs(state, pos)[tok]) + q["rb"] - v["rb"]) / (horizon + 1)
    return float(vals.mean()), float(vals.var(ddof=1))


def main():
    state = (MASK, MASK, MASK)
    action = (1, 1)
    horizon = 2
    v = exact_v(state, horizon + 1)
    adv = exact_advantage(state, action, horizon)
    assert abs(v["weight"] - 1.0) < 1e-12
    assert abs(v["mc"] - v["rb"]) < 1e-12, (v["mc"], v["rb"])
    assert abs(adv["A_full"] - (adv["Q"] - adv["V"])) < 1e-12

    mc_mean, var_crn = monte_carlo_advantage(state, action, horizon, crn=True)
    ind_mean, var_ind = monte_carlo_advantage(state, action, horizon, crn=False)
    assert abs(mc_mean - adv["A_full"]) < 5e-3
    assert abs(ind_mean - adv["A_full"]) < 5e-3

    # No-op: force the exact action that the V branch took, then share future
    # order/token uniforms.  The completed states and scores must coincide.
    rng = np.random.default_rng(123)
    keys, us = rng.random(3), rng.random(3)
    order = sorted(range(3), key=lambda i: keys[i], reverse=True)
    first = order[0]
    p = probs(state, first); tok = int(us[first] >= p[0])
    vpath = sample_policy(state, 3, rng, order, us)
    q0 = list(state); q0[first] = tok
    qpath = sample_policy(tuple(q0), 2, rng, order[1:], us)
    q_total = math.log(p[tok]) + qpath["mc"]
    assert qpath["state"] == vpath["state"]
    assert abs(q_total - vpath["mc"]) < 1e-12

    result = {
        "status": "VALIDATED",
        "state": state,
        "action": action,
        "horizon": horizon,
        "exact": adv,
        "mc_mean_crn": mc_mean,
        "mc_mean_independent": ind_mean,
        "variance_crn": var_crn,
        "variance_independent": var_ind,
        "rb_equals_expected_realized_pathll": bool(abs(v["mc"] - v["rb"]) < 1e-12),
        "branch_commit_counts": {"Q": horizon + 1, "V": horizon + 1},
        "noop_identical": True,
    }
    out = Path(__file__).resolve().parents[1] / "results" / "exact_toy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
