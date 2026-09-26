import numpy as np
import pandas as pd
import duckdb

print("Loading data...")
dump_dir = "output/exp7_h1_dump_check/exp7_dump/"
b2_path = "output/exp9_kernel_check/exp9_store/c5_matcher/c163e30c76d2e9a6a3cd/scores.parquet"

P_base = duckdb.query(f"SELECT * FROM '{dump_dir}h1_matcher_oof.parquet'").df()
P_b2 = duckdb.query(f"SELECT p FROM '{b2_path}'").df()

# Sort exactly as in the pipeline before assignment
P_base = P_base.sort_values(["q", "prune_rank"], kind="stable").reset_index(drop=True)
P_base["p"] = P_b2["p"].values
P = P_base
print(f"Assigned {len(P)} rows directly after sorting")

Qd = duckdb.query(f"SELECT * FROM '{dump_dir}h1_queries.parquet'").df()
truth_n = Qd.truth_n.values
cty = Qd.country.values
nq = len(Qd)
fold = Qd.fold.values

def prep(D):
    D = D.sort_values(["q", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    g = D.groupby("q", sort=False)
    D["r"] = g.cumcount().values + 1
    D["mx"] = g.p.transform("max").values
    D["cs"] = g.p.cumsum().values
    D["tot"] = g.p.transform("sum").values
    D["_l"] = np.log(np.clip(1 - D.p.values, 1e-9, 1))
    D["e0"] = np.exp(D.groupby("q", sort=False)._l.transform("sum").values)
    return D

def decide(D, rule, prm):
    r, p = D.r.values, D.p.values
    if rule == "gap_new":
        return (p >= prm["s"]) & (p >= D.mx.values - prm["g"]) & (r <= prm["m"])
    
    ef = 1.25 * D.cs.values / (r + 0.25 * (D.tot.values + prm["m"]))
    best = pd.Series(ef).groupby(D.q.values).transform("max").values
    kstar = pd.Series(np.where(ef >= best, r, 10**9)).groupby(D.q.values).transform("min").values
    return (r <= kstar) & (best > D.e0.values) & (D.mx.values >= prm["s"])

def constrain(D, sel):
    idx = np.flatnonzero(sel)
    o = np.lexsort((-D.p.values[idx], D.code.values[idx]))
    cs = D.code.values[idx][o]
    first = np.r_[True, cs[1:] != cs[:-1]]
    out = np.zeros(len(D), bool); out[idx[o][first]] = True
    return out

D = prep(pd.DataFrame({"q": P.q.values, "code": P.code.values, "label": P.label.values.astype(bool), "p": P.p.values}))

def f05(sel):
    k = np.bincount(D.q.values[sel], minlength=nq).astype(float)
    tp = np.bincount(D.q.values[sel & D.label.values], minlength=nq).astype(float)
    f = np.where(truth_n == 0, (k == 0) * 1.0, np.where(k == 0, 0.0, 1.25 * tp / np.maximum(k + 0.25 * truth_n, 1e-12)))
    return f, k, tp

anchor_base = decide(D, "expF", {"m": 0.11734, "s": 0.6})
anchor_sel = constrain(D, anchor_base)
f_anchor, k_anchor, tp_anchor = f05(anchor_sel)

card = lambda f, lo, hi: float(f[(truth_n >= lo) & (truth_n <= hi)].mean())

print(f"Actual exact B2 anchor macro: {f_anchor.mean():.16f}")
print(f"Anchor card1: {card(f_anchor, 1, 1):.6f}")
print(f"Anchor card2-5: {card(f_anchor, 2, 5):.6f}")
print(f"Anchor card0: {card(f_anchor, 0, 0):.6f}")

anchor_per_fold = [float(f_anchor[fold == j].mean()) for j in range(5)]

results = []
s_vals = [0.4, 0.5, 0.6]
gamma_vals = [0.10, 0.15, 0.20, 0.25, 0.30]
max_cands_vals = [1, 2, 3, 5]

print("Running grid...")
best_sc = -1
best_r = None
best_f = None
for s in s_vals:
    for gamma in gamma_vals:
        for mc in max_cands_vals:
            sel = decide(D, "gap_new", {"s": s, "g": gamma, "m": mc})
            sel = constrain(D, sel)
            
            f, k, tp = f05(sel)
            sc = f.mean()
            
            sc_1 = card(f, 1, 1)
            sc_2_5 = card(f, 2, 5)
            sc_0 = card(f, 0, 0)
            
            pf = [float(f[fold == j].mean()) for j in range(5)]
            folds_up = sum(1 for nj in range(5) if pf[nj] > anchor_per_fold[nj])
            
            res = {
                "s": s, "gamma": gamma, "max_cands": mc,
                "macro": sc, "delta": sc - f_anchor.mean(),
                "card1": sc_1, "c1_delta": sc_1 - card(f_anchor, 1, 1),
                "card2_5": sc_2_5, "c25_delta": sc_2_5 - card(f_anchor, 2, 5),
                "card0": sc_0, "c0_delta": sc_0 - card(f_anchor, 0, 0),
                "folds_up": folds_up,
                "pf": pf
            }
            results.append(res)
            if sc > best_sc:
                best_sc = sc
                best_r = res
                best_f = f.copy()

df = pd.DataFrame(results).sort_values("macro", ascending=False)
print("\nTop 10 combinations:")
print(df.head(10).drop(columns=['pf']).to_string(index=False))

print("\nBest Parameters:")
res_copy = best_r.copy()
res_copy.pop('pf')
print(res_copy)

print("\nPer-fold scores:")
print(best_r['pf'])

print("\nChecks:")
print(f"Overall >= 0.9400: {best_sc >= 0.9400}")
print(f"Multi-match (2-5) gain >= 0.0020: {best_r['c25_delta'] >= 0.0020}")
print(f"Zero 1-match loss: {best_r['c1_delta'] >= -1e-9}")

# Check any case where 1-match loses score
loses = df[df.c1_delta < 0]
print("\nAny case where 1-match loses score:")
if len(loses) > 0:
    print(f"Yes, {len(loses)} combinations lose 1-match score.")
else:
    print("No")

