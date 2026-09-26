import json, random, re, sys, time, unicodedata
from collections import Counter
sys.path.insert(0, "/home/akshaya/amlc26-agy2")
import duckdb, pandas as pd
from experiments.agy2_exp4_fresh_holdout_validation import clean_text, norm_address

T0 = time.time(); log = lambda m: print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
TRN = "/home/akshaya/Downloads/student_resource/student_resource/dataset/train/"
R2 = "/home/akshaya/amlc26-agy2/"
isbr = lambda s: any(0x0900 <= ord(c) <= 0x0D7F for c in s)
DEVTOK = re.compile(r"[^\s,.\-/()&:;#]+")
def dtoks(raw): return [unicodedata.normalize("NFC", t) for t in DEVTOK.findall(raw) if isbr(t)]

excl = set()
for f in ("exclude_ids/prev_val_s1_50k.ids", "exclude_ids/prev_val_s1_holdout_50k.ids", "expected/exp5_holdout_s1.ids", "expected/h3_s1.ids", "expected/h4_s1.ids"):
    excl |= {l.strip() for l in open(R2 + f) if l.strip()}

con = duckdb.connect(); con.execute("SET memory_limit='4GB'; SET threads=4")
log("Loading targets...")
# We don't have nonascii_targets.parquet, so we query directly
NA = con.execute(f"""
    select cast(substring(entity_id, 4) as bigint) * 4 + 2 as code, business_name as name, business_address as addr from '{TRN}train_source2.tsv'
    union all
    select cast(substring(entity_id, 4) as bigint) * 4 + 3 as code, business_name as name, business_address as addr from '{TRN}train_source3.tsv'
""").df()

log("Filtering Brahmic targets...")
bn = {c: n for c, n in zip(NA.code.values.tolist(), NA.name.values) if pd.notna(n) and isbr(str(n))}
ba = {c: a for c, a in zip(NA.code.values.tolist(), NA.addr.values) if pd.notna(a) and isbr(str(a))}
del NA

log("Reading ground truth...")
enc = lambda e: int(e[3:]) * 4 + (2 if e[1] == "2" else 3)
pn, pa = [], []
with open(TRN + "train_ground_truth.tsv") as fh:
    next(fh)
    for line in fh:
        s1, _, rest = line.rstrip("\n").partition("\t")
        if s1 in excl or not rest: continue
        for e in rest.split(","):
            c = enc(e)
            if c in bn: pn.append((s1, c))
            if c in ba: pa.append((s1, c))

log("Sampling and loading S1...")
rnd = random.Random(42); rnd.shuffle(pn); rnd.shuffle(pa); pn, pa = pn[:300_000], pa[:300_000]
need = {s for s, _ in pn} | {s for s, _ in pa}; S1 = {}
with open(TRN + "train_source1.tsv", encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        e, nm, ad, _ = line.rstrip("\n").split("\t")
        if e in need: S1[e] = (clean_text(nm), norm_address(ad))

def learn(pairs, raw, field, min_c=5):
    cd, cl, cdl = Counter(), Counter(), Counter()
    for s, c in pairs:
        D, L = set(dtoks(str(raw[c]))), set(S1[s][field].split())
        cd.update(D); cl.update(L)
        for d in D:
            for l in L: cdl[(d, l)] += 1
    best = {}
    for (d, l), v in cdl.items():
        if v >= min_c:
            x = 2 * v / (cd[d] + cl[l])
            if d not in best or x > best[d][1]: best[d] = (l, x)
    return {d: l for d, (l, _) in best.items()}, len(cd), sum(cd[d] for d in best) / max(sum(cd.values()), 1)

log("Learning dictionaries...")
DN, nn, mn = learn(pn, bn, 0)
DA, na_, ma = learn(pa, ba, 1)

log(f"name dict {len(DN):,}/{nn:,} tokens (mass {mn:.3f}); addr dict {len(DA):,}/{na_:,} tokens (mass {ma:.3f})")
json.dump(DN, open("experiments/x2_dict_name.json", "w"), ensure_ascii=False)
log("Saved experiments/x2_dict_name.json")
