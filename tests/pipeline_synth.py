"""Synthetic AMLC-shaped data for pipeline tests (no real data). Train: US/India; test: US/India/France (unseen)."""
from __future__ import annotations

import random
from pathlib import Path

W = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo "
     "sierra tango acme global prime star metro royal sunrise orion vega nova").split()
SUF = {"US": ["inc", "llc", "corp", "store"], "India": ["pvt ltd", "traders", "enterprises"],
       "France": ["sarl", "sas", "société", "boulangerie"]}
ST = {"US": ["main street", "oak avenue", "park lane"], "India": ["mg road", "station road", "hill view"],
      "France": ["rue de la paix", "avenue des champs", "boulevard saint michel"]}
CITY = {"US": [("springfield", "IL", "62701"), ("riverside", "CA", "92501")],
        "India": [("bangalore", "Karnataka", "560001"), ("pune", "Maharashtra", "411001")],
        "France": [("paris", "", "75008"), ("lyon", "", "69002")]}
HDR = "entity_id\tbusiness_name\tbusiness_address\tcountry"


def _addr(rng, c):
    city, st, pc = rng.choice(CITY[c])
    if c == "France":
        return f"{rng.randint(1, 99)} {rng.choice(ST[c])}, {pc} {city}"
    return f"{rng.randint(1, 999)} {rng.choice(ST[c])}, {city}, {st} {pc}"


def _noisy(rng, s):
    t = s.split()
    if rng.random() < 0.3 and len(t) > 1:
        t = t[:-1]
    out = " ".join(t)
    return out.upper() if rng.random() < 0.5 else out


def _split(rng, n_s1, countries, id0, nid, prefix_ok=True):
    s1, s2, s3, gt = [], [], [], []
    for i in range(id0, id0 + n_s1):
        c = rng.choice(countries)
        name = f"{rng.choice(W)} {rng.choice(W)} {rng.choice(SUF[c])}"
        a = _addr(rng, c)
        s1.append((f"S1-{i}", name, a, c))
        links = []
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3, 5])):       # zero-, single- and multi-match S1s
            nid += 1
            tg = rng.choice(["S2", "S3"])
            a2 = a if rng.random() < 0.7 else ("" if rng.random() < 0.3 else _addr(rng, c))
            (s2 if tg == "S2" else s3).append((f"{tg}-{nid}", _noisy(rng, name), a2, c))
            links.append(f"{tg}-{nid}")
        gt.append((f"S1-{i}", ",".join(links)))
    for _ in range(n_s1):                                          # distractors
        nid += 1
        c = rng.choice(countries)
        tg = rng.choice(["S2", "S3"])
        (s2 if tg == "S2" else s3).append((f"{tg}-{nid}", f"{rng.choice(W)} {rng.choice(W)} {rng.choice(SUF[c])}", _addr(rng, c), c))
    return s1, s2, s3, gt, nid


def _write(p: Path, header, rows):
    p.write_text(header + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


DEVANAGARI = ["सेवन", "गैलक्सी", "प्रोजेक्ट्स", "सॉफ्टवेयर", "टेक्नोलॉजी", "इंडस्ट्रीज", "सर्विसेज", "ट्रेडर्स"]


def make(root: Path, n_train=3000, n_test=1500, seed=0, translit=0, train_as_test=False) -> dict:
    """Writes data/{train,test}_*.tsv and H0..H4 id lists; returns paths and sizes.
    translit: number of extra India train S1 whose true S2 record carries a Devanagari name (S1 = its romanization).
    train_as_test: test_* files are byte copies of train_* (for h4-vs-test feature parity on identical records)."""
    import sys
    rng = random.Random(seed)
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, gt, nid = _split(rng, n_train, ["US", "India"], 1, 100000)
    if translit:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from experiments.agy2_lane_romanization import romanize
        for k in range(translit):
            dev = " ".join(rng.sample(DEVANAGARI, 2))
            sid, nid = f"S1-{n_train + 1 + k}", nid + 1
            a_ = _addr(rng, "India")
            s1.append((sid, romanize(dev) + " pvt ltd", a_, "India"))
            s2.append((f"S2-{nid}", dev, a_, "India"))
            gt.append((sid, f"S2-{nid}"))
    for n, rows in (("train_source1", s1), ("train_source2", s2), ("train_source3", s3)):
        _write(d / f"{n}.tsv", HDR, rows)
    _write(d / "train_ground_truth.tsv", "source1_entity_id\tmatched_entity_ids", gt)
    if train_as_test:
        for n in ("source1", "source2", "source3"):
            (d / f"test_{n}.tsv").write_bytes((d / f"train_{n}.tsv").read_bytes())
        t1 = s1
    else:
        t1, t2, t3, _, _ = _split(rng, n_test, ["US", "India", "France"], 900000, 5000000)
        t1.append(("S1-999999", "zzqx unmatched entity", "", "France"))  # guaranteed empty candidate set candidate
        for n, rows in (("test_source1", t1), ("test_source2", t2), ("test_source3", t3)):
            _write(d / f"{n}.tsv", HDR, rows)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import experiments.agy2_exp5_corrected_holdout as X5
    from experiments.agy2_exp7_prep_h3 import salted_holdout
    ids = [r[0] for r in s1]
    r = ids[:]
    rng.shuffle(r)
    h0, h1 = r[:300], r[300:1100]
    X5.SALT = "amlc26-exp5-holdout-v1"
    h2 = X5.select_holdout(ids, set(h0) | set(h1), 400)
    h3 = salted_holdout(ids, set(h0) | set(h1) | set(h2), 300, "amlc26-h3-holdout-v1")
    h4 = salted_holdout(ids, set(h0) | set(h1) | set(h2) | set(h3), 300, "amlc26-h4-holdout-v1")
    files = {}
    for n, l in (("H0", h0), ("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4)):
        p = root / f"{n}.ids"
        p.write_text("\n".join(l) + "\n")
        files[n] = str(p)
    return {"data": d, "files": files, "n_test_s1": len(t1), "h3_n": len(h3)}
