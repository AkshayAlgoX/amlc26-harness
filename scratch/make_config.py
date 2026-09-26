import json

base = json.load(open("experiments/configs/h4_frozen.json"))

# Extend features to 69
extra_feats = [
    "t_name_kind",
    "fx_name_char3",
    "fx_name_jw",
    "fx_name_tokjacc",
    "fx_name_skeljacc",
    "fx_name_skelcov",
    "d_name_cov",
    "d_name_jac"
]
base["matcher"]["features"].extend(extra_feats)
base["matcher"]["best_iters"] = [986, 985, 844, 893, 995]

json.dump(base, open("experiments/configs/h4_x2_frozen.json", "w"), indent=2)
