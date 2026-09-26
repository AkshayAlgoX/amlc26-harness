import re
text = open("experiments/agy2_exp9_h4_freeze.py").read()
text = text.replace(
    'cfg["matcher"]["features"] = list(M_FEATS) + extra',
    'from experiments.agy2_x2_features import R_FEATURES\n    cfg["matcher"]["features"] = list(M_FEATS) + extra + R_FEATURES'
)
open("experiments/agy2_exp9_h4_freeze.py", "w").write(text)
