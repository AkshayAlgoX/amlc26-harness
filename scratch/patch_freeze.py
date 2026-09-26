import re
text = open("experiments/agy2_pipeline_freeze.py").read()
text = text.replace(
    'cfg["matcher"]["features"] = [f for f in list(M_FEATS) + list(NEW_FEATURES) if f not in drop]',
    'from experiments.agy2_x2_features import R_FEATURES\n    cfg["matcher"]["features"] = [f for f in list(M_FEATS) + list(NEW_FEATURES) + R_FEATURES if f not in drop]'
)
open("experiments/agy2_pipeline_freeze.py", "w").write(text)
