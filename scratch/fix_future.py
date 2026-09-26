code = open("experiments/agy2_pipeline_freeze.py").read()
code = code.replace("from experiments.agy2_x2_features import R_FEATURES\n", "")
code = code.replace("from experiments import agy2_exp9_features as F9", "from experiments import agy2_exp9_features as F9\nfrom experiments.agy2_x2_features import R_FEATURES")
open("experiments/agy2_pipeline_freeze.py", "w").write(code)
