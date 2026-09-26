code = open("experiments/agy2_pipeline_freeze.py").read()
code = "from experiments.agy2_x2_features import R_FEATURES\n" + code
open("experiments/agy2_pipeline_freeze.py", "w").write(code)
