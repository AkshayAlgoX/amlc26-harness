import sys, re
code = open("experiments/agy2_x2_pipeline.py").read()
code = code.replace("agy2_b2r_features", "agy2_x2_features")
open("experiments/agy2_x2_pipeline.py", "w").write(code)
