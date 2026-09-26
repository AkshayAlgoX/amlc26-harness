import json, hashlib
cfg = json.load(open("experiments/configs/h4_frozen.json"))
s1 = hashlib.sha256(open("experiments/agy2_pipeline.py", "rb").read()).hexdigest()
s2 = hashlib.sha256(open("experiments/agy2_pipeline_freeze.py", "rb").read()).hexdigest()
cfg["code_sha256"]["experiments/agy2_pipeline.py"] = s1
cfg["code_sha256"]["experiments/agy2_pipeline_freeze.py"] = s2

if "experiments/agy2_x2_features.py" not in cfg["code_sha256"]:
    cfg["code_sha256"]["experiments/agy2_x2_features.py"] = hashlib.sha256(open("experiments/agy2_x2_features.py", "rb").read()).hexdigest()

json.dump(cfg, open("experiments/configs/h4_frozen.json", "w"), indent=2)
