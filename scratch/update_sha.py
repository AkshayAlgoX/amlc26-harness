import json, hashlib
cfg = json.load(open("experiments/configs/h4_x2_frozen.json"))
s1 = hashlib.sha256(open("experiments/agy2_pipeline.py", "rb").read()).hexdigest()
s2 = hashlib.sha256(open("experiments/agy2_x2_features.py", "rb").read()).hexdigest()
cfg["code_sha256"]["experiments/agy2_pipeline.py"] = s1
cfg["code_sha256"]["experiments/agy2_x2_features.py"] = s2
json.dump(cfg, open("experiments/configs/h4_x2_frozen.json", "w"), indent=2)
