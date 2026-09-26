import re

code = open("experiments/agy2_pipeline.py").read()

# Remove flag_RROM5 from PR_FEATS
code = code.replace('"flag_R1", "flag_R5", "flag_RROM5", "flag_b2", "flag_b3", "flag_b4", "exact_name", "exact_addr", "country", "src_s3",', '"flag_R1", "flag_R5", "flag_b2", "flag_b3", "flag_b4", "exact_name", "exact_addr", "country", "src_s3",')

# Remove from bits unpacking
code = code.replace('("flag_R5", R5), ("flag_RROM5", RROM5), ("flag_b2", B2C)', '("flag_R5", R5), ("flag_b2", B2C)')

# Remove from num_channels
code = code.replace('P[["flag_A", "flag_B", "flag_F200", "flag_R5", "flag_RROM5", "flag_b2", "flag_b3", "flag_b4"]]', 'P[["flag_A", "flag_B", "flag_F200", "flag_R5", "flag_b2", "flag_b3", "flag_b4"]]')

# Remove from N generation
code = code.replace('("flag_A", "flag_B", "flag_F100", "flag_F200", "flag_R1", "flag_R5", "flag_RROM5", "flag_b2", "flag_b3", "flag_b4", "num_channels"):', '("flag_A", "flag_B", "flag_F100", "flag_F200", "flag_R1", "flag_R5", "flag_b2", "flag_b3", "flag_b4", "num_channels"):')

# Remove RROM5 retrieval logic
code = re.sub(r'# Romanization Reverse TF-IDF.*?texts\.clear\(\); codes\.clear\(\); rawnames\.clear\(\)', 'texts.clear(); codes.clear(); rawnames.clear()', code, flags=re.DOTALL)

# Remove Stage 2 block
code = re.sub(r'# ---------------- stage 4\.5: Stage-2 features.*?(?=\n    if MODE == "test":)', '', code, flags=re.DOTALL)

# Remove RROM5 retrieval bits array
code = code.replace('rrank = np.where((bits & (R5 | RROM5)) > 0, rka, 99).astype(np.int64)', 'rrank = np.where((bits & R5) > 0, rka, 99).astype(np.int64)')
code = code.replace('rsc = np.where((bits & (R5 | RROM5)) > 0, rsa, 0.0).astype(np.float32)', 'rsc = np.where((bits & R5) > 0, rsa, 0.0).astype(np.float32)')

open("experiments/agy2_pipeline.py", "w").write(code)
