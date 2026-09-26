code = open("experiments/agy2_x2_features.py").read()
old = """    if not nonascii_mask.any():
        dcov, djac = _compute_dict_features(q_raw_names, t_raw_names)
    return kind, char3, jw, tokjacc, skeljacc_arr, skelcov_arr, dcov, djac"""
new = """    if not nonascii_mask.any():
        dcov, djac = _compute_dict_features(q_raw_names, t_raw_names)
        return kind, char3, jw, tokjacc, skeljacc_arr, skelcov_arr, dcov, djac"""
code = code.replace(old, new)
open("experiments/agy2_x2_features.py", "w").write(code)
