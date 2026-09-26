from kaggle.api.kaggle_api_extended import KaggleApi
import sys
api = KaggleApi()
api.authenticate()
try:
    api.kernels_push("/tmp/x2-bundle-fixed/h4")
    print("SUCCESS: FIXED H4 kernel launched")
except Exception as e:
    print(f"FAILED: {e}")
