from kaggle.api.kaggle_api_extended import KaggleApi
import sys

api = KaggleApi()
api.authenticate()

# Push the h4 kernel
try:
    api.kernels_push("/tmp/x2-bundle-final/h4")
    print("SUCCESS: H4 kernel launched")
except Exception as e:
    print(f"FAILED: {e}")
