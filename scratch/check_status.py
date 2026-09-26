from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
try:
    status = api.kernels_status("akshayalgox/amlc26-x2-h4")
    print(f"Status of amlc26-x2-h4: {status}")
except Exception as e:
    print(f"Error checking status: {e}")
