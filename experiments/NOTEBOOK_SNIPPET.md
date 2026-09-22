# For teammates working in Colab / Kaggle / any notebook

Train however you like (PyTorch, HF Trainer, Unsloth, LLaMA-Factory...). Only three things are fixed:
use `data/folds.csv` for the split, save predictions in original target units, and register the
result with `log_external`. That's the whole contract.

## 1. Setup cell

```python
!git clone https://github.com/<captain>/<repo>.git amlc && cd amlc && pip -q install -r requirements.txt
%cd amlc
# put train.csv / test.csv / sample_test_out.csv in data/raw/ (or symlink from /kaggle/input/...)

import os
# Colab: mount the team's shared Drive folder so your outputs land where the captain can blend them
# from google.colab import drive; drive.mount("/content/drive")
# os.environ["AMLC_ARTIFACTS"] = "/content/drive/MyDrive/amlc26-artifacts"
```

## 2. Use the shared folds

```python
import pandas as pd
from src.data import train_with_folds, load_test
from src.config import col

train = train_with_folds()          # has a 'fold' column from data/folds.csv
test = load_test()
ID, TARGET = col("id"), col("target")

oof_parts, test_parts = [], []
for k in sorted(train["fold"].unique()):
    trn, val = train[train.fold != k], train[train.fold == k]
    # ... fit on trn, predict val and test ...
    # val_pred / test_pred must be in ORIGINAL units (expm1 if you trained on log1p)
    oof_parts.append(pd.DataFrame({"id": val[ID], "pred": val_pred}))
    test_parts.append(test_pred)
```

## 3. Register it (validates + scores + writes artifacts)

```python
import numpy as np
from src.registry import log_external

oof_df = pd.concat(oof_parts)
test_df = pd.DataFrame({"id": test[ID], "pred": np.mean(test_parts, axis=0)})
log_external("tm1-deep-001-clip336", oof=oof_df, test=test_df,
             notes="CLIP-L/14@336 + DeBERTa fusion, log target, Huber, 5 folds x 3 ep", minutes=95)
```

It refuses to save if an id is missing/duplicated or a prediction is NaN, prints your CV with the
official metric, and writes `oof/`, `test/` and `logs/` files. If you're on Kaggle (no Drive mount),
download those three files from `artifacts/` and drop them into the shared folder.

Naming: `<owner>-<track>-<NNN>[-tag]` — e.g. `tm1-deep-002-clip336-ft`. Never reuse a number.
