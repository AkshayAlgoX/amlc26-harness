"""AMLC26 Stage-2 Competition Model (Label-Blind Target-Side Features).
PREPARE ONLY. Not to be launched until EXP7 completes.

Loads existing X2 pair scores, computes target-side competition features,
trains a Stage 2 LightGBM model, and evaluates locally on H1.
"""
import sys
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_store as S
from src.er.metric import macro_f05
from src.er.ids import decode_target_id

def compute_competition_features(df: pd.DataFrame, score_col='p'):
    """
    Computes target-side competition features without labels.
    Input df must contain ['q', 'code', 'fold', score_col]
    """
    # Group by target (code)
    # We want features for a specific (q, code) pair:
    # - best competing S1 score (max p of OTHER q's for this code)
    # - this S1's rank among claimants (rank of p for this code)
    # - number of S1 claimants above thresholds
    # - score margin vs best competing S1
    # - target-side density
    
    # Sort by code and score descending
    df = df.sort_values(['code', score_col], ascending=[True, False]).reset_index()
    df['claimant_rank'] = df.groupby('code').cumcount() + 1
    
    # Compute max score per code
    code_max = df.groupby('code')[score_col].transform('max')
    # If the current row IS the max, the "best competing" is the 2nd best.
    # To get 2nd best efficiently:
    code_2nd_max = df.groupby('code')[score_col].nth(1).reset_index()
    # Handle codes with only 1 claimant
    df['best_competing_p'] = np.where(
        df['claimant_rank'] == 1,
        df['code'].map(code_2nd_max.set_index('code')[score_col]).fillna(0.0),
        code_max
    )
    
    df['margin_vs_best_competing'] = df[score_col] - df['best_competing_p']
    df['density_above_01'] = df.groupby('code')[score_col].transform(lambda x: (x > 0.1).sum())
    df['density_above_05'] = df.groupby('code')[score_col].transform(lambda x: (x > 0.5).sum())
    df['density_above_09'] = df.groupby('code')[score_col].transform(lambda x: (x > 0.9).sum())
    df['total_claimants'] = df.groupby('code')['q'].transform('count')
    
    # Restore original order
    df = df.set_index('index').sort_index()
    return df


def lgb_crossfit_stage2(df, features, params):
    """Cross-fit a Stage 2 LightGBM model."""
    scores = np.zeros(len(df), dtype=np.float32)
    models = []
    
    # StratifiedGroupKFold on S1 group
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    X = df[features]
    y = df['label']
    groups = df['q']
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        trn_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=trn_data)
        
        clf = lgb.train(
            params,
            trn_data,
            num_boost_round=1000,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        scores[val_idx] = clf.predict(X.iloc[val_idx], num_iteration=clf.best_iteration)
        models.append(clf)
    
    return scores, models

def decode_expected_f05_one_to_one(df, score_col='p_stage2'):
    """Re-use the existing Expected-F0.5 and one-to-one machinery."""
    # To properly evaluate, we apply the threshold logic per q.
    # We can just simulate the greedy decoding.
    
    df = df.sort_values(['q', score_col], ascending=[True, False])
    m_val = 0.1173
    s_val = 0.5
    
    # Apply expected F0.5 rule
    preds = {}
    for q, group in df.groupby('q'):
        group = group[group[score_col] >= s_val]
        p = group[score_col].values
        if len(p) == 0:
            continue
        cumsum_p = np.cumsum(p)
        k = np.arange(1, len(p) + 1)
        expF = 1.25 * cumsum_p / (k + 0.25 * (np.sum(p) + m_val))
        best_k = np.argmax(expF) + 1
        preds[q] = group['code'].values[:best_k].tolist()
    
    # Flatten and do one-to-one greedy resolution
    # Sort all predicted pairs globally by score_col
    flat_preds = []
    for q, codes in preds.items():
        # Get scores for these codes
        group = df[df['q'] == q].set_index('code')
        for c in codes:
            flat_preds.append({'q': q, 'code': c, 'p': group.loc[c, score_col]})
            
    if not flat_preds:
        return {}
        
    flat_df = pd.DataFrame(flat_preds).sort_values('p', ascending=False)
    
    resolved_preds = {}
    seen_codes = set()
    for _, row in flat_df.iterrows():
        q, c = int(row['q']), int(row['code'])
        if c not in seen_codes:
            seen_codes.add(c)
            resolved_preds.setdefault(q, set()).add(c)
            
    return resolved_preds

import argparse
from src.er.loader import load_source_tsv

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Competition Model")
    parser.add_argument("--scores-parquet", type=str, required=True, help="Path to X2 scores parquet")
    parser.add_argument("--queries-parquet", type=str, required=True, help="Path to H1 queries parquet")
    parser.add_argument("--truth-parquet", type=str, required=True, help="Path to H1 truth parquet")
    args = parser.parse_args()
    
    print(f"Loading data from {args.scores_parquet}...")
    df = pd.read_parquet(args.scores_parquet)
    # We expect 'q', 'code', 'p', 'label' to be in df. If 'label' is not, we join with truth.
    
    queries = pd.read_parquet(args.queries_parquet)
    truth = pd.read_parquet(args.truth_parquet)
    
    # Merge label if necessary
    if 'label' not in df.columns:
        # Create a set of true (q, code) pairs
        truth_set = set(zip(truth['q'], truth['code']))
        df['label'] = df.apply(lambda r: int((r['q'], r['code']) in truth_set), axis=1)

    print("Computing competition features...")
    df = compute_competition_features(df, score_col='p')
    
    features = [
        'p', 'claimant_rank', 'best_competing_p', 
        'margin_vs_best_competing', 'density_above_01', 
        'density_above_05', 'density_above_09', 'total_claimants'
    ]
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print("Training Stage 2 Model...")
    df['p_stage2'], models = lgb_crossfit_stage2(df, features, params)
    
    print("Decoding and resolving one-to-one...")
    preds = decode_expected_f05_one_to_one(df, score_col='p_stage2')
    
    print("Evaluating...")
    # Group truth by q
    truth_map = truth.groupby('q')['code'].apply(set).to_dict()
    # Add empty sets for queries with no truth
    for q in queries['q']:
        if q not in truth_map:
            truth_map[q] = set()
            
    # S1_ids for mapping back
    q_to_s1 = dict(zip(queries['q'], queries['s1_id']))
    
    # Map predictions and truth to s1_id string sets for macro_f05
    y_true = {}
    for q, codes in truth_map.items():
        y_true[q_to_s1[q]] = {decode_target_id(c) for c in codes}
        
    y_pred = {}
    for q, codes in preds.items():
        y_pred[q_to_s1[q]] = {decode_target_id(c) for c in codes}
        
    score = macro_f05(y_true, y_pred)
    print(f"Stage 2 H1 Macro F0.5: {score:.6f}")

if __name__ == "__main__":
    main()
