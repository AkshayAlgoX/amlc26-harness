# AMLC26 — Fresh Holdout Retrieval Validation Report

**Date:** 2026-09-25 05:10:43 IST  
**Holdout Population:** 50,000 Source 1 Entities (`data/val_s1_holdout_50k.tsv`, stable SHA-256 hash, disjoint from prior sample)  
**Ground Truth Links:** 173,064 (789 S2, 172,275 S3)  
**Streaming Runtime:** 605.8s | **Peak RAM:** 6332.7 MB  

## 1. Fresh Holdout Population & Distribution Audit

| Attribute | Population Count | Share (%) |
| :--- | :--- | :--- |
| **Total Holdout S1 Entities** | **50,000** | 100.00% |
| Overlap with Previous 50k Sample | 0 | 0.00% (Strictly Disjoint) |
| US S1 Entities | 30,067 | 60.13% |
| India S1 Entities | 19,933 | 39.87% |
| **Total True Links** | **173,064** | 100.00% |
| Source 2 True Links | 789 | 0.46% |
| Source 3 True Links | 172,275 | 99.54% |

### Ground Truth Cardinality Distribution

| Cardinality Bucket | S1 Count | Share (%) | Oracle Macro F0.5 |
| :--- | :--- | :--- | :--- |
| `card=0 (singleton)` | 2,798 | 5.60% | 1.000000 |
| `card=1` | 2,632 | 5.26% | 0.965426 |
| `card=2` | 8,649 | 17.30% | 0.986357 |
| `card=3` | 11,809 | 23.62% | 0.989751 |
| `card=4` | 11,148 | 22.30% | 0.990654 |
| `card=5` | 7,272 | 14.54% | 0.991673 |
| `card>5` | 5,692 | 11.38% | 0.991762 |

## 2. Proposed Retrieval Configuration Performance

Configuration: `A (Exact Name) + B (Exact Address) + Forward TF-IDF@100 + Reverse TF-IDF@5 + B2 + B3_capped + B4 + Canon (Separate View)`

| Metric | Measured Value | Target / Baseline |
| :--- | :--- | :--- |
| **Unique Union Recall** | **96.8723%** (167,651/173,064) | Baseline: 96.924% |
| Source 2 Recall | **96.0710%** (758/789) | - |
| Source 3 Recall | **96.8759%** (166,893/172,275) | - |
| **Oracle Macro F0.5** | **0.989167** | **Target: >= 0.990000** |
| **Mean Candidates/S1** | **991.6** | **Target: <= 120.0** (Baseline: ~205.0) |
| p50 (Median) Candidates/S1 | 458 | - |
| p95 Candidates/S1 | **3357** | **Target: <= 300.0** |
| p99 Candidates/S1 | **9155** | **Target: <= 500.0** |
| Max Candidates/S1 | 166845 | - |
| Total Unique Candidate Pairs | 49,582,434 | Baseline: ~10,250,000 |
| True Links Missed | 5,413 (3.13%) | Baseline: 5,308 (3.08%) |

## 3. Marginal Contributions over `A+B + F@100 + R@5`

| Component Channel | Links Captured Standalone | Marginal Links Rescued | Marginal Recall Gain | Standalone Mean Candidates/S1 |
| :--- | :--- | :--- | :--- | :--- |
| **Base: A+B + F@100 + R@5** | 164,012 | Reference | 94.7696% | 771.5 |
| `+ B2` (City + Street Num) | 63,575 | +2,360 | +1.364% | 205.3 |
| `+ B3_capped` (Num + Rare Token, max 50) | 75,619 | +1,949 | +1.126% | 589.3 |
| `+ B4` (PIN + Rare Token) | 7,280 | +197 | +0.114% | 0.4 |
| `+ Canon View` (Learned, separate) | 30,095 | +12 | +0.007% | 11.6 |
| `Reverse@5` (over A+B+F@100) | 162,429 | +6,475 | +3.741% | ~23.4 |

## 4. Subgroup Evaluations

| Subgroup | Population Count | Recall (%) | Oracle Macro F0.5 | Mean Candidates/S1 |
| :--- | :--- | :--- | :--- | :--- |
| **US Subgroup** | 30,067 S1s (103,923 links) | 98.357% | 0.994805 | 778.6 |
| **India Subgroup** | 19,933 S1s (69,141 links) | 94.640% | 0.980662 | 1313.0 |
| **Empty-Address Target Links** | 7,712 target links | 78.605% | - | - |

## 5. Stop Condition Assessment

- **Oracle Macro F0.5 >= 0.990**: **FAIL** (Measured: **0.989167**)
- **Candidate Budget (Mean <= 120)**: **FAIL** (Measured: **991.6**)
- **Candidate Budget (p95 <= 300)**: **FAIL** (Measured: **3357**)
- **Candidate Budget (p99 <= 500)**: **FAIL** (Measured: **9155**)

> [!WARNING]
> Retrieval stop conditions not fully met under current budgeting. Residual misses are dominated by transliteration (Devanagari vs Latin) and missing addresses in Source 2/3.
