# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PENDING
**Team Members:** PENDING
**Submission Date:** PENDING

> Draft. Every number below comes from a logged run (file named). Items still marked PENDING (H4 validation,
> test inference) are filled in only after those runs. Nothing here is projected.

---

## 1. Executive Summary
The pipeline retrieves candidates with country-partitioned lexical channels, then narrows them with a learned
LightGBM pruner (Top-80 per Source 1 entity). A LightGBM pair matcher scores what remains, with features chosen
from error mining: rare-token and character name evidence, explicit missing-address handling, street-number and
PIN conflicts, and competition among an entity's candidates. The match set per entity is chosen by maximising
expected F0.5 on calibrated out-of-fold scores, then global one-to-one conflict resolution is applied.

On a clean, untouched holdout (H3) the earlier model scored macro F0.5 0.923461. The current model scores
0.939414 on H1 cross-validation. Its fresh-holdout (H4) score is PENDING.

## 2. Methodology

### 2.1 Problem Analysis
- Country is a hard partition: 100% of the 7,638,365 training links are same-country.
- Every S2/S3 record belongs to at most one S1 (no repeated target in the training ground truth).
- Singletons (entities with no match) are 5.5–5.8% of S1 and score 0/1.
- About 3.3% of true links share almost no retrievable signal (retrieval misses in H1: 5,743 of 172,636).
- The test set adds an unseen country, France (15.0% of test S1). Its addresses are accented (28.3% contain
  non-ASCII characters) and it has no training labels.

### 2.2 Solution Strategy
**Approach Type:** blocking (multi-channel retrieval) + learned pruner + pairwise classifier + expected-F0.5 set
decision.

**Core Innovation:** one fail-closed code path for validation and inference, and error-mining-driven matcher
features. B2 gained +0.0155 macro F0.5 on H1 out-of-fold, better in 5/5 folds.

## 3. Candidate Generation (Blocking)
- **Blocking keys used:**
  - exact normalised name;
  - exact normalised address;
  - forward TF-IDF top-200 (word tokens, per country);
  - reverse TF-IDF top-5 (each record against every S1 of its country);
  - city + street number;
  - street number + rare address tokens;
  - postcode + rare token.
  Keys that match more than 200 targets are frequency-capped.
- **Candidate pairs generated:** H1: 236 per S1 before pruning, 80 after. Test: PENDING.
- **How true matches were kept:** recall was measured per channel. H1 pool recall is 96.67% and the best-case
  (oracle) F0.5 at the Top-80 budget is 0.988. `candidate_pairs.tsv` holds exactly the pairs the matcher scores.

## 4. Matching Model
**Features used:**
- Name features:
  - Jaro-Winkler;
  - token Jaccard and containment (both directions);
  - IDF-weighted overlap, where IDF comes from source 2 and source 3 names;
  - character 3-gram cosine;
  - length ratio.
- Address features:
  - Jaro-Winkler, Jaccard, containment and digit overlap;
  - all of the above are set to missing when either address is empty;
  - street-number and postcode match or conflict (absence is separate from conflict).
- Other:
  - retrieval channel flags, ranks and scores;
  - the pruner's out-of-fold probability and rank;
  - within-entity competition: the gap to the entity's best name/address score, and the number of strong names.

**Model type:** LightGBM (binary). Pruner and matcher are cross-fitted on the same 5 folds grouped by S1.

**Threshold selection method:**
- Expected-F0.5 set rule: for each entity, keep the top-k candidates that maximise expected F0.5; predict nothing
  when an empty prediction has higher expected score.
- Its parameters were selected on H1 out-of-fold scores only.
- Then greedy one-to-one conflict resolution across all test entities.

## 5. Validation
- Validation is grouped by entity (5 folds). The development holdouts (H2, H3) were consumed one at a time.
  H4 is sealed until the final check.
- Results:
  - H3: 0.923461 for the earlier model;
  - H1 out-of-fold: 0.939414 for the current model;
  - H4: PENDING.
- France cannot be validated: it has no labels.

## 6. Other
- Only the provided data is used. There are no external lookups, APIs or pretrained models; LightGBM is
  MIT-licensed.
- Reproduction: see `code/business_entity_resolution/README.md`.
