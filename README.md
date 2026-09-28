# Amazon ML Challenge 2026: Business Entity Resolution

## 1. Problem
This repository contains the solution developed for the Amazon ML Challenge 2026. The objective of the competition is large-scale business entity resolution. The task requires linking entities from multiple distinct sources by identifying which records refer to the same real-world business, balancing recall and precision rigorously to maximize the Macro F0.5 evaluation metric limit.

## 2. Input Sources
The data schema operates primarily over three fragmented data repositories:
- Source 1 (S1)
- Source 2 (S2)
- Source 3 (S3)

## 3. Candidate-Generation / Blocking Approach
To handle the scale of 1.73M source entities without encountering memory constraints, the approach utilizes a memory-safe dual-pass constraint mapping backed by a local SQLite engine. It employs strict blocking keys (including country matching, compact text deduplication, and token intersection) to generate candidate pairs efficiently, isolating highly probable matches and preventing combinatorial explosions.

## 4. Feature Engineering
The system extracts comprehensive similarity signals utilizing a 17-dimensional feature space, including:
- Name Features: `rapidfuzz.fuzz.ratio`, `wratio`, `token_set_ratio`, `token_sort_ratio`, length metrics, and Jaccard intersections.
- Address Features: `compact_addr`, fuzzy ratios, token-set comparisons, and length metric disparities.
- Contextual Features: Numeric intersections, numerical containment ratio, and strict global country match validation.

## 5. Model
Rescoring is conducted through an optimized `HistGradientBoostingClassifier` trained via Scikit-Learn to estimate the probability that a candidate pair represents an identical entity.

## 6. Deduplication
Deduplication logic is applied to sanitize the raw matching pool, eliminating redundant cyclic paths (e.g., A-B matched and B-A matched) and aggregating predictions coherently.

## 7. Final Matching Logic
The output strictly complies with our *Margin V1* final method tailored for high-precision scenarios:
- **Baseline Threshold**: 0.985
- **Margin Delta**: 0.005
Candidates are retained only if their model probability score is greater than 0.985, and any alternative matches for the same entity are heavily truncated by only accepting matches falling within 0.005 of the top-scoring candidate. 

## 8. Validation and Structural Checks
Final results are validated using strict structural validation scripts to parse output schemas, check constraint integrity (acyclic enforcement), and verify valid ID formats. **Note:** The local validation metrics are utilized exclusively for basic calibration; we do not claim that the local validation score predicts the hidden leaderboard.

## 9. Reproducibility Instructions
To run the code structure conceptually (assuming you have access to the original competition dataset):
1. Install dependencies from `requirements.txt`.
2. Extract the competition dataset into `dataset/` (not included).
3. Execute `src/train_pairwise_matcher.py` if retraining from scratch.
4. Execute `src/final_inference_safe.py` to compile blocked candidates.
5. Execute `src/run_exp_margin.py` to apply the final predictions utilizing the Margin V1 filtering baseline.
6. Verify output correctness via `src/validate_submission.py`.

## 10. Data Restrictions Notice
**IMPORTANT:** The original competition datasets, candidate pairs, trained model `.pkl` artifacts, and SQLite blocking databases are **intentionally not included** in this public GitHub repository due to large file sizes and competition constraints regarding proprietary data artifacts. To execute this code, the dataset must be separately obtained through the competition platform if permitted. No private personal information is exposed in this repository.
