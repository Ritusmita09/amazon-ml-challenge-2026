# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Sentinel
**Submission Date:** September 2026

---

## 1. Executive Summary
Our approach is a two-stage pipeline consisting of a high-recall, precision-constrained SQLite blocking mechanism (generating candidates natively), followed by a highly-tuned feature extraction step and a HistGradientBoostingClassifier. The system is optimized heavily for Macro F0.5 via a Margin-Drop matching logic (baseline threshold 0.985, Margin Delta 0.005) to retain high-confidence entities while naturally rejecting distant runner-ups.

---

## 2. Methodology

### 2.1 Problem Analysis
Given the extreme penalty for false positives dictated by the Macro F0.5 metric, generating an unrestrained amount of candidates dramatically escalates the risk of catastrophic false positive cascades. In earlier trials, memory issues arose during unrestricted token indexing. We resolved this through memory-bound local database lookup and batched inference pipelines.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Gradient Boosting Classifier (GBDT)
**Core Innovation:** A safe blocking architecture backed by a local SQLite engine (avoiding RAM exhaustion) seamlessly integrated with a RapidFuzz-powered feature tensor, optimized globally implementing a Margin V1 algorithm. 

**FINAL METHOD:**
- Candidate generation / blocking strategy: Memory-safe SQLite based extraction.
- Feature engineering: Token, string ratio, and Jaccard intersections.
- Model architecture: HistGradientBoostingClassifier
- Matching process: Candidates are rescored with the GBDT.
- Baseline threshold = 0.985
- Margin-Drop rule = Delta 0.005 (only retaining candidate matches within 0.005 of the top-scoring candidate, providing the top score > 0.985 baseline)

**REJECTED / PRIOR EXPERIMENTS:**
- B-0.992 (Threshold Only)
- B-0.995 (Threshold Only)
- Top-K truncation variants
- Margin Delta 0.010
- Margin Delta 0.0075
- Margin Delta 0.0025
- country-specific threshold experiments
- address-key experiments
- other exploratory variants

---

## 3. Candidate Generation (Blocking)
Our candidate generation operates on a dual-pass constraint mapping source text to a local SQLite database (`final_inference_safe.py`). 

- **Blocking keys used:** Country matching combined with compact_text deduplication and token intersection.
- **Safe Mode Design:** Candidates must strictly conform to block size limits utilizing SQLite indexing, to preclude unbounded memory allocation panics. 

---

## 4. Matching Model

**Features used:**
- Name features: `rapidfuzz.fuzz.ratio`, `wratio`, `token_set_ratio`, `token_sort_ratio`, length metrics, Jaccard intersection.
- Address features: `compact_addr`, fuzzy ratios, token-set comparisons, length disparities.
- Other features: Numeric intersections, numerical containment ratio, and strict global country match validation. (Total 17 dimensions).

**Model type:** HistGradientBoostingClassifier (via Scikit-Learn)

---

## 5. Results & Error Analysis

- **Final Observed Hidden Leaderboard Score:** 0.656 (Margin V1, Delta 0.005)
- **Previous Best Score:** 0.654 (B-0.992 submission)

Margin V1 with a 0.005 Delta is the highest observed score among the submitted variants. 

- **Common false positives (wrong merges):** Broad token matches across diverse addresses or generic names clustered within dense S2/S3 geographic zones.
- **Common false negatives (missed matches):** Legitimate matches rejected by strict threshold boundaries or falling outside the dynamic margin window.

---

## 6. Conclusion
The combination of memory-safe candidate generation through SQLite constraints and our dynamic Margin-Drop (Delta 0.005) post-processing logic produced a competitively scalable and highly precise solution capable of handling 1.73M source entities effectively.

---

## Appendix

### A. Code Artifacts
Code ships in `src/` inside the final bundle, along with an explicit `requirements.txt` based on exact execution environment dependencies. Reproducibility relies sequentially on `final_inference_safe.py` (generating candidates), baseline `src/run_exp_a.py` processing, and thresholding via `src/run_exp_margin.py`.
