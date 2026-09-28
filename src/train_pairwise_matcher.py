import sqlite3
import pandas as pd
import random
import pickle
import time
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

from feature_extractor import (
    compute_features, extract_tokens, extract_numbers, normalize_text
)

BASE = Path("dataset")
WORK = Path("work")

TOKEN_DB = WORK / "name_token_validation.sqlite"
NUMBER_DB = WORK / "multi_blocking_numbers.sqlite"
BASE_CANDIDATES = WORK / "validation_candidates.tsv"

NAME_TOP_K = 1000
ADDRESS_TOP_K = 200

# We will sample 12,500 S1 entities total (10k train, 2.5k val) for this run.
# S1 entities starting from the top of validation_candidates.tsv
TOTAL_S1 = 12500
TRAIN_RATIO = 0.8

MAX_NEGATIVES_PER_S1 = 30 # Limit for memory/balance

def main():
    print("="*60)
    print("TRAIN PAIRWISE MATCHER (HistGradientBoosting)")
    print("="*60)

    start_time = time.time()

    # 1. LOAD S1 TARGETS
    base_df = pd.read_csv(BASE_CANDIDATES, sep="\t", dtype=str, keep_default_na=False, nrows=TOTAL_S1)
    validation_ids = list(base_df["source1_entity_id"])

    random.seed(42)
    random.shuffle(validation_ids)

    train_size = int(len(validation_ids) * TRAIN_RATIO)
    train_s1 = set(validation_ids[:train_size])
    val_s1 = set(validation_ids[train_size:])

    print(f"Total S1: {len(validation_ids):,}")
    print(f"Train S1: {len(train_s1):,} | Val S1: {len(val_s1):,}")

    # Base candidates map
    base_candidates = {}
    for row in base_df.itertuples(index=False, name=None):
        if row[1]:
            base_candidates[row[0]] = {x.strip() for x in row[1].split(",") if x.strip()}
        else:
            base_candidates[row[0]] = set()

    # 2. LOAD S1 INFO
    print("Loading S1 records...")
    s1_records = {}
    for chunk in pd.read_csv(BASE / "train" / "train_source1.tsv", sep="\t", dtype=str, chunksize=100000, keep_default_na=False):
        found = chunk[chunk["entity_id"].isin(validation_ids)]
        for row in found.itertuples(index=False, name=None):
            s1_records[row[0]] = {
                "name": row[1],
                "address": row[2],
                "country": str(row[3]).strip().casefold(),
                "name_tokens": extract_tokens(row[1]),
                "numbers": extract_numbers(row[2])
            }
        if len(s1_records) == len(validation_ids):
            break

    # 3. LOAD GROUND TRUTH
    print("Loading Ground Truth...")
    ground_truth = {}
    for chunk in pd.read_csv(BASE / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, chunksize=100000, keep_default_na=False):
        found = chunk[chunk["source1_entity_id"].isin(validation_ids)]
        for row in found.itertuples(index=False, name=None):
            if row[1]:
                ground_truth[row[0]] = {x.strip() for x in row[1].split(",") if x.strip()}
            else:
                ground_truth[row[0]] = set()
        if len(ground_truth) == len(validation_ids):
            break

    # 4. CANDIDATE RETRIEVAL PER S1
    print("Retrieving Expansion Candidates via SQLite...")
    token_conn = sqlite3.connect(TOKEN_DB)

    print("Preparing token frequency table...")
    token_conn.execute("DROP TABLE IF EXISTS temp.token_frequency")
    token_conn.execute("""
        CREATE TEMP TABLE token_frequency AS
        SELECT token, COUNT(*) AS frequency
        FROM token_index
        GROUP BY token
    """)
    token_conn.execute("CREATE INDEX temp.idx_token_frequency ON token_frequency(token)")
    token_conn.commit()

    number_conn = sqlite3.connect(NUMBER_DB)

    def get_ranked_candidates(conn, index_table, col_name, country, tokens, top_k):
        if not tokens: return []
        tlist = list(tokens)
        placeholders = ",".join("?" for _ in tlist)
        query = f"""
            SELECT i.entity_id
            FROM {index_table} i
            JOIN {index_table.split('_')[0]}_frequency tf ON tf.{col_name} = i.{col_name}
            WHERE i.country = ? AND i.{col_name} IN ({placeholders})
            GROUP BY i.entity_id
            ORDER BY COUNT(DISTINCT i.{col_name}) DESC, SUM(1.0 / tf.frequency) DESC
            LIMIT ?
        """
        rows = conn.execute(query, [country, *tlist, top_k]).fetchall()
        return [r[0] for r in rows]

    all_required_s2s3 = set()
    s1_candidates = {}

    for c, s1_id in enumerate(validation_ids, 1):
        info = s1_records[s1_id]
        n_cands = get_ranked_candidates(token_conn, "token_index", "token", info["country"], info["name_tokens"], NAME_TOP_K)
        a_cands = get_ranked_candidates(number_conn, "number_index", "number", info["country"], info["numbers"], ADDRESS_TOP_K)

        combined = base_candidates.get(s1_id, set()) | set(n_cands) | set(a_cands)
        s1_candidates[s1_id] = {
            "name_cands": n_cands,
            "addr_cands": a_cands,
            "all_cands": list(combined)
        }
        all_required_s2s3.update(combined)

        if c % 1000 == 0:
            print(f"Retrieved targets for {c}/{len(validation_ids)} S1s", end="\r", flush=True)
    print(f"\nRetrieved total initial candidate IDs: {len(all_required_s2s3):,}")

    token_conn.close()
    number_conn.close()

    # 5. LOAD REQUIRED S2/S3 RECORDS
    print("Scanning S2/S3 for candidate features...")
    candidate_records = {}

    s2_needed = {x for x in all_required_s2s3 if x.startswith("S2")}
    if s2_needed:
        for chunk in pd.read_csv(BASE/"train"/"train_source2.tsv", sep="\t", dtype=str, chunksize=200000, keep_default_na=False):
            found = chunk[chunk["entity_id"].isin(s2_needed)]
            for row in found.itertuples(index=False, name=None):
                candidate_records[row[0]] = {"name": row[1], "address": row[2], "country": str(row[3]).strip().casefold()}
            if len(candidate_records) == len(s2_needed):
                break

    s3_needed = {x for x in all_required_s2s3 if x.startswith("S3")}
    if s3_needed:
        for chunk in pd.read_csv(BASE/"train"/"train_source3.tsv", sep="\t", dtype=str, chunksize=200000, keep_default_na=False):
            found = chunk[chunk["entity_id"].isin(s3_needed)]
            for row in found.itertuples(index=False, name=None):
                candidate_records[row[0]] = {"name": row[1], "address": row[2], "country": str(row[3]).strip().casefold()}
            if len(candidate_records) >= len(all_required_s2s3):
                break

    print(f"Loaded {len(candidate_records):,} S2/S3 records to memory.")

    # 6. FEATURE EXTRACTION
    print("Extracting features (this may take a few minutes)...")

    train_features = []
    val_features = []

    for c, s1_id in enumerate(validation_ids, 1):
        is_train = s1_id in train_s1
        info = s1_records.get(s1_id)
        if not info: continue

        cands_dict = s1_candidates.get(s1_id, {})
        all_cands = cands_dict.get("all_cands", [])

        true_ids = ground_truth.get(s1_id, set())

        # Sub-sample negative candidates for training to balance and speed up
        if is_train:
            pos_cands = [x for x in all_cands if x in true_ids]
            neg_cands = [x for x in all_cands if x not in true_ids]

            # Use top negative candidates (hard negatives) because Top-K ranked them high
            if len(neg_cands) > MAX_NEGATIVES_PER_S1:
                n_cands_set = set(cands_dict.get("name_cands", []))
                base_cands_set = base_candidates.get(s1_id, set())
                hard_neg = [x for x in neg_cands if x in base_cands_set or x in n_cands_set]
                random.shuffle(hard_neg)
                other_neg = [x for x in neg_cands if x not in hard_neg]
                random.shuffle(other_neg)

                selected_neg = (hard_neg + other_neg)[:MAX_NEGATIVES_PER_S1]
            else:
                selected_neg = neg_cands

            eval_cands = pos_cands + selected_neg
        else:
            eval_cands = all_cands # For validation, keep ALL candidates

        for cand_id in eval_cands:
            c_info = candidate_records.get(cand_id)
            if not c_info: continue

            feats = compute_features(
                info["name"], info["address"], info["country"],
                c_info["name"], c_info["address"], c_info["country"],
                s1_name_tokens=info["name_tokens"], s1_numbers=info["numbers"]
            )

            label = int(cand_id in true_ids)

            row_dict = {
                "s1_id": s1_id,
                "cand_id": cand_id,
                "label": label
            }
            row_dict.update(feats)

            if is_train:
                train_features.append(row_dict)
            else:
                val_features.append(row_dict)

        if c % 1000 == 0:
            print(f"Processed features for {c}/{len(validation_ids)} S1s", end="\r", flush=True)

    print("\nFeature extraction complete.")

    train_df = pd.DataFrame(train_features)
    val_df = pd.DataFrame(val_features)

    print(f"Train Dataset: {len(train_df):,} pairs ({train_df['label'].sum():,} positives).")
    print(f"Validation Dataset: {len(val_df):,} pairs ({val_df['label'].sum():,} positives).")

    # 7. TRAIN MODEL
    print("Training HistGradientBoostingClassifier...")
    feat_cols = [c for c in train_df.columns if c not in ["s1_id", "cand_id", "label"]]

    print(f"Feature list ({len(feat_cols)}):")
    for f in feat_cols: print("  -", f)

    X_train = train_df[feat_cols].astype(float)
    y_train = train_df["label"].astype(int)

    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    print("Exporting model & validation dataset...")

    val_df.to_parquet(WORK / "pairwise_val_features.parquet", index=False)

    with open(WORK / "pairwise_matcher.pkl", "wb") as f:
        pickle.dump({"model": clf, "features": feat_cols}, f)

    duration = time.time() - start_time
    print(f"Training Pipeline Complete in {duration:.1f}s.")

if __name__ == "__main__":
    main()
