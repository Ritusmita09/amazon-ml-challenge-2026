import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
import pickle
import heapq
import collections
import array
import re
import json
import gc
import multiprocessing as mp
from pathlib import Path
from rapidfuzz import fuzz

BASE = Path("dataset")
WORK = Path("work")

TEST_S1 = BASE / "test" / "test_source1.tsv"
TEST_S2 = BASE / "test" / "test_source2.tsv"
TEST_S3 = BASE / "test" / "test_source3.tsv"

MODEL_PATH = WORK / "pairwise_matcher.pkl"
RESULTS_MATCHING = WORK / "matching_results_safe.tsv"
RESULTS_CANDIDATE = WORK / "candidate_pairs_safe.tsv"
SUMMARY_FILE = WORK / "final_submission_summary_safe.txt"
CHECKPOINT_FILE = WORK / "inference_checkpoint.json"

NAME_K = 2500
ADDR_K = 400
THRESHOLD = 0.985
BATCH_SIZE = 2000
MAX_TOKEN_FREQUENCY = 5000

GENERIC_TOKENS = {
    "the", "and", "for", "from", "with", "ltd", "limited", "llc", "inc",
    "corp", "corporation", "company", "co", "pvt", "private", "plc", "llp",
}

# -----------------------------------------------------
# FASTER STRING OPS (inlined for speed)
# -----------------------------------------------------
def normalize_text(text):
    if not text: return ""
    text = str(text).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def compact_text(text):
    if not text: return ""
    text = str(text).casefold()
    text = text.replace("&", "and")
    return re.sub(r"[^\w]", "", text)

def extract_tokens(text):
    norm = normalize_text(text)
    return {
        t for t in norm.split()
        if len(t) >= 3 and t not in GENERIC_TOKENS
    }

def extract_numbers(text):
    if not text: return set()
    return set(re.findall(r"\d+", str(text)))


# -----------------------------------------------------
# GLOBAL WORKER STATE
# -----------------------------------------------------
clf = None
feat_cols = None
s2_bytes = None
s3_bytes = None
allowed_tokens = None
allowed_numbers = None
inverted_name = None
inverted_addr = None
s_file_idx = None
s_line_offset = None
country_to_id = None

def init_worker():
    global clf, feat_cols, s2_bytes, s3_bytes
    global allowed_tokens, allowed_numbers, inverted_name, inverted_addr
    global s_file_idx, s_line_offset, country_to_id

    try:
        import warnings
        warnings.simplefilter('ignore', UserWarning)

        # 1. Model
        with open(MODEL_PATH, "rb") as f:
            pkg = pickle.load(f)
        clf = pkg["model"]
        feat_cols = pkg["features"]

        # 2. Bytes
        s2_bytes = open(TEST_S2, 'rb').read()
        s3_bytes = open(TEST_S3, 'rb').read()

        # 3. Index Build
        global_token_tf = collections.Counter()
        global_number_tf = collections.Counter()

        inverted_name = collections.defaultdict(lambda: array.array('I'))
        inverted_addr = collections.defaultdict(lambda: array.array('I'))

        s_file_idx = array.array('B')
        s_line_offset = array.array('Q')

        country_to_id = {}
        def get_country_id(c):
            if c not in country_to_id: country_to_id[c] = len(country_to_id)
            return country_to_id[c]

        current_idx = 0
        ref_files = [(0, s2_bytes), (1, s3_bytes)]

        for f_idx, b_data in ref_files:
            idx = b_data.find(b'\n', 0) + 1 # skip header

            while idx < len(b_data):
                start = idx
                idx = b_data.find(b'\n', start)
                if idx == -1: idx = len(b_data)

                line = b_data[start:idx].decode('utf-8')
                idx += 1

                if not line: continue
                parts = line.split('\t')
                if len(parts) < 4: continue

                name = parts[1]
                addr = parts[2]
                c = parts[3].strip().casefold()
                cid = get_country_id(c)

                s_file_idx.append(f_idx)
                s_line_offset.append(start)

                tks = extract_tokens(name)
                nums = extract_numbers(addr)

                for tk in tks:
                    global_token_tf[tk] += 1
                    key = (cid, tk)
                    arr = inverted_name[key]
                    if len(arr) <= MAX_TOKEN_FREQUENCY:
                        arr.append(current_idx)

                for num in nums:
                    global_number_tf[num] += 1
                    key = (cid, num)
                    arr = inverted_addr[key]
                    if len(arr) <= MAX_TOKEN_FREQUENCY:
                        arr.append(current_idx)

                current_idx += 1

        allowed_tokens = {k: v for k, v in global_token_tf.items() if v <= MAX_TOKEN_FREQUENCY}
        allowed_numbers = {k: v for k, v in global_number_tf.items() if v <= MAX_TOKEN_FREQUENCY}

        for k in list(inverted_name.keys()):
            if k[1] not in allowed_tokens:
                del inverted_name[k]

        for k in list(inverted_addr.keys()):
            if k[1] not in allowed_numbers:
                del inverted_addr[k]

    except Exception as e:
        print("Worker init failed:", e)

# -----------------------------------------------------
# PROCESS CHUNK FUNCTION
# -----------------------------------------------------
def process_s1_row(row):
    try:
        s1_id = row[0]
        name = str(row[1])
        addr = str(row[2])
        country = str(row[3]).strip().casefold()

        cid = country_to_id.get(country, -1)

        t1 = extract_tokens(name)
        num1 = extract_numbers(addr)

        tks = t1 & allowed_tokens.keys()
        nums = num1 & allowed_numbers.keys()

        # Heatmap / Top-K
        scores_name = collections.defaultdict(lambda: [0, 0.0])
        for tk in tks:
            tf = 1.0 / allowed_tokens[tk]
            arr = inverted_name.get((cid, tk))
            if arr:
                for cand_id in arr:
                    scores_name[cand_id][0] += 1
                    scores_name[cand_id][1] += tf

        scores_addr = collections.defaultdict(lambda: [0, 0.0])
        for num in nums:
            tf = 1.0 / allowed_numbers[num]
            arr = inverted_addr.get((cid, num))
            if arr:
                for cand_id in arr:
                    scores_addr[cand_id][0] += 1
                    scores_addr[cand_id][1] += tf

        name_heap = []
        for c_id, sc in scores_name.items():
            sort_key = (sc[0], sc[1])
            if len(name_heap) < NAME_K:
                heapq.heappush(name_heap, (sort_key, c_id))
            else:
                heapq.heappushpop(name_heap, (sort_key, c_id))

        addr_heap = []
        for c_id, sc in scores_addr.items():
            sort_key = (sc[0], sc[1])
            if len(addr_heap) < ADDR_K:
                heapq.heappush(addr_heap, (sort_key, c_id))
            else:
                heapq.heappushpop(addr_heap, (sort_key, c_id))

        combined_cad_ids = {item[1] for item in name_heap}
        combined_cad_ids.update({item[1] for item in addr_heap})

        if not combined_cad_ids:
            return s1_id, [], []

        # Feature Precomputation S1
        n1 = normalize_text(name)
        c1 = compact_text(name)
        a1 = normalize_text(addr)
        ac1 = compact_text(addr)
        cnt1 = country
        l1 = len(n1)
        alen1 = len(a1)

        actual_cand_str_ids = []
        all_feats = []

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            for cid_int in combined_cad_ids:
                f_idx = s_file_idx[cid_int]
                offset = s_line_offset[cid_int]
                if f_idx == 0:
                    b_data = s2_bytes
                else:
                    b_data = s3_bytes

                idx = b_data.find(b'\n', offset)
                if idx == -1: idx = len(b_data)

                line = b_data[offset:idx].decode('utf-8')
                parts = line.split('\t')

                c_id_str = parts[0]
                c_name = parts[1]
                c_addr = parts[2]
                c_country = parts[3].strip().casefold()

                actual_cand_str_ids.append(c_id_str)

                # Feature extraction inline
                n2 = normalize_text(c_name)
                c2 = compact_text(c_name)
                a2 = normalize_text(c_addr)
                ac2 = compact_text(c_addr)
                t2 = extract_tokens(c_name)
                num2 = extract_numbers(c_addr)

                feats = [
                    1.0 if (n1 == n2 and n1 != "") else 0.0,
                    1.0 if (c1 == c2 and c1 != "") else 0.0
                ]

                if n1 and n2:
                    feats.extend([
                        fuzz.ratio(n1, n2) / 100.0,
                        fuzz.WRatio(n1, n2) / 100.0,
                        fuzz.token_set_ratio(n1, n2) / 100.0,
                        fuzz.token_sort_ratio(n1, n2) / 100.0
                    ])
                else:
                    feats.extend([0.0, 0.0, 0.0, 0.0])

                l2 = len(n2)
                feats.extend([
                    float(abs(l1 - l2)),
                    (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0
                ])

                intersect_t = len(t1 & t2)
                union_t = len(t1 | t2)
                feats.extend([
                    float(intersect_t),
                    float(intersect_t / union_t) if union_t > 0 else 0.0
                ])

                feats.extend([
                    1.0 if (a1 == a2 and a1 != "") else 0.0,
                    1.0 if (ac1 == ac2 and ac1 != "") else 0.0
                ])

                if a1 and a2:
                    feats.extend([
                        fuzz.ratio(a1, a2) / 100.0,
                        fuzz.WRatio(a1, a2) / 100.0,
                        fuzz.token_set_ratio(a1, a2) / 100.0
                    ])
                else:
                    feats.extend([0.0, 0.0, 0.0])

                feats.append(float(abs(alen1 - len(a2))))

                intersect_num = len(num1 & num2)
                union_num = len(num1 | num2)
                feats.append(float(intersect_num))

                if not num1 and not num2:
                    feats.append(1.0)
                elif not num1 or not num2:
                    feats.append(0.0)
                else:
                    feats.append(float(intersect_num / union_num))

                num_contain = 0.0
                if len(num1) > 0 or len(num2) > 0:
                    if (num1 and num1.issubset(num2)) or (num2 and num2.issubset(num1)):
                        num_contain = 1.0
                feats.append(float(num_contain))

                cnt2 = str(c_country).strip().casefold()
                feats.append(1.0 if (cnt1 == cnt2 and cnt1 != "") else 0.0)

                all_feats.append(feats)

        # Inference
        X_chunk = np.array(all_feats, dtype=np.float32)

        # Surpress HistGradientBoosting warnings again just in case predict_proba raises it alone
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            probs = clf.predict_proba(X_chunk)[:, 1]

        passing = []
        for ix, p in enumerate(probs):
            if p >= THRESHOLD:
                passing.append(actual_cand_str_ids[ix])

        return s1_id, actual_cand_str_ids, passing

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return empty on failure so we don't crash the whole batch
        return row[0], [], []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=BATCH_SIZE)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-batches', type=int, dest='max_batches', default=-1)
    args = parser.parse_args()

    print("="*60)
    print("FINAL INFERENCE ON TEST SET (SAFE MODE)")
    print("="*60)
    start_time = time.time()

    if not TEST_S1.exists() or not TEST_S2.exists() or not TEST_S3.exists():
        print("ERROR: Test data not found.")
        sys.exit(1)

    print("Pre-counting S1 rows...")
    s1_count = sum(1 for _ in open(TEST_S1, encoding="utf-8")) - 1
    print(f"Total Test S1 Entities: {s1_count:,}")

    total_s1 = 0
    total_matched = 0
    total_singletons = 0
    total_candidate_pairs = 0
    total_retained_pairs = 0
    processed_batches = 0

    mode = "w"
    skip_rows = 0

    if args.resume and CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            ckpt = json.load(f)
            total_s1 = ckpt.get("total_s1", 0)
            total_matched = ckpt.get("total_matched", 0)
            total_singletons = ckpt.get("total_singletons", 0)
            total_candidate_pairs = ckpt.get("total_candidate_pairs", 0)
            total_retained_pairs = ckpt.get("total_retained_pairs", 0)
            processed_batches = ckpt.get("processed_batches", 0)
            skip_rows = total_s1
            mode = "a"
        print(f"Resuming from batch {processed_batches} (rows processed: {total_s1})")

    fm = open(RESULTS_MATCHING, mode, encoding="utf-8", buffering=1) # line-buffered
    fc = open(RESULTS_CANDIDATE, mode, encoding="utf-8", buffering=1)

    if mode == "w":
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

    import math
    if skip_rows > 0:
        skip_indices = range(1, skip_rows + 1)
        s1_iter = pd.read_csv(TEST_S1, sep="\t", dtype=str, chunksize=args.batch_size, keep_default_na=False, skiprows=skip_indices)
    else:
        s1_iter = pd.read_csv(TEST_S1, sep="\t", dtype=str, chunksize=args.batch_size, keep_default_na=False)

    WORKERS = args.workers
    print(f"\nStarting {WORKERS} background workers for inference...")
    print("Each worker will independently build the index for ~2 minutes...")

    try:
        with mp.Pool(WORKERS, initializer=init_worker) as pool:
            for batch_idx, s1_chunk in enumerate(s1_iter):
                if args.max_batches != -1 and batch_idx >= args.max_batches:
                    print("Reached max_batches limit. Stopping gracefully.")
                    break

                batch_start = time.time()
                rows = [tuple(x) for x in s1_chunk.values]
                current_batch_size = len(rows)

                print(f"  Batch {processed_batches+1} (Rows up to {total_s1 + current_batch_size})...", flush=True)

                # Map across workers - chunksize optimizes IPC
                results = pool.imap(process_s1_row, rows, chunksize=100)

                for s1_id, cands, matches in results:
                    cands_str = ",".join(cands)
                    matches_str = ",".join(matches)
                    fc.write(f"{s1_id}\t{cands_str}\n")
                    if matches:
                        fm.write(f"{s1_id}\t{matches_str}\n")
                        total_retained_pairs += len(matches)
                        total_matched += 1
                    else:
                        fm.write(f"{s1_id}\t\n")
                        total_singletons += 1
                    total_candidate_pairs += len(cands)

                total_s1 += current_batch_size
                processed_batches += 1

                # Save checkpoint atomically
                ckpt_data = {
                    "total_s1": total_s1,
                    "total_matched": total_matched,
                    "total_singletons": total_singletons,
                    "total_candidate_pairs": total_candidate_pairs,
                    "total_retained_pairs": total_retained_pairs,
                    "processed_batches": processed_batches
                }
                ckpt_tmp = CHECKPOINT_FILE.with_suffix(".json.tmp")
                with open(ckpt_tmp, "w") as f_ckpt:
                    json.dump(ckpt_data, f_ckpt)
                ckpt_tmp.replace(CHECKPOINT_FILE)

                elapsed = time.time() - batch_start
                print(f"    -> batch {processed_batches} took {elapsed:.1f}s (~{len(rows)/elapsed:.1f} rows/s)", flush=True)

                # Explicit memory collector
                del rows, results, s1_chunk
                gc.collect()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting safely.")
    finally:
        fm.close()
        fc.close()

    print("\n[3/3] Inference Complete or Stopped!", flush=True)
    elapsed = time.time() - start_time

    summary_text = (
        "FINAL SUBMISSION SUMMARY (SAFE MODE)\n"
        "============================================================\n"
        f"Configuration: Name Top-K = {NAME_K}, Address Top-K = {ADDR_K}\n"
        f"Pairwise Threshold: {THRESHOLD}\n"
        f"Total Source-1 Entities Evaluated: {total_s1:,}\n"
        f"Total Candidate Pairs Evaluated:   {total_candidate_pairs:,}\n"
        f"Total Retained (Matched) Pairs:    {total_retained_pairs:,}\n"
        f"Source-1 Entities with Matches:    {total_matched:,}\n"
        f"Singleton Source-1 Entities:       {total_singletons:,}\n"
        f"Total Execution Time:              {elapsed:.1f}s\n"
        "\n"
        "Generated Files:\n"
        f"- {RESULTS_CANDIDATE.absolute()}\n"
        f"- {RESULTS_MATCHING.absolute()}\n"
        f"- {CHECKPOINT_FILE.absolute()}\n"
    )

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f_sum:
        f_sum.write(summary_text)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary_text)

if __name__ == "__main__":
    mp.freeze_support()
    main()
