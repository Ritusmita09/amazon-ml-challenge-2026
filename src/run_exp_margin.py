import csv
import sys
import time
import pickle
import numpy as np
import warnings
from rapidfuzz import fuzz
import psutil
import os

csv.field_size_limit(sys.maxsize)
sys.path.append('.') # import our feature extractors
from feature_extractor import normalize_text, compact_text, extract_tokens, extract_numbers

MARGIN = 0.005

print("Loading test sources...")
def load_source_file(filepath):
    lookup = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 4:
                lookup[row[0]] = (row[1], row[2], row[3].strip().casefold())
    return lookup

s1_db = load_source_file("dataset/test/test_source1.tsv")
s2_db = load_source_file("dataset/test/test_source2.tsv")
s3_db = load_source_file("dataset/test/test_source3.tsv")
print("Finished loading dictionaries.")

clf_dict = pickle.load(open("work/pairwise_matcher.pkl", "rb"))
clf = clf_dict["model"]

input_file = "work/matching_results_dedup_v1.tsv"
out_margin = "work/matching_results_margin_v1.tsv"

start_time = time.time()

all_s1_ids = []
all_matched_lists = []
with open(input_file, "r", encoding="utf-8") as fin:
    reader = csv.reader(fin, delimiter="\t")
    next(reader)
    for row in reader:
        all_s1_ids.append(row[0])
        matches_str = row[1].strip() if len(row) > 1 else ""
        if matches_str:
            all_matched_lists.append(matches_str.split(","))
        else:
            all_matched_lists.append([])

total_s1 = len(all_s1_ids)

fm = open(out_margin, "w", encoding="utf-8", newline='')
writer_m = csv.writer(fm, delimiter="\t")
writer_m.writerow(["source1_entity_id", "matched_entity_ids"])

BATCH_SIZE = 10000
num_batches = (total_s1 + BATCH_SIZE - 1) // BATCH_SIZE

empty_margin = 0
matched_s1_margin = 0
retained_margin = 0
total_pairs_rescored = 0

print("Starting inference...")

for b in range(num_batches):
    start_idx = b * BATCH_SIZE
    end_idx = min((b + 1) * BATCH_SIZE, total_s1)

    batch_s1_ids = all_s1_ids[start_idx:end_idx]
    batch_match_lists = all_matched_lists[start_idx:end_idx]

    X_batch = []
    meta_batch = []

    for i_in_batch, s1_id in enumerate(batch_s1_ids):
        m_list = batch_match_lists[i_in_batch]
        if not m_list:
            continue

        s1_name, s1_addr, s1_country = s1_db.get(s1_id, ("", "", ""))
        n1 = normalize_text(s1_name)
        c1 = compact_text(s1_name)
        a1 = normalize_text(s1_addr)
        ac1 = compact_text(s1_addr)
        cnt1 = s1_country
        l1 = len(n1)
        alen1 = len(a1)
        t1 = extract_tokens(s1_name)
        num1 = extract_numbers(s1_addr)

        for c_id_str in m_list:
            total_pairs_rescored += 1
            if c_id_str.startswith("S2"):
                c_name, c_addr, c_country = s2_db.get(c_id_str, ("", "", ""))
            elif c_id_str.startswith("S3"):
                c_name, c_addr, c_country = s3_db.get(c_id_str, ("", "", ""))
            else:
                continue

            cnt2 = c_country
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

            feats.append(1.0 if (cnt1 == cnt2 and cnt1 != "") else 0.0)

            X_batch.append(feats)
            meta_batch.append((i_in_batch, c_id_str))

    # Batch predict
    cand_scores_by_idx = [[] for _ in range(end_idx - start_idx)]

    if len(X_batch) > 0:
        X_np = np.array(X_batch, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            probs = clf.predict_proba(X_np)[:, 1]

        for i_pair, p in enumerate(probs):
            i_in_batch, c_id_str = meta_batch[i_pair]
            cand_scores_by_idx[i_in_batch].append((p, c_id_str))

    # Write to file
    for i_in_batch, s1_id in enumerate(batch_s1_ids):
        scored_cands = cand_scores_by_idx[i_in_batch]
        if scored_cands:
            scored_cands.sort(key=lambda x: x[0], reverse=True)
            top_score = scored_cands[0][0]
            
            retained_candidates = [c_id for sc, c_id in scored_cands if top_score - sc <= MARGIN]
            retained_candidates.sort()
            
            writer_m.writerow([s1_id, ",".join(retained_candidates)])
            matched_s1_margin += 1
            retained_margin += len(retained_candidates)
        else:
            writer_m.writerow([s1_id, ""])
            empty_margin += 1

    if (b + 1) % 5 == 0:
        print(f"Processed {(b + 1) * BATCH_SIZE} entities...")

fm.close()

end_time = time.time()
process = psutil.Process()
mem_mb = process.memory_info().rss / 1024 / 1024

print("\n=== RESULTS MARGIN V1 (Delta <= 0.005) ===")
print(f"total rows: {total_s1}")
print(f"unique Source-1 IDs: {total_s1}")
print(f"Source-1 entities with at least one match: {matched_s1_margin}")
print(f"empty-match Source-1 entities: {empty_margin}")
print(f"total retained pairs: {retained_margin}")
if matched_s1_margin > 0:
    print(f"average matches per matched Source-1 entity: {retained_margin / matched_s1_margin:.2f}")

print("\n=== PERFORMANCE ===")
print(f"total pairs rescored: {total_pairs_rescored}")
print(f"exact runtime: {end_time - start_time:.2f}s")
print(f"peak/observed RAM: {mem_mb:.2f} MB")
