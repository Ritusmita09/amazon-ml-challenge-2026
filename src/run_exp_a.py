import csv
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

input_file = "work/matching_results_safe.tsv"
output_file = "work/matching_results_dedup_v1.tsv"

rows_by_s1 = defaultdict(list)

# Read the file
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for row in reader:
        s1 = row[0]
        matches = row[1].strip() if len(row) > 1 else ""
        if matches:
            rows_by_s1[s1].append(matches.split(","))
        else:
            rows_by_s1[s1].append([])

duplicates = {s1: arrs for s1, arrs in rows_by_s1.items() if len(arrs) > 1}
print(f"Total unique S1 IDs: {len(rows_by_s1)}")
print(f"Total duplicate S1 IDs: {len(duplicates)}")

if duplicates:
    # Let's inspect the first 5 duplicates
    print("\nSample of 5 duplicates behavior:")
    for s1, arrs in list(duplicates.items())[:5]:
        print(f"{s1}: {arrs}")
        
# Deduplicate: union of all matched IDs
total_rows = 0
empty_rows = 0
with open(output_file, "w", encoding="utf-8", newline='') as fout:
    writer = csv.writer(fout, delimiter="\t")
    writer.writerow(["source1_entity_id", "matched_entity_ids"])
    for s1, arrs in rows_by_s1.items():
        total_rows += 1
        combined_matches = set()
        for arr in arrs:
            for m in arr:
                if m:
                    combined_matches.add(m)
        
        matches_str = ",".join(sorted(list(combined_matches)))
        if not matches_str:
            empty_rows += 1
            writer.writerow([s1, ""])
        else:
            writer.writerow([s1, matches_str])

print(f"\nDeduplication complete.")
print(f"Output file: {output_file}")
print(f"Total rows written: {total_rows}")
print(f"Rows with empty matches: {empty_rows}")
