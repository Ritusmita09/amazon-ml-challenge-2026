import re
from rapidfuzz import fuzz

GENERIC_TOKENS = {
    "the", "and", "for", "from", "with", "ltd", "limited", "llc", "inc",
    "corp", "corporation", "company", "co", "pvt", "private", "plc", "llp",
}

def normalize_text(text):
    if text is None:
        return ""
    text = str(text).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compact_text(text):
    if text is None:
        return ""
    text = str(text).casefold()
    text = text.replace("&", "and")
    text = re.sub(r"[^\w]", "", text)
    return text

def extract_tokens(text, min_len=3):
    norm = normalize_text(text)
    return {
        t for t in norm.split()
        if len(t) >= min_len and t not in GENERIC_TOKENS
    }

def extract_numbers(text):
    if text is None:
        return set()
    return set(re.findall(r"\d+", str(text)))

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def compute_features(s1_name, s1_address, s1_country,
                     c_name, c_address, c_country,
                     s1_name_tokens=None, s1_numbers=None):
    
    # Pre-computation where necessary
    n1 = normalize_text(s1_name)
    n2 = normalize_text(c_name)
    
    c1 = compact_text(s1_name)
    c2 = compact_text(c_name)
    
    a1 = normalize_text(s1_address)
    a2 = normalize_text(c_address)
    
    ac1 = compact_text(s1_address)
    ac2 = compact_text(c_address)
    
    if s1_name_tokens is None:
        t1 = extract_tokens(s1_name)
    else:
        t1 = s1_name_tokens
        
    t2 = extract_tokens(c_name)
    
    if s1_numbers is None:
        num1 = extract_numbers(s1_address)
    else:
        num1 = s1_numbers
    
    num2 = extract_numbers(c_address)
    
    # Base Features structure
    feats = {}
    
    # --- NAME FEATURES ---
    feats["name_exact_match"] = int(n1 == n2 and n1 != "")
    feats["name_compact_match"] = int(c1 == c2 and c1 != "")
    
    if n1 and n2:
        feats["name_fuzz_ratio"] = fuzz.ratio(n1, n2) / 100.0
        feats["name_fuzz_wratio"] = fuzz.WRatio(n1, n2) / 100.0
        feats["name_token_set_ratio"] = fuzz.token_set_ratio(n1, n2) / 100.0
        feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(n1, n2) / 100.0
    else:
        feats["name_fuzz_ratio"] = 0.0
        feats["name_fuzz_wratio"] = 0.0
        feats["name_token_set_ratio"] = 0.0
        feats["name_token_sort_ratio"] = 0.0
        
    len1, len2 = len(n1), len(n2)
    feats["name_length_diff"] = abs(len1 - len2)
    feats["name_length_ratio"] = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0
    
    intersect_t = len(t1 & t2)
    union_t = len(t1 | t2)
    feats["name_shared_token_count"] = intersect_t
    feats["name_shared_token_ratio"] = intersect_t / union_t if union_t > 0 else 0.0
    
    # --- ADDRESS FEATURES ---
    feats["address_exact_match"] = int(a1 == a2 and a1 != "")
    feats["address_compact_match"] = int(ac1 == ac2 and ac1 != "")
    
    if a1 and a2:
        feats["address_fuzz_ratio"] = fuzz.ratio(a1, a2) / 100.0
        feats["address_fuzz_wratio"] = fuzz.WRatio(a1, a2) / 100.0
        feats["address_token_set_ratio"] = fuzz.token_set_ratio(a1, a2) / 100.0
    else:
        feats["address_fuzz_ratio"] = 0.0
        feats["address_fuzz_wratio"] = 0.0
        feats["address_token_set_ratio"] = 0.0
        
    alen1, alen2 = len(a1), len(a2)
    feats["address_length_diff"] = abs(alen1 - alen2)
    
    intersect_num = len(num1 & num2)
    union_num = len(num1 | num2)
    feats["address_number_overlap"] = intersect_num
    feats["address_number_jaccard"] = jaccard(num1, num2)
    
    num_contain = 0
    if len(num1) > 0 or len(num2) > 0:
        if (num1 and num1.issubset(num2)) or (num2 and num2.issubset(num1)):
            num_contain = 1
    feats["address_number_containment"] = num_contain
    
    # --- OTHER FEATURES ---
    cnt1 = str(s1_country).strip().casefold()
    cnt2 = str(c_country).strip().casefold()
    feats["country_same"] = int(cnt1 == cnt2 and cnt1 != "")
    
    return feats

