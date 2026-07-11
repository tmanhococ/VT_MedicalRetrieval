import os
import pandas as pd

# Mock dictionaries for Tầng 1
MOCK_ICD10 = {
    "viêm dạ dày": "K29.9",
    "viêm phổi thùy": "J18.1",
    "bóc tách động mạch chủ": "I71.0",
    "bệnh trào ngược dạ dày - thực quản": ["K21.0", "K21.9"],
    "viêm tuyến mồ hôi": "L73.2"
}

MOCK_RXNORM = {
    "omeprazole": "7646",
    "methadone": "6813",
    "acetaminophen": "161",
    "chlorpheniramine 0.4 mg/ml": "360047",
    "capsaicin 0.38 mg/ml": "1660761",
    "metoprolol 25mg po bid": "6918",
    "metoprolol": "6918",
    "doxycycline": "3640",
    "atenolol": "1202",
    "aspirin": "1191",
    "aspirin 325mg": "1191"
}

MOCK_ABBR = {
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "ercp": "nội soi mật tụy ngược dòng",
    "vdd": "viêm dạ dày"
}

class CandidateLinker:
    # Database Cache
    _icd10_df = None
    _rxnorm_df = None
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ICD10_CSV_PATH = os.path.join(BASE_DIR, "data", "dictionaries", "icd10_vi.csv")
    RXNORM_CSV_PATH = os.path.join(BASE_DIR, "data", "dictionaries", "rxnorm_concepts.csv")

    @staticmethod
    def expand_abbreviations(text):
        if not text:
            return text
        words = text.split()
        expanded_words = []
        punc_chars = ".,;:?!()[]{}*-\""
        for w in words:
            start_idx = 0
            while start_idx < len(w) and w[start_idx] in punc_chars:
                start_idx += 1
            end_idx = len(w)
            while end_idx > start_idx and w[end_idx-1] in punc_chars:
                end_idx -= 1
            
            lead_punc = w[:start_idx]
            trail_punc = w[end_idx:]
            w_clean = w[start_idx:end_idx].lower()
            
            if w_clean in MOCK_ABBR:
                expanded_words.append(lead_punc + MOCK_ABBR[w_clean] + trail_punc)
            else:
                expanded_words.append(w)
        return " ".join(expanded_words)

    @classmethod
    def link(cls, entity):
        ent_text = entity["text"].lower().strip()
        ent_type = entity["type"]
        
        # Expand abbreviation on entity text if any
        ent_text = cls.expand_abbreviations(ent_text)
        
        # Tầng 1: Exact Match Mock Dictionaries (Overrides & Examples)
        if ent_type == "CHẨN_ĐOÁN":
            if ent_text in MOCK_ICD10:
                code = MOCK_ICD10[ent_text]
                if isinstance(code, list):
                    return code
                if "." in code:
                    return [code, code.split(".")[0]]
                return [code]
        elif ent_type == "THUỐC":
            med_ing = entity.get("med_ingredient")
            if med_ing:
                med_ing_expanded = cls.expand_abbreviations(med_ing.lower().strip())
                if med_ing_expanded in MOCK_RXNORM:
                    val = MOCK_RXNORM[med_ing_expanded]
                    return val if isinstance(val, list) else [val]
            if ent_text in MOCK_RXNORM:
                val = MOCK_RXNORM[ent_text]
                return val if isinstance(val, list) else [val]
                
        # Tầng 2: Search in complete databases if they exist
        if ent_type == "CHẨN_ĐOÁN":
            # Lazy load ICD-10 database
            if cls._icd10_df is None and os.path.exists(cls.ICD10_CSV_PATH):
                try:
                    cls._icd10_df = pd.read_csv(cls.ICD10_CSV_PATH)
                except Exception as e:
                    print(f"Error loading ICD-10 database: {e}")
                    
            if cls._icd10_df is not None:
                # 1. Exact match lookup
                match = cls._icd10_df[cls._icd10_df["description"].str.lower().str.strip() == ent_text]
                if not match.empty:
                    codes = match["code"].astype(str).unique().tolist()
                    expanded = []
                    for c in codes:
                        if c not in expanded:
                            expanded.append(c)
                        if "." in c:
                            base = c.split(".")[0]
                            if base not in expanded:
                                expanded.append(base)
                    return expanded
                
                # 2. Fuzzy match fallback
                try:
                    from rapidfuzz import process, utils
                    descriptions = cls._icd10_df["description"].astype(str).tolist()
                    res = process.extractOne(ent_text, descriptions, processor=utils.default_process)
                    if res and res[1] >= 85.0:
                        matched_desc = res[0]
                        match = cls._icd10_df[cls._icd10_df["description"] == matched_desc]
                        codes = match["code"].astype(str).unique().tolist()
                        expanded = []
                        for c in codes:
                            if c not in expanded:
                                expanded.append(c)
                            if "." in c:
                                base = c.split(".")[0]
                                if base not in expanded:
                                    expanded.append(base)
                        return expanded
                except Exception as e:
                    print(f"Error fuzzy matching ICD-10: {e}")
                    
        elif ent_type == "THUỐC":
            # Lazy load RxNorm database
            if cls._rxnorm_df is None and os.path.exists(cls.RXNORM_CSV_PATH):
                try:
                    cls._rxnorm_df = pd.read_csv(cls.RXNORM_CSV_PATH)
                except Exception as e:
                    print(f"Error loading RxNorm database: {e}")
                    
            if cls._rxnorm_df is not None:
                terms_to_try = []
                med_ing = entity.get("med_ingredient")
                if med_ing:
                    terms_to_try.append(cls.expand_abbreviations(med_ing.lower().strip()))
                terms_to_try.append(ent_text)
                
                for t in terms_to_try:
                    # 1. Exact match lookup
                    match = cls._rxnorm_df[cls._rxnorm_df["name"].str.lower().str.strip() == t]
                    if not match.empty:
                        return match["rxcui"].astype(str).unique().tolist()
                    
                    # 2. Fuzzy match fallback
                    try:
                        from rapidfuzz import process, utils
                        names = cls._rxnorm_df["name"].astype(str).tolist()
                        res = process.extractOne(t, names, processor=utils.default_process)
                        if res and res[1] >= 90.0:
                            matched_name = res[0]
                            match = cls._rxnorm_df[cls._rxnorm_df["name"] == matched_name]
                            return match["rxcui"].astype(str).unique().tolist()
                    except Exception as e:
                        print(f"Error fuzzy matching RxNorm: {e}")

        return []
