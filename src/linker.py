# Mock dictionaries for Tầng 1
MOCK_ICD10 = {"viêm dạ dày": "K29.9", "viêm phổi thùy": "J18.1", "bóc tách động mạch chủ": "I71.0"}
MOCK_RXNORM = {"omeprazole": "7646", "methadone": "6813", "acetaminophen": "161"}
MOCK_ABBR = {
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "ercp": "nội soi mật tụy ngược dòng",
    "vdd": "viêm dạ dày"
}

class CandidateLinker:
    @staticmethod
    def expand_abbreviations(text):
        if not text:
            return text
        words = text.split()
        expanded_words = []
        punc_chars = ".,;:?!()[]{}*-\""
        for w in words:
            # Tìm ký tự đặc biệt ở đầu và cuối
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
        ent_text = entity["text"].lower()
        ent_type = entity["type"]
        
        # Expand abbreviation on entity text if any
        ent_text = cls.expand_abbreviations(ent_text)
        
        # Tầng 1: Exact Match Dictionary
        if ent_type == "CHẨN_ĐOÁN":
            if ent_text in MOCK_ICD10:
                code = MOCK_ICD10[ent_text]
                if "." in code:
                    return [code, code.split(".")[0]]
                return [code]
        elif ent_type == "THUỐC":
            # Đọc cấu trúc thuốc từ LLM để so khớp database
            med_ing = entity.get("med_ingredient")
            if med_ing:
                med_ing_expanded = cls.expand_abbreviations(med_ing.lower())
                if med_ing_expanded in MOCK_RXNORM:
                    return [MOCK_RXNORM[med_ing_expanded]]
            if ent_text in MOCK_RXNORM:
                return [MOCK_RXNORM[ent_text]]
                
        # Tầng 2: RRF (BM25 + SapBERT) - Fallback mock
        # Trả về danh sách rỗng nếu không khớp
        return []
