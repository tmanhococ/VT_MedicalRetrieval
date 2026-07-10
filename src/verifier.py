import re

NEGATION_TRIGGERS = [
    r"không\s+có", r"chưa\s+phát\s+hiện", r"không\s+ghi\s+nhận",
    r"loại\s+trừ", r"chưa\s+từng", r"không\s+thấy",
    r"\bkhông\b", r"không\s+bị"
]

HISTORICAL_TRIGGERS = [
    r"tiền\s+sử", r"lịch\s+sử", r"năm\s+\d{4}", r"trước\s+đây"
]

POST_NEGATION_TRIGGERS = [
    r"âm\s+tính", r"phủ\s+định", r"bình\s+thường"
]

TERMINATORS = [
    r"\.", r"\n", r"\bnhưng\b", r"\btuy\s+nhiên\b", r"\bvà\b",
    r"\bbut\b", r"\bhowever\b", r"\band\b"
]

# Pre-compile regex patterns at the module level for optimal performance
COMPILED_NEGATION_TRIGGERS = [re.compile(trig, re.IGNORECASE) for trig in NEGATION_TRIGGERS]
COMPILED_HISTORICAL_TRIGGERS = [re.compile(trig, re.IGNORECASE) for trig in HISTORICAL_TRIGGERS]
COMPILED_POST_NEGATION_TRIGGERS = [re.compile(trig, re.IGNORECASE) for trig in POST_NEGATION_TRIGGERS]
COMPILED_TERMINATORS = [re.compile(term, re.IGNORECASE) for term in TERMINATORS]

class NegExVerifier:
    @staticmethod
    def verify_entity(raw_text, entity):
        ent_copy = entity.copy()
        ent_copy["assertions"] = list(entity.get("assertions", []))
            
        start, end = ent_copy["position"]
        
        # Quét ngữ cảnh trước thực thể (window 50 ký tự)
        context_start = max(0, start - 50)
        pre_context = raw_text[context_start:start]
        
        # Kiểm tra nếu bị ngăn bởi ngắt câu (tìm terminator cuối cùng trong pre_context)
        latest_term_idx = -1
        for term in COMPILED_TERMINATORS:
            matches = list(term.finditer(pre_context))
            if matches:
                latest_term_idx = max(latest_term_idx, matches[-1].end())
        if latest_term_idx != -1:
            pre_context = pre_context[latest_term_idx:]
                
        # Kiểm tra trigger phủ định (pre-context)
        for trig in COMPILED_NEGATION_TRIGGERS:
            if trig.search(pre_context):
                if "isNegated" not in ent_copy["assertions"]:
                    ent_copy["assertions"].append("isNegated")
                    
        # Kiểm tra trigger tiền sử
        for trig in COMPILED_HISTORICAL_TRIGGERS:
            if trig.search(pre_context):
                if "isHistorical" not in ent_copy["assertions"]:
                    ent_copy["assertions"].append("isHistorical")
                    
        # Quét ngữ cảnh sau thực thể (window 30 ký tự)
        context_end = min(len(raw_text), end + 30)
        post_context = raw_text[end:context_end]
        
        # Truncate post_context tại terminator đầu tiên
        earliest_term_idx = len(post_context)
        for term in COMPILED_TERMINATORS:
            match = term.search(post_context)
            if match:
                earliest_term_idx = min(earliest_term_idx, match.start())
        post_context = post_context[:earliest_term_idx]
        
        # Kiểm tra trigger phủ định (post-context)
        for trig in COMPILED_POST_NEGATION_TRIGGERS:
            if trig.search(post_context):
                if "isNegated" not in ent_copy["assertions"]:
                    ent_copy["assertions"].append("isNegated")
                    
        return ent_copy
