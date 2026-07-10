import re

NEGATION_TRIGGERS = [
    r"không\s+có", r"chưa\s+phát\s+hiện", r"không\s+ghi\s+nhận",
    r"âm\s+tính", r"loại\s+trừ", r"chưa\s+từng", r"không\s+thấy"
]

HISTORICAL_TRIGGERS = [
    r"tiền\s+sử", r"lịch\s+sử", r"năm\s+\d{4}", r"trước\s+đây"
]

TERMINATORS = [r"\.", r"\,", r"nhưng", r"tuy\s+nhiên", r"và"]

class NegExVerifier:
    @staticmethod
    def verify_entity(raw_text, entity):
        ent_copy = entity.copy()
        if "assertions" not in ent_copy:
            ent_copy["assertions"] = []
            
        start, end = ent_copy["position"]
        # Quét ngữ cảnh trước thực thể (window 50 ký tự)
        context_start = max(0, start - 50)
        pre_context = raw_text[context_start:start]
        
        # Kiểm tra nếu bị ngăn bởi ngắt câu
        for term in TERMINATORS:
            if re.search(term, pre_context):
                # Cắt bớt phần sau Terminator
                pre_context = re.split(term, pre_context)[-1]
                
        # Kiểm tra trigger phủ định
        for trig in NEGATION_TRIGGERS:
            if re.search(trig, pre_context, re.IGNORECASE):
                if "isNegated" not in ent_copy["assertions"]:
                    ent_copy["assertions"].append("isNegated")
                    
        # Kiểm tra trigger tiền sử
        for trig in HISTORICAL_TRIGGERS:
            if re.search(trig, pre_context, re.IGNORECASE):
                if "isHistorical" not in ent_copy["assertions"]:
                    ent_copy["assertions"].append("isHistorical")
                    
        return ent_copy
