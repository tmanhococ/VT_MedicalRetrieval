import re

HEADER_RULES = {
    "HISTORY": {
        "regex": r"^\d+[\.\s]+tiền\s+sử\s+bệnh(.*?)$|^\d+[\.\s]+lịch\s+sử\s+bệnh(.*?)$|^tiền\s+sử\s+bệnh(.*?)$|^các\s+tình\s+trạng\s+bệnh\s+lý\s+mạn\s+tính",
        "default_type_hint": "CHẨN_ĐOÁN",
        "default_assertion_hint": ["isHistorical"]
    },
    "MEDICATIONS": {
        "regex": r"^[\*\-\s]*thuốc\s+trước\s+khi\s+nhập\s+viện(.*?)$|^[\*\-\s]*thuốc\s+đang\s+dùng\s+trước\s+khi\s+nhập\s+viện(.*?)$",
        "default_type_hint": "THUỐC",
        "default_assertion_hint": ["isHistorical"]
    },
    "CURRENT_HISTORY": {
        "regex": r"^(?:\d+[\.\s]+)?bệnh\s+sử(.*?)$|^(?:\d+[\.\s]+)?tiền\s+sử\s+bệnh\s+hiện\s+tại(.*?)$|^(?:\d+[\.\s]+)?tiền\s+sử\s+bệnh\s+bệnh\s+hiện\s+tại(.*?)$|^(?:\d+[\.\s]+)?lịch\s+sử\s+bệnh\s+hiện\s+tại(.*?)$",
        "default_type_hint": None,
        "default_assertion_hint": []
    },
    "SYMPTOMS": {
        "regex": r"^[\*\-\s]*(?:các\s+)?triệu\s+chứng\s+hiện\s+tại(.*?)$|^[\*\-\s]*(?:các\s+)?triệu\s+chứng\s+khi\s+nhập\s+viện(.*?)$",
        "default_type_hint": "TRIỆU_CHỨNG",
        "default_assertion_hint": []
    },
    "LABS": {
        "regex": r"^[\*\-\s]*kết\s+quả\s+xét\s+nghiệm(.*?)$|^[\*\-\s]*cận\s+lâm\s+sàng(.*?)$|^[\*\-\s]*kết\s+quả\s+chẩn\s+đoán\s+hình\s+ảnh(.*?)$",
        "default_type_hint": "TÊN_XÉT_NGHIỆM",
        "default_assertion_hint": []
    },
    "DIAGNOSIS": {
        "regex": r"^[\*\-\s]*(?:các\s+)?phát\s+hiện\s+chẩn\s+đoán\s+khác(.*?)$|^[\*\-\s]*(?:chẩn\s+đoán|chẩn\s+đoán\s+sơ\s+bộ)(.*?)$",
        "default_type_hint": "CHẨN_ĐOÁN",
        "default_assertion_hint": []
    },
    "HOSPITAL_EVALUATION": {
        "regex": r"^\d+[\.\s]+đánh\s+giá\s+tại\s+bệnh\s+viện(.*?)$|^\.đánh\s+giá\s+tại\s+bệnh\s+viện(.*?)$|^\d+[\.\s]+khám\s+tại\s+bệnh\s+viện(.*?)$|^tình\s+trạng\s+lúc\s+vào(.*?)$",
        "default_type_hint": None,
        "default_assertion_hint": []
    }
}

class StructuralParser:
    @staticmethod
    def segment(text):
        if not text:
            return []
            
        lines = text.splitlines()
        blocks = []
        current_block = None
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if not current_block:
                    current_block = {
                        "section_type": "UNSTRUCTURED",
                        "default_type_hint": None,
                        "default_assertion_hint": [],
                        "lines": []
                    }
                current_block["lines"].append(line)
                continue
            
            matched_category = None
            for cat, rule in HEADER_RULES.items():
                if re.match(rule["regex"], line_stripped, re.IGNORECASE):
                    matched_category = cat
                    break
            
            if matched_category:
                if current_block:
                    blocks.append(current_block)
                current_block = {
                    "section_type": matched_category,
                    "default_type_hint": HEADER_RULES[matched_category]["default_type_hint"],
                    "default_assertion_hint": HEADER_RULES[matched_category]["default_assertion_hint"],
                    "lines": [line]
                }
            else:
                if not current_block:
                    current_block = {
                        "section_type": "UNSTRUCTURED",
                        "default_type_hint": None,
                        "default_assertion_hint": [],
                        "lines": []
                    }
                current_block["lines"].append(line)
                
        if current_block:
            blocks.append(current_block)
            
        for b in blocks:
            b["content"] = "\n".join(b["lines"])
        return blocks

    @staticmethod
    def clean_spacing(text):
        if not text:
            return text
            
        result = []
        chars = list(text)
        n = len(chars)
        
        for i in range(n):
            result.append(chars[i])
            if i < n - 1:
                curr_char = chars[i]
                next_char = chars[i+1]
                
                # Tách chữ thường dính chữ hoa: tạiBệnh -> tại Bệnh
                if curr_char.isalpha() and curr_char.islower() and next_char.isalpha() and next_char.isupper():
                    result.append(' ')
                # Tách số dính chữ: ngày15 -> ngày 15, 2chai -> 2 chai
                elif curr_char.isdigit() and next_char.isalpha():
                    result.append(' ')
                elif curr_char.isalpha() and next_char.isdigit():
                    result.append(' ')
                    
        text = "".join(result)
        
        # Sửa các lỗi dính chữ thường gặp thông dụng và bảo tồn case
        common_errors = {
            "tiềnsử": "tiền sử", "Tiềnsử": "Tiền sử", "TIỀNSỬ": "TIỀN SỬ",
            "bệnhsử": "bệnh sử", "Bệnhsử": "Bệnh sử", "BỆNHSỬ": "BỆNH SỬ",
            "nhậpviện": "nhập viện", "Nhậpviện": "Nhập viện", "NHẬPVIỆN": "NHẬP VIỆN",
            "tiếpsử": "tiếp sử", "Tiếpsử": "Tiếp sử", "TIẾPSỬ": "TIẾP SỬ",
            "dùngthuốc": "dùng thuốc", "Dùngthuốc": "Dùng thuốc", "DÙNGTHUỐC": "DÙNG THUỐC",
            "bịđau": "bị đau", "Bịđau": "Bị đau", "BỊĐAU": "BỊ ĐAU",
            "sửdùng": "sử dùng", "Sửdùng": "Sử dùng", "SỬDỤNG": "SỬ DỤNG"
        }
        for err, corr in common_errors.items():
            text = text.replace(err, corr)
        return text
