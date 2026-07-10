import os
import json
from src.config import Config

# GBNF Grammar String to enforce JSON schema output
GBNF_GRAMMAR = r"""
root ::= "[" ws ( object ( "," ws object )* )? ws "]"
object ::= "{" ws ( "text" ws ":" ws string "," ws "type" ws ":" ws type_enum "," ws "assertions" ws ":" ws array_str "," ws "med_brand" ws ":" ws (string | "null") "," ws "med_ingredient" ws ":" ws (string | "null") "," ws "med_strength" ws ":" ws (string | "null") "," ws "med_form" ws ":" ws (string | "null") ) "}"
array_str ::= "[" ws ( string ( "," ws string )* )? ws "]"
type_enum ::= "\"TRIỆU_CHỨNG\"" | "\"TÊN_XÉT_NGHIỆM\"" | "\"KẾT_QUẢ_XÉT_NGHIỆM\"" | "\"CHẨN_ĐOÁN\"" | "\"THUỐC\""
string ::= "\"" ([^"\\] | "\\" [\"\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
ws ::= [ \t\n\r]*
"""


class LLMExtractor:
    @staticmethod
    def build_prompt(block):
        content = block["content"]
        type_hint = block["default_type_hint"] or "Không rõ"
        assertion_hint = ", ".join(block["default_assertion_hint"]) if block["default_assertion_hint"] else "Không"
        
        prompt = f"""Bạn là một chuyên gia AI Y tế xuất sắc trong việc phân tích bệnh án tiếng Việt.
Nhiệm vụ của bạn là trích xuất các thực thể y khoa từ đoạn văn bản được cung cấp.

Yêu cầu nghiêm ngặt:
1. Trích xuất đúng nguyên văn (verbatim), không tự ý sửa đổi chính tả hoặc định dạng của thực thể gốc.
2. Phân loại theo 5 nhãn: TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC.
3. Xác định trạng thái Assertions nếu có: isNegated (bị phủ định), isHistorical (tiền sử), isFamily (người thân mắc).
4. Nếu nhãn là THUỐC, hãy tách thêm: med_brand (biệt dược), med_ingredient (hoạt chất), med_strength (hàm lượng), med_form (dạng bào chế). Nếu không có thì để null.

Gợi ý phân khu hiện tại:
- Nhãn ưu tiên: {type_hint}
- Assertion mặc định: {assertion_hint}

Đoạn văn bản cần trích xuất:
\"\"\"{content}\"\"\"

Hãy trả về kết quả dưới dạng danh sách thực thể JSON khớp chính xác schema.
"""
        return prompt

    def __init__(self):
        # Sẽ tải mô hình llama-cpp khi chạy trên môi trường thực tế
        self.llm = None

    def load_model(self):
        if not self.llm and os.path.exists(Config.LLM_MODEL_PATH):
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=Config.LLM_MODEL_PATH,
                n_gpu_layers=Config.GPU_LAYERS_OFFLOAD,
                n_ctx=2048,
                verbose=False
            )

    def extract(self, block):
        self.load_model()
        if not self.llm:
            # Fallback mock data for test or if model doesn't exist yet
            return []
            
        prompt = self.build_prompt(block)
        # Generate with grammar enforcement
        response = self.llm(
            prompt,
            max_tokens=512,
            grammar=GBNF_GRAMMAR
        )
        output_text = response["choices"][0]["text"]
        try:
            entities = json.loads(output_text)
            if not isinstance(entities, list):
                entities = [entities]
            return entities
        except Exception:
            return []
