from src.parser import StructuralParser

def test_segment_by_headers():
    raw_text = "1. Tiền sử bệnh\nBệnh nhân đau đầu.\n2. Bệnh sử hiện tại\nĐau bụng dữ dội."
    blocks = StructuralParser.segment(raw_text)
    assert len(blocks) == 2
    assert blocks[0]["section_type"] == "HISTORY"
    assert blocks[0]["default_type_hint"] == "CHẨN_ĐOÁN"
    assert "isHistorical" in blocks[0]["default_assertion_hint"]
    assert blocks[1]["section_type"] == "CURRENT_HISTORY"

def test_clean_word_joining():
    text_joined = "Bệnh nhân có tiền sửdùng thuốc và bịđau bụng."
    cleaned = StructuralParser.clean_spacing(text_joined)
    # "sửdùng" -> "sử dùng", "bịđau" -> "bị đau"
    assert "sử dùng" in cleaned
    assert "bị đau" in cleaned
