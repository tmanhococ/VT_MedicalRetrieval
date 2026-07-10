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

def test_clean_word_joining_advanced():
    # Splitting lower-to-upper transition
    assert "tại Bệnh" in StructuralParser.clean_spacing("tạiBệnh viện")
    
    # Splitting digit-to-letter and letter-to-digit
    assert "ngày 15" in StructuralParser.clean_spacing("ngày15")
    assert "uống 2 chai" in StructuralParser.clean_spacing("uống2chai")
    
    # Case-preservation checks for spacing replacements
    assert "Tiền sử" in StructuralParser.clean_spacing("Tiềnsử dùng thuốc")
    assert "tiền sử" in StructuralParser.clean_spacing("tiềnsử dùng thuốc")
    assert "TIỀN SỬ" in StructuralParser.clean_spacing("TIỀNSỬ DÙNG THUỐC")
    
    # Safe handling of None or empty text
    assert StructuralParser.clean_spacing(None) is None
    assert StructuralParser.clean_spacing("") == ""

def test_segment_by_headers_advanced():
    # Segmenting with the new DIAGNOSIS category
    raw_text = "Chẩn đoán\nBệnh nhân bị viêm phổi.\nCác phát hiện chẩn đoán khác\nChưa ghi nhận gì thêm."
    blocks = StructuralParser.segment(raw_text)
    assert len(blocks) == 2
    assert blocks[0]["section_type"] == "DIAGNOSIS"
    assert blocks[0]["default_type_hint"] == "CHẨN_ĐOÁN"
    assert blocks[1]["section_type"] == "DIAGNOSIS"
    
    # Safe handling of None or empty text
    assert StructuralParser.segment(None) == []
    assert StructuralParser.segment("") == []

def test_segment_preserves_layout():
    # Do not discard empty lines
    raw_text = "1. Tiền sử bệnh\n\nBệnh nhân đau đầu.\n\n2. Bệnh sử hiện tại"
    blocks = StructuralParser.segment(raw_text)
    assert len(blocks) == 2
    assert "\n\n" in blocks[0]["content"]
