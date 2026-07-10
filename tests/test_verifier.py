from src.verifier import NegExVerifier

def test_negation_detection():
    raw_text = "Bệnh nhân không có biểu hiện sốt cao."
    entity = {"text": "sốt cao", "position": [29, 36], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" in verified["assertions"]

def test_historical_detection():
    raw_text = "Tiền sử năm 2022 bị viêm gan B."
    entity = {"text": "viêm gan B", "position": [20, 30], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isHistorical" in verified["assertions"]

def test_capitalized_terminator():
    # Capitalized "Nhưng" should terminate the negation propagation
    raw_text = "Không có sốt Nhưng bệnh nhân bị đau đầu."
    entity = {"text": "đau đầu", "position": [32, 39], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" not in verified["assertions"]

def test_comma_non_terminator():
    # Comma should NOT terminate negation propagation
    raw_text = "Không có sốt, đau đầu."
    entity = {"text": "đau đầu", "position": [14, 21], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" in verified["assertions"]

def test_english_terminators():
    # "but", "however", "and" should terminate propagation
    raw_text = "Không có sốt but bệnh nhân bị đau đầu."
    entity = {"text": "đau đầu", "position": [32, 39], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" not in verified["assertions"]

def test_expanded_negation_triggers():
    # "không" as word boundary
    raw_text = "Bệnh nhân không sốt."
    entity = {"text": "sốt", "position": [15, 18], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" in verified["assertions"]

    # "không bị"
    raw_text = "Bệnh nhân không bị ho."
    entity = {"text": "ho", "position": [18, 20], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" in verified["assertions"]

def test_no_subword_boundary_splits():
    # "vào" should not match terminator "và", so negation propagates
    raw_text = "Không ghi nhận bệnh nhân vào viện vì sốt."
    entity = {"text": "sốt", "position": [37, 40], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" in verified["assertions"]

def test_post_negation_triggers():
    # "sốt xuất huyết âm tính" should be negated
    raw_text = "Bệnh nhân có kết quả xét nghiệm sốt xuất huyết âm tính."
    entity = {"text": "sốt xuất huyết", "position": [32, 46], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" in verified["assertions"]

def test_line_break_terminator():
    # "\n" should terminate propagation
    raw_text = "Không có ho\nSốt cao 39 độ."
    entity = {"text": "Sốt cao", "position": [12, 19], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" not in verified["assertions"]

def test_deep_copy_assertions():
    # Mutating returned entity assertions should not affect original entity assertions
    raw_text = "Không có sốt."
    entity = {"text": "sốt", "position": [9, 12], "assertions": ["existing_assertion"]}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" in verified["assertions"]
    assert "isNegated" not in entity["assertions"]
    assert len(entity["assertions"]) == 1
