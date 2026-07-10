from src.linker import CandidateLinker

def test_abbreviation_expansion():
    text = "Bệnh nhân bị COPD nặng."
    expanded = CandidateLinker.expand_abbreviations(text)
    assert "bệnh phổi tắc nghẽn mạn tính" in expanded.lower()

def test_abbreviation_expansion_preserves_punctuation():
    text = "COPD. ERCP, bệnh nhân bị COPD?"
    expanded = CandidateLinker.expand_abbreviations(text)
    assert "bệnh phổi tắc nghẽn mạn tính." in expanded.lower()
    assert "nội soi mật tụy ngược dòng," in expanded.lower()
    assert "bệnh phổi tắc nghẽn mạn tính?" in expanded.lower()

def test_dictionary_exact_match():
    # Mock dictionary lookup
    entity = {"text": "viêm dạ dày", "type": "CHẨN_ĐOÁN"}
    candidates = CandidateLinker.link(entity)
    # K29 là mã ICD-10 của viêm dạ dày
    assert "K29" in candidates

def test_link_returns_both_specific_and_base_code():
    entity = {"text": "viêm dạ dày", "type": "CHẨN_ĐOÁN"}
    candidates = CandidateLinker.link(entity)
    assert candidates == ["K29.9", "K29"]

def test_link_supports_rxnorm_drug_lookups():
    # Test med_ingredient lookup
    entity_ingredient = {"text": "some brand name", "type": "THUỐC", "med_ingredient": "omeprazole"}
    candidates_ingredient = CandidateLinker.link(entity_ingredient)
    assert "7646" in candidates_ingredient

    # Test fallback text lookup
    entity_text = {"text": "omeprazole", "type": "THUỐC"}
    candidates_text = CandidateLinker.link(entity_text)
    assert "7646" in candidates_text

def test_link_unmatched_returns_empty():
    entity = {"text": "unmatched term", "type": "CHẨN_ĐOÁN"}
    candidates = CandidateLinker.link(entity)
    assert candidates == []

def test_link_resolves_abbreviation_before_match():
    # If we map "vdd" to "viêm dạ dày" (which is in MOCK_ICD10 as K29.9),
    # CandidateLinker.link should expand "vdd" to "viêm dạ dày" and then match K29.9.
    entity = {"text": "vdd", "type": "CHẨN_ĐOÁN"}
    candidates = CandidateLinker.link(entity)
    assert "K29.9" in candidates

