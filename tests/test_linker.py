from src.linker import CandidateLinker

def test_abbreviation_expansion():
    text = "Bệnh nhân bị COPD nặng."
    expanded = CandidateLinker.expand_abbreviations(text)
    assert "bệnh phổi tắc nghẽn mạn tính" in expanded.lower()

def test_dictionary_exact_match():
    # Mock dictionary lookup
    entity = {"text": "viêm dạ dày", "type": "CHẨN_ĐOÁN"}
    candidates = CandidateLinker.link(entity)
    # K29 là mã ICD-10 của viêm dạ dày
    assert "K29" in candidates
