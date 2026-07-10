from src.verifier import NegExVerifier

def test_negation_detection():
    raw_text = "Bệnh nhân không có biểu hiện sốt cao."
    entity = {"text": "sốt cao", "position": [30, 37], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isNegated" in verified["assertions"]

def test_historical_detection():
    raw_text = "Tiền sử năm 2022 bị viêm gan B."
    entity = {"text": "viêm gan B", "position": [20, 30], "assertions": []}
    verified = NegExVerifier.verify_entity(raw_text, entity)
    assert "isHistorical" in verified["assertions"]
