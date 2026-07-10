from src.resolver import PositionResolver

def test_resolve_position_exact():
    raw_text = "Bệnh nhân có sốt cao đột ngột."
    # LLM sinh "sốt cao"
    entities = [{"text": "sốt cao", "type": "TRIỆU_CHỨNG"}]
    resolved = PositionResolver.resolve(raw_text, entities)
    assert len(resolved) == 1
    assert resolved[0]["position"] == [13, 20]
    assert resolved[0]["text"] == "sốt cao"

def test_resolve_position_joined_words():
    raw_text = "Tiền sửdùng thuốc methadonekéo dài."
    # LLM sinh "dùng thuốc methadone kéo dài"
    entities = [{"text": "dùng thuốc methadone kéo dài", "type": "THUỐC"}]
    resolved = PositionResolver.resolve(raw_text, entities)
    assert len(resolved) == 1
    # Khớp vị trí của "dùng thuốc methadonekéo dài" trong gốc
    assert resolved[0]["text"] == "dùng thuốc methadonekéo dài"
    assert resolved[0]["position"] == [7, 34]

def test_resolve_position_no_match():
    raw_text = "Bệnh nhân bình thường."
    entities = [{"text": "ho nhiều", "type": "TRIỆU_CHỨNG"}]
    resolved = PositionResolver.resolve(raw_text, entities)
    assert len(resolved) == 0

def test_resolve_position_short_text():
    raw_text = "Bệnh nhân ho."
    entities = [{"text": "ho", "type": "TRIỆU_CHỨNG"}]
    resolved = PositionResolver.resolve(raw_text, entities)
    assert len(resolved) == 1
    assert resolved[0]["position"] == [10, 12]
    assert resolved[0]["text"] == "ho"
