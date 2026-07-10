from src.extractor import LLMExtractor

def test_prompt_generation():
    block = {
        "content": "Bệnh nhân sốt cao và đau đầu.",
        "default_type_hint": "TRIỆU_CHỨNG",
        "default_assertion_hint": []
    }
    prompt = LLMExtractor.build_prompt(block)
    assert "sốt cao" in prompt or "TRIỆU_CHỨNG" in prompt

def test_extract_fallback_when_no_model():
    extractor = LLMExtractor()
    block = {
        "content": "Bệnh nhân sốt cao.",
        "default_type_hint": "TRIỆU_CHỨNG",
        "default_assertion_hint": []
    }
    # Since Config.LLM_MODEL_PATH does not exist, extract should return []
    entities = extractor.extract(block)
    assert entities == []

def test_extract_with_mock_llm():
    extractor = LLMExtractor()
    mock_response = {
        "choices": [
            {
                "text": '[{"text": "sốt cao", "type": "TRIỆU_CHỨNG", "assertions": [], "med_brand": null, "med_ingredient": null, "med_strength": null, "med_form": null}, {"text": "đau đầu", "type": "TRIỆU_CHỨNG", "assertions": [], "med_brand": null, "med_ingredient": null, "med_strength": null, "med_form": null}]'
            }
        ]
    }
    extractor.llm = lambda prompt, max_tokens, grammar: mock_response
    
    block = {
        "content": "Bệnh nhân sốt cao.",
        "default_type_hint": "TRIỆU_CHỨNG",
        "default_assertion_hint": []
    }
    entities = extractor.extract(block)
    assert len(entities) == 2
    assert entities[0]["text"] == "sốt cao"
    assert entities[0]["type"] == "TRIỆU_CHỨNG"
    assert entities[1]["text"] == "đau đầu"
    assert entities[1]["type"] == "TRIỆU_CHỨNG"


def test_extract_invalid_json():
    extractor = LLMExtractor()
    mock_response = {
        "choices": [
            {
                "text": 'invalid json string'
            }
        ]
    }
    extractor.llm = lambda prompt, max_tokens, grammar: mock_response
    
    block = {
        "content": "Bệnh nhân sốt cao.",
        "default_type_hint": "TRIỆU_CHỨNG",
        "default_assertion_hint": []
    }
    entities = extractor.extract(block)
    assert entities == []

