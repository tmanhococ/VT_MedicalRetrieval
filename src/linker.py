import os
import json
import sqlite3
from src.config import Config

# Mock dictionaries for Tầng 1
MOCK_ICD10 = {"viêm dạ dày": "K29.9", "viêm phổi thùy": "J18.1", "bóc tách động mạch chủ": "I71.0"}
MOCK_RXNORM = {"omeprazole": "7646", "methadone": "6813", "acetaminophen": "161"}
MOCK_ABBR = {"copd": "bệnh phổi tắc nghẽn mạn tính", "ercp": "nội soi mật tụy ngược dòng"}

class CandidateLinker:
    @staticmethod
    def expand_abbreviations(text):
        words = text.split()
        expanded_words = []
        for w in words:
            w_clean = w.strip(".,;:?!").lower()
            if w_clean in MOCK_ABBR:
                # Retain capitalization if we can or just use lowercase mock expansion
                expanded_words.append(MOCK_ABBR[w_clean])
            else:
                expanded_words.append(w)
        return " ".join(expanded_words)

    @classmethod
    def link(cls, entity):
        ent_text = entity["text"].lower()
        ent_type = entity["type"]
        
        # Tầng 1: Exact Match Dictionary
        if ent_type == "CHẨN_ĐOÁN":
            if ent_text in MOCK_ICD10:
                code = MOCK_ICD10[ent_text]
                if "." in code:
                    return [code, code.split(".")[0]]
                return [code]
        elif ent_type == "THUỐC":
            # Đọc cấu trúc thuốc từ LLM để so khớp database
            med_ing = entity.get("med_ingredient")
            if med_ing and med_ing.lower() in MOCK_RXNORM:
                return [MOCK_RXNORM[med_ing.lower()]]
            if ent_text in MOCK_RXNORM:
                return [MOCK_RXNORM[ent_text]]
                
        # Tầng 2: RRF (BM25 + SapBERT) - Fallback mock
        # Trả về danh sách rỗng nếu không khớp
        return []
