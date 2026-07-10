import os
import json
from main import run_pipeline_for_text

def test_full_pipeline_run():
    raw_text = "1. Tiền sử bệnh\nBệnh nhân viêm dạ dày. Đã ngừng dùng thuốc omeprazole."
    output_entities = run_pipeline_for_text(raw_text)
    assert len(output_entities) >= 2
    # Thực thể viêm dạ dày
    diagnoses = [e for e in output_entities if e["type"] == "CHẨN_ĐOÁN"]
    assert len(diagnoses) > 0
    # Chú ý: CandidateLinker.link() trả về mảng chứa cả mã chi tiết và mã cha, ví dụ: ["K29.9", "K29"]
    assert "K29.9" in diagnoses[0]["candidates"]
