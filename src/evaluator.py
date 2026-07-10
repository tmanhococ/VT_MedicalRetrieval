import os
import json
from jiwer import wer
from src.config import Config
from main import run_pipeline_for_text

def evaluate():
    gt_dir = os.path.join(Config.DATA_DIR, "ground_truth_7")
    input_dir = Config.INPUT_DIR
    
    if not os.path.exists(gt_dir):
        print("Chưa có nhãn ground truth 7 file để test.")
        return
        
    total_wer = 0
    count = 0
    for filename in os.listdir(gt_dir):
        if filename.endswith(".json"):
            idx = filename.split(".")[0]
            txt_file = os.path.join(input_dir, f"{idx}.txt")
            if not os.path.exists(txt_file):
                continue
                
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
            with open(os.path.join(gt_dir, filename), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                
            pred_data = run_pipeline_for_text(text)
            
            # Simple calculation of text match similarity via word error rate
            gt_texts = " ".join([e["text"] for e in gt_data])
            pred_texts = " ".join([e["text"] for e in pred_data])
            if not gt_texts:
                # If both are empty, WER is 0.0 (no error). If predictions exist, WER is 1.0 (maximum error).
                wer_score = 0.0 if not pred_texts else 1.0
            else:
                wer_score = wer(gt_texts, pred_texts)
            total_wer += wer_score
            count += 1
            
    avg_wer = total_wer / count if count > 0 else 0
    print(f"WER trung bình trên tập mẫu: {avg_wer:.4f}")

if __name__ == "__main__":
    evaluate()
