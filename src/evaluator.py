import os
import json
from jiwer import wer
from src.config import Config
from src.extractor import LLMExtractor
from main import run_pipeline_for_text

def evaluate():
    gt_dir = os.path.join(Config.DATA_DIR, "ground_truth_7")
    input_dir = Config.INPUT_DIR
    
    if not os.path.exists(gt_dir):
        print("Chưa có nhãn ground truth 7 file để test.")
        return
        
    extractor = LLMExtractor()
    total_wer = 0
    total_assertions_jaccard = 0
    total_candidates_jaccard = 0
    total_composite = 0
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
                
            pred_data = run_pipeline_for_text(text, extractor=extractor)
            
            gt_texts = " ".join([e["text"] for e in gt_data])
            pred_texts = " ".join([e["text"] for e in pred_data])
            
            if not gt_texts.strip():
                wer_score = 0.0 if not pred_texts.strip() else 1.0
            else:
                wer_score = wer(gt_texts, pred_texts)
            
            text_wer_score = max(0.0, 1.0 - wer_score)
            total_wer += wer_score
            
            # Calculate Jaccard for Assertions and Candidates
            file_assertion_scores = []
            file_candidate_scores = []
            matched_preds = set()
            
            for gt_ent in gt_data:
                match_pred = None
                # Match by position first
                for p_idx, pred_ent in enumerate(pred_data):
                    if p_idx not in matched_preds and pred_ent["position"] == gt_ent["position"]:
                        match_pred = pred_ent
                        matched_preds.add(p_idx)
                        break
                # Fallback to exact text match
                if match_pred is None:
                    for p_idx, pred_ent in enumerate(pred_data):
                        if p_idx not in matched_preds and pred_ent["text"].lower() == gt_ent["text"].lower():
                            match_pred = pred_ent
                            matched_preds.add(p_idx)
                            break
                            
                if match_pred is not None:
                    # Assertions Jaccard
                    set_a = set(gt_ent.get("assertions", []))
                    set_b = set(match_pred.get("assertions", []))
                    if not set_a and not set_b:
                        a_jaccard = 1.0
                    else:
                        a_jaccard = len(set_a & set_b) / len(set_a | set_b)
                    
                    # Candidates Jaccard
                    set_c = set(gt_ent.get("candidates", []))
                    set_d = set(match_pred.get("candidates", []))
                    if not set_c and not set_d:
                        c_jaccard = 1.0
                    else:
                        c_jaccard = len(set_c & set_d) / len(set_c | set_d)
                else:
                    a_jaccard = 0.0
                    c_jaccard = 0.0
                
                file_assertion_scores.append(a_jaccard)
                file_candidate_scores.append(c_jaccard)
                
            # Add 0.0 for unmatched predictions
            for p_idx in range(len(pred_data)):
                if p_idx not in matched_preds:
                    file_assertion_scores.append(0.0)
                    file_candidate_scores.append(0.0)
                    
            if not gt_data and not pred_data:
                assertions_jaccard = 1.0
                candidates_jaccard = 1.0
            else:
                assertions_jaccard = sum(file_assertion_scores) / len(file_assertion_scores)
                candidates_jaccard = sum(file_candidate_scores) / len(file_candidate_scores)
                
            total_assertions_jaccard += assertions_jaccard
            total_candidates_jaccard += candidates_jaccard
            
            # Composite Score
            composite = 0.3 * text_wer_score + 0.3 * assertions_jaccard + 0.4 * candidates_jaccard
            total_composite += composite
            count += 1
            
    avg_wer = total_wer / count if count > 0 else 0
    avg_assertions_jaccard = total_assertions_jaccard / count if count > 0 else 0
    avg_candidates_jaccard = total_candidates_jaccard / count if count > 0 else 0
    avg_composite = total_composite / count if count > 0 else 0
    
    print(f"WER trung bình trên tập mẫu: {avg_wer:.4f}")
    print(f"Text Score (1 - WER) trung bình: {max(0.0, 1.0 - avg_wer):.4f}")
    print(f"Assertions Jaccard trung bình: {avg_assertions_jaccard:.4f}")
    print(f"Candidates Jaccard trung bình: {avg_candidates_jaccard:.4f}")
    print(f"Điểm tổng hợp (Composite Score) trung bình: {avg_composite:.4f}")

if __name__ == "__main__":
    evaluate()
