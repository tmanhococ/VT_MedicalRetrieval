from rapidfuzz.distance import Levenshtein

class PositionResolver:
    @staticmethod
    def resolve(raw_text, entities):
        resolved_entities = []
        for ent in entities:
            ent_text = ent["text"]
            # Tìm kiếm vị trí tối ưu trong raw_text
            best_match = None
            best_score = 0.0
            best_pos = None
            
            n_chars = len(ent_text)
            # Cắt cửa sổ trượt để so sánh độ tương đồng
            for i in range(max(0, len(raw_text) - n_chars + 5)):
                for l in range(max(1, n_chars - 5), n_chars + 5):
                    if i + l > len(raw_text):
                        continue
                    candidate = raw_text[i:i+l]
                    # Tính toán edit distance mờ (rapidfuzz)
                    # Giảm trọng số của khoảng trắng bằng cách làm sạch khoảng trắng trước khi đo
                    c_clean = candidate.replace(" ", "")
                    e_clean = ent_text.replace(" ", "")
                    score = Levenshtein.normalized_similarity(c_clean, e_clean)
                    
                    if score > best_score:
                        best_score = score
                        best_match = candidate
                        best_pos = [i, i+l]
                    elif score == best_score and best_score > 0.0:
                        # Tie-breaker: prefer the candidate with higher spacing/length similarity
                        curr_space_score = Levenshtein.normalized_similarity(candidate, ent_text)
                        best_space_score = Levenshtein.normalized_similarity(best_match, ent_text)
                        if curr_space_score > best_space_score:
                            best_match = candidate
                            best_pos = [i, i+l]
                            
            if best_score >= 0.85:
                ent_copy = ent.copy()
                ent_copy["text"] = best_match # Trả về chuỗi nguyên bản gốc
                ent_copy["position"] = best_pos
                resolved_entities.append(ent_copy)
            else:
                # Fallback lấy exact index nếu khớp hoàn hảo
                idx = raw_text.find(ent_text)
                if idx != -1:
                    ent_copy = ent.copy()
                    ent_copy["position"] = [idx, idx + len(ent_text)]
                    resolved_entities.append(ent_copy)
        return resolved_entities
