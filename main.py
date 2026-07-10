import os
import json
import zipfile
from src.parser import StructuralParser
from src.extractor import LLMExtractor
from src.resolver import PositionResolver
from src.verifier import NegExVerifier
from src.linker import CandidateLinker
from src.config import Config

def run_pipeline_for_text(text):
    # Step 1: Parser
    blocks = StructuralParser.segment(text)
    all_entities = []
    extractor = LLMExtractor()
    
    for block in blocks:
        # Tiền xử lý chữ dính cho block gửi đi
        block_clean = block.copy()
        block_clean["content"] = StructuralParser.clean_spacing(block["content"])
        
        # Step 2: Extract thô
        raw_ents = extractor.extract(block_clean)
        if not raw_ents:
            continue
            
        # Step 3: Resolve Position
        resolved_ents = PositionResolver.resolve(text, raw_ents)
        
        # Step 4: NegEx Verifier & Step 5: Linking
        for ent in resolved_ents:
            ent_verified = NegExVerifier.verify_entity(text, ent)
            candidates = CandidateLinker.link(ent_verified)
            
            # Dọn dẹp schema nộp bài
            final_ent = {
                "text": ent_verified["text"],
                "position": ent_verified["position"],
                "type": ent_verified["type"],
                "assertions": ent_verified.get("assertions", []),
                "candidates": candidates
            }
            all_entities.append(final_ent)
            
    return all_entities

def main():
    input_dir = Config.INPUT_DIR
    output_dir = Config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            idx = filename.split(".")[0]
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
            
            result = run_pipeline_for_text(text)
            
            out_file = os.path.join(output_dir, f"{idx}.json")
            with open(out_file, "w", encoding="utf-8") as out_f:
                json.dump(result, out_f, ensure_ascii=False, indent=2)
                
    # Zip output
    zip_path = os.path.join(Config.BASE_DIR, "output.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                zipf.write(os.path.join(output_dir, filename), arcname=os.path.join("output", filename))
    print("Finished. output.zip generated successfully.")

if __name__ == "__main__":
    main()
