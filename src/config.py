import os

class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INPUT_DIR = os.path.join(DATA_DIR, "input")
    DICT_DIR = os.path.join(DATA_DIR, "dictionaries")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # Model Config
    LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", os.path.join(DATA_DIR, "models", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
    EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")
    
    # Device Config (Set 0 for CPU, -1 for Colab GPU offload all layers)
    GPU_LAYERS_OFFLOAD = int(os.environ.get("GPU_LAYERS_OFFLOAD", -1 if os.environ.get("COLAB_GPU") else 0))
    
    # Context Length
    LLM_CTX_LENGTH = int(os.environ.get("LLM_CTX_LENGTH", 2048))


