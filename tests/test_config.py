import os
from src.config import Config

def test_config_paths():
    assert Config.DATA_DIR == os.path.join(Config.BASE_DIR, "data")
    assert Config.INPUT_DIR == os.path.join(Config.DATA_DIR, "input")
    assert Config.DICT_DIR == os.path.join(Config.DATA_DIR, "dictionaries")
    assert Config.OUTPUT_DIR == os.path.join(Config.BASE_DIR, "output")
    assert isinstance(Config.LLM_MODEL_PATH, str)
    assert Config.LLM_CTX_LENGTH == 2048


