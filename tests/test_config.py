import os
from src.config import Config

def test_config_paths():
    assert Config.DATA_DIR == os.path.abspath("data")
    assert Config.INPUT_DIR == os.path.abspath(os.path.join("data", "input"))
    assert Config.DICT_DIR == os.path.abspath(os.path.join("data", "dictionaries"))
    assert isinstance(Config.LLM_MODEL_PATH, str)
