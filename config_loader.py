import importlib.util
import sys
import os

print(__file__)
config_path = os.environ.get("CONFIG", "configs.py")
spec = importlib.util.spec_from_file_location("configs", config_path)
configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configs)
