import importlib.util
import os

config_path = os.environ.get("CONFIG", "configs.py")
print(f"Loading config from {config_path}")
spec = importlib.util.spec_from_file_location("configs", config_path)
configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configs)
