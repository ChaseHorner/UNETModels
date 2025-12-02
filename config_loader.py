import importlib.util
import os

'''
This is where the correct configuration file is loaded from.
It uses the CONFIG environment variable to determine which config file to load.

Use 'export CONFIG=path/to/configs.py' to set the environment variable before running the code.
This is done automatically when using the scheduler in line 32 of scheduler.py.

'''

config_path = os.environ.get("CONFIG", "configs.py")
print(f"Loading config from {config_path}")
spec = importlib.util.spec_from_file_location("configs", config_path)
configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configs)
