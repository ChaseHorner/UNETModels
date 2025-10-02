import importlib

# 👇 choose one place to define your active config
ACTIVE_CONFIG = "configs_v0"   # change here to switch project-wide

# dynamically load the chosen config
_config = importlib.import_module(f"{__name__}.{ACTIVE_CONFIG}")

# re-export everything from that config file
globals().update(vars(_config))
