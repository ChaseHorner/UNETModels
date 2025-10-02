import importlib
from . import configs

# assume configs has BASE_MODEL = "unet" or "resnet"
_model = importlib.import_module(f"{__name__}.{configs.BASE_MODEL}")

# re-export everything from that model
globals().update(vars(_model))
