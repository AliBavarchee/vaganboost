import json
from importlib import resources

try:
    with resources.open_text("vaganboostktf.config", "default_params.json") as f:
        DEFAULT_PARAMS = json.load(f)
except FileNotFoundError:
    DEFAULT_PARAMS = {}

__all__ = ['DEFAULT_PARAMS']