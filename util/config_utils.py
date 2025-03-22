import os
import yaml
import importlib
from types import SimpleNamespace
from data.tokenizer import Tokenizer

BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs')
DEFAULT_PATH = os.path.join(BASE_PATH, 'default.yml')

DEFAULT_VOCAB_SIZE = 50257 + 3  # 50257 tokens + 3 special tokens

def dict_to_namespace(d):
    """
    Recursively converts a dictionary to a SimpleNamespace object.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    return d

def merge_configs(config, default):
    result = default.copy()
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = merge_configs(value, result.get(key, {}))
        else:
            result[key] = value
    return result

def load_config(name, vocab_size=DEFAULT_VOCAB_SIZE):
    """
    Loads a config file from 'configs/{name}.yml', fills in missing values 
    from 'configs/default.yml', and returns the merged configuration as a SimpleNamespace object.
    """
    config_path = os.path.join(BASE_PATH, f"{name}.yml")
    
    with open(DEFAULT_PATH, 'r') as f:
        default = dict_to_namespace(yaml.safe_load(f))

    with open(config_path, 'r') as f:
        config = dict_to_namespace(yaml.safe_load(f))
        
    config = merge_configs(config, default)
    config.model['vocab_size'] = vocab_size

    return config

def load_model_from_config(config):
    """
    Dynamically imports and initializes a model based on the config.
    The model type is specified in config.model.type.
    The corresponding module is expected to be in the 'models' folder.
    The class name is assumed to match the module name (normalized to remove hyphens
    and adjusted in capitalization).
    """
    model_type = getattr(config.model, 'type', None)
    if not model_type:
        raise ValueError("Config must specify a model type in 'model.type'.")
    
    model_type = model_type.strip().replace('-', '').lower()

    try:
        module = importlib.import_module(f"models.{model_type}")
    except ImportError as e:
        raise ImportError(f"Could not import module for model type '{model_type}'.") from e

    model_class = None

    for attr in dir(module):
        if attr.lower() == model_type:
            model_class = getattr(module, attr)
            break
    
    if model_class is None:
        raise AttributeError(f"Could not find model class in module '{model_type}'.")    

    model = model_class(config.model)
    model.config = config
    return model
