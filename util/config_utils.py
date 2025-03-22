import os
import yaml
import importlib
from types import SimpleNamespace
from data.tokenizer import Tokenizer
from copy import deepcopy

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
    result = deepcopy(default)
    for key in config.__dict__:
        value = getattr(config, key)
        if isinstance(value, SimpleNamespace):
            setattr(result, key, merge_configs(value, getattr(default, key)))
        else:
            setattr(result, key, value)
            
    return result

def load_config(name):
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

    return config

def load_model_from_config(config, vocab_size=DEFAULT_VOCAB_SIZE):
    """
    Dynamically imports and initializes a model based on the config.
    The model type is specified in config.model.
    The corresponding module is expected to be in the 'models' folder.
    The class name is assumed to match the module name (normalized to remove hyphens
    and adjusted in capitalization).
    """
    
    if not hasattr(config, 'vocab_size'):
        config.vocab_size = vocab_size
    
    model_type = config.model
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

    model = model_class(config)
    model.config = config
    return model
