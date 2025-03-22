import os
import yaml
import importlib

BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs')
DEFAULTS_PATH = os.path.join(BASE_PATH, 'defaults.yml')

def merge_dicts(main_dict, default_dict):
    result = default_dict.copy()
    for key, value in main_dict.items():
        if (key in result and isinstance(result[key], dict) and isinstance(value, dict)):
            result[key] = merge_dicts(value, result[key])
        else:
            result[key] = value
    return result

def load_config(name):
    """
    Loads a config file from 'configs/{name}.yml', fills in missing values 
    from 'configs/defaults.yml', and returns the merged configuration as a dict.
    """
    config_path = os.path.join(BASE_PATH, f"{name}.yml")
    
    with open(DEFAULTS_PATH, 'r') as f:
        defaults = yaml.safe_load(f)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return merge_dicts(config, defaults)

def load_model_from_config(config):
    """
    Dynamically imports and initializes a model based on the config.
    The model type is specified in config["model"]["type"].
    The corresponding module is expected to be in the 'models' folder.
    The class name is assumed to match the module name (normalized to remove hyphens
    and adjusted in capitalization).
    """
    model_type = config.get("model", {}).get("type", None)
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

    model = model_class(config)
    model.config = config
    return model