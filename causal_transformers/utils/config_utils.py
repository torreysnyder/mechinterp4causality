from functools import wraps
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from hydra import main
from causal_transformers.paths import CONFIG_PATH


def get_config_entry(cfg, key_string):
    keys = key_string.split('.')
    value = cfg
    for key in keys:
        value = getattr(value, key, None)
        if value is None:
            return ''
        return value


def pretty_print_cfg(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


def load_experiment_config(experiment_name):
    with initialize(config_path="../../config/experiments", version_base=None):
        return compose(config_name=experiment_name)


def load_config(cfg_dict: dict = None):
    if cfg_dict is None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="config")
    else:
        cfg = OmegaConf.create(cfg_dict)
    return cfg


CONFIG_NAME = "config"
def hydra_config(func):
    @wraps(func)
    @main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
