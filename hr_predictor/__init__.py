import os
import yaml


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_PATH, "hr_predictor/config.yaml")


def get_config_dict():
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    return config
