import os
import yaml
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
import logging

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except Exception as e:
        raise e

def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at: {path}")

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    return ConfigBox(content) 