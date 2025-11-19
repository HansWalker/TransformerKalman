from __future__ import annotations
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, List
import yaml
def import_from_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    mod = importlib.import_module(cfg["File"])
    cls = getattr(mod, cfg["Class"])

    return [cls, cfg]

def import_from_config(cfg_path: str) -> List[Any]:
    """Import a class from a configuration file.

    Args:
        cfg_path (str): Path to the configuration file.

    Returns a list containing the imported class and corresponding CFG
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        root_cfg = yaml.safe_load(f)
    models  = root_cfg.get("Models", []) or []
    filters = root_cfg.get("Filters", []) or []
    scename = root_cfg.get("Scenario")

    config_paths = root_cfg.get("Config_Dirs", {})

    scenerio_dir = config_paths.get("scenarios_dir")
    model_dir = config_paths.get("model_dir")
    kalman_filters_dir = config_paths.get("kalman_filters_dir")

    config_path = f"{scenerio_dir}/{scename}.yaml"
    scene_import = import_from_cfg(config_path)

    model_imports = {}
    for name in models:
        config_path = f"{model_dir}/{name}.yaml"
        model_imports[name] = import_from_cfg(config_path)
    
    filter_imports = {}
    for name in filters:
        config_path = f"{kalman_filters_dir}/{name}.yaml"
        filter_imports[name] = import_from_cfg(config_path)

    return root_cfg, scene_import, model_imports, filter_imports