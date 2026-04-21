# utils/config.py
import os
import yaml
import logging

def load_config(config_path: str = None, default_config: str = None, logger: logging = None) -> dict:
    ROOT_DIR = _find_root()
    
    if config_path is not None:
        if not os.path.isabs(config_path):
            config_path = os.path.join(ROOT_DIR, config_path)
        if not os.path.exists(config_path):
            logger.warning(f"Config '{config_path}' not found, falling back to default.")
            config_path = None

    if config_path is None:
        if default_config is None:
            raise ValueError("No config file found and no default_config provided.")
        config_path = default_config

    logger.info(f"Loading config from: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

    return config


def _find_root() -> str:
    """
    Walks up from this file until it finds the project root
    (identified by the presence of a configs/ directory).
    """
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(current, 'configs')):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # hit filesystem root
            raise RuntimeError("Could not find project root (no 'configs/' directory found)")
        current = parent


