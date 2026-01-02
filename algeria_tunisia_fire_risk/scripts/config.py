"""Configuration loader for geo pipeline.
Reads config.yaml if present else falls back to defaults.
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "base_path": "/home/swift/Desktop/DATA",
}

def load_config(conf_path: Path | None = None) -> Dict[str, Any]:
    conf_path = conf_path or Path(DEFAULTS["base_path"]) / "config.yaml"
    if conf_path.exists():
        with open(conf_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    # shallow merge defaults
    merged = {**DEFAULTS, **data}
    return merged

# Convenience accessors
_cfg_cache: Dict[str, Any] | None = None

def get_config() -> Dict[str, Any]:
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = load_config()
    return _cfg_cache

def get_base() -> Path:
    base = Path(get_config()["base_path"]).resolve()
    if not base.exists():
        # Fallback to current working directory (workspace root) when default path is not present
        return Path.cwd().resolve()
    return base
