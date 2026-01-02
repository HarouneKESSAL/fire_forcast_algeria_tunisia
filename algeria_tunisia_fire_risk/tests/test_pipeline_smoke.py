"""Smoke tests for pipeline modules.
Run quickly to ensure imports and key functions do not raise immediately.
"""
from pathlib import Path
import importlib.util

MODULE = Path(__file__).resolve().parent.parent / "scripts" / "geo_pipeline.py"

def test_import_geo_pipeline():
    spec = importlib.util.spec_from_file_location("geo_pipeline", MODULE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    assert hasattr(mod, "process_climate")
    assert hasattr(mod, "process_elevation")
    assert hasattr(mod, "process_landcover")
    assert hasattr(mod, "build_climate_nc_2024")

def test_config_loaded():
    from scripts.config import get_config
    cfg = get_config()
    assert "base_path" in cfg
