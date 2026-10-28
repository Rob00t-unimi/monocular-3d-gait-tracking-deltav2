"""
Path validation, project root resolution, and configuration loading utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

DEFAULT_CONFIG_REL_PATH = Path("configs") / "default_config.yaml"
LEGACY_CONFIG_REL_PATH = Path("config.yaml")


def get_project_root() -> Path:
    """Returns the canonical root directory of the repository."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_project_path(path_like: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    Resolves a path relative to the repository root if it is not already absolute
    or if it points to a legacy absolute path containing 'data/Gait_Dataset'.
    """
    root = base_dir if base_dir is not None else get_project_root()
    p = Path(path_like)
    if p.exists():
        return p.resolve()

    raw_str = str(path_like).replace("\\", "/")
    for marker in ("data/Gait_Dataset/", "data/gt_cache/", "data/inference_results/", "configs/", "results/"):
        if marker in raw_str:
            suffix = raw_str.split(marker, 1)[1]
            candidate = root / marker.rstrip("/") / suffix
            return candidate

    if not p.is_absolute():
        return (root / p).resolve()
    return p


def resolve_config_path(config_arg: Optional[Union[str, Path]] = None, base_dir: Optional[Path] = None) -> Path:
    """
    Resolves the pipeline YAML configuration path.
    Supports 'configs/default_config.yaml' by default and falls back seamlessly
    when legacy '--config config.yaml' is passed.
    """
    root = base_dir if base_dir is not None else get_project_root()
    if config_arg is None:
        return root / DEFAULT_CONFIG_REL_PATH

    cfg_path = Path(config_arg)
    if cfg_path.is_file():
        return cfg_path.resolve()

    rel_candidate = root / cfg_path
    if rel_candidate.is_file():
        return rel_candidate.resolve()

    if cfg_path.name in ("config.yaml", "default_config.yaml"):
        fallback = root / DEFAULT_CONFIG_REL_PATH
        if fallback.is_file():
            return fallback.resolve()

    return rel_candidate


def validate_file_exists(filepath: Union[str, Path], description: str = "File") -> Path:
    """Validates that a file exists on disk, raising FileNotFoundError otherwise."""
    resolved = resolve_project_path(filepath)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found at: {resolved}")
    return resolved


def validate_directory_exists(dirpath: Union[str, Path], description: str = "Directory") -> Path:
    """Validates that a directory exists on disk, raising FileNotFoundError otherwise."""
    resolved = resolve_project_path(dirpath)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} not found at: {resolved}")
    return resolved


def load_pipeline_config(config_arg: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads the YAML configuration dictionary from disk."""
    cfg_path = resolve_config_path(config_arg)
    if not cfg_path.is_file():
        return {}
    if yaml is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    return {}
