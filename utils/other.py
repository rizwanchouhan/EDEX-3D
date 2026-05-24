import sys
from pathlib import Path


def class_from_str(str, module=None, none_on_fail=False) -> type:
    """
    Get a class object from its string representation.
    Args:
        str (str): String representation of the class.
        module (module): Module to search for the class (default is the current module).
        none_on_fail (bool): Return None if the class is not found (default is False).

    Returns:
        type: Class object.

    Raises:
        RuntimeError: If the class is not found and none_on_fail is False.
    """
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")


def get_path_to_assets() -> Path:
    """
    Get the path to the assets directory.
    Returns:
        Path: Path to the assets directory.
    """
    return Path(__file__).parents[1] / "assets"


def get_path_to_externals() -> Path:
    """
    Get the path to the external directory.
    Returns:
        Path: Path to the external directory.
    """
    return Path(__file__).parents[1] / "external"
