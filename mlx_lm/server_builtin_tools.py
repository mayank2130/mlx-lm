import importlib.util
from pathlib import Path
from typing import Any, List, Optional

_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    path = Path(__file__).resolve().parents[1] / "tools" / "server" / "server-tools.py"
    if not path.exists():
        raise FileNotFoundError(f"Built-in tools module not found at {path}")

    spec = importlib.util.spec_from_file_location("mlx_server_tools", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load built-in tools from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def create_server_tools(enabled_tools: Optional[List[str]] = None):
    module = _load_module()
    tools = module.ServerTools()
    tools.setup(enabled_tools or ["all"])
    return tools


def get_server_tools_class():
    return _load_module().ServerTools
