import importlib.util
import inspect
import sys
from pathlib import Path
from typing import List, Type

from .llm_methods import PipelineStep


def load_plugins(plugin_dir: str) -> List[Type[PipelineStep]]:
    """Load PipelineStep subclasses from a directory."""
    steps: List[Type[PipelineStep]] = []
    path = Path(plugin_dir)
    if not path.exists():
        return steps
    for file in path.glob("*.py"):
        module_name = f"plugin_{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        if not spec or not spec.loader:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, PipelineStep) and obj is not PipelineStep:
                steps.append(obj)
    return steps
