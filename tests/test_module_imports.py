"""Every extracted module must resolve its dependencies inside this project."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(
    '.'.join(path.relative_to(ROOT).with_suffix('').parts)
    for folder in ('src', 'pipeline')
    for path in (ROOT / folder).rglob('*.py')
    if path.name != '__init__.py'
)


@pytest.mark.parametrize('name', MODULES)
def test_standalone_module_imports(name):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(ROOT)
