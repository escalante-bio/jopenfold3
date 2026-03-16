# Vendored subset of OpenFold3 (Apache 2.0).
# Only the model definitions and data constants are included.

import importlib.util

import gemmi
from packaging import version

from . import hacks  # noqa: F401

if version.parse(gemmi.__version__) >= version.parse("0.7.3"):
    gemmi.set_leak_warnings(False)
