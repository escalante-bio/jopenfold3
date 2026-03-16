# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from pathlib import Path

# Root of the vendored openfold3 package
_VENDOR_ROOT = Path(__file__).resolve().parents[3]  # .../jopenfold3/_vendor/openfold3
_VENDOR_PREFIX = "jopenfold3._vendor.openfold3"


def _import_all_py_files_from_dir(directory: Path):
    """Imports all Python files in the specified directory."""
    for path in directory.resolve().glob("*.py"):
        rel = path.relative_to(_VENDOR_ROOT)
        module_name = _VENDOR_PREFIX + "." + ".".join(
            list(rel.parent.parts) + [rel.stem]
        )
        importlib.import_module(module_name)


_import_all_py_files_from_dir(
    Path(__file__).parent.parent / Path("framework/single_datasets")
)
