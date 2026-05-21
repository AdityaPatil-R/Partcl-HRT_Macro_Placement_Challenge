"""
Central import point for the external PlacementCost dependency.

The TILOS MacroPlacement plc_client_os module lives in the git submodule at
external/MacroPlacement/CodeElements/Plc_client/. This module adds that path
*once* so the rest of the package can simply do:

    from macro_place._plc import PlacementCost
"""

import sys
import importlib.util
from pathlib import Path

_PLC_CLIENT_DIR = Path(__file__).resolve().parent.parent / "external" / "MacroPlacement" / "CodeElements" / "Plc_client"

# Force-load the pure Python version (.py) instead of the Cython .so, which
# requires the unavailable `circuit_training` package.
_py_path = _PLC_CLIENT_DIR / "plc_client_os.py"
_spec = importlib.util.spec_from_file_location("plc_client_os", str(_py_path))
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("plc_client_os", _mod)
_spec.loader.exec_module(_mod)

from plc_client_os import PlacementCost  # noqa  # type: ignore
__all__ = ["PlacementCost"]
