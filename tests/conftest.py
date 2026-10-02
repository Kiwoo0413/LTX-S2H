"""
tests/conftest.py
Pytest configuration for LTX_SDR_TO_HDR tests.
"""

import sys
from pathlib import Path

# Add library root to sys.path
lib_root = Path(__file__).resolve().parent.parent
if str(lib_root) not in sys.path:
    sys.path.insert(0, str(lib_root))
