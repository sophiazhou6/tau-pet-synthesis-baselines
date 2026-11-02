"""The existing Dense-U-Net baseline (denseunet/model.py), registered under the same interface."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from denseunet.model import DenseUNet3D  # noqa: E402,F401
