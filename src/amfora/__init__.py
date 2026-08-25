"""
AMFOrA - Automatic Macroscopic Fabric and Orientation Analysis

A Python package for automated analysis of ceramic sherd fabric, including:
- Inclusion detection and measurement using individual color channels for improved detection accuracy across varied pastes
- Void detection and analysis
- Orientation analysis
- Color analysis
- Enhanced edge detection and masking

Author: Alec Iacobucci
"""

from importlib.metadata import PackageNotFoundError, version

from .core.analysis import *
from .core.detection import *
from .core.statistics import *
from .core.visualization import *

try:
    # Single source of truth: the version declared in pyproject.toml. Bump it there only.
    __version__ = version("amfora")
except PackageNotFoundError:  # running from an uninstalled source checkout
    __version__ = "0.0.0+dev"
__author__ = "Alec Iacobucci"
