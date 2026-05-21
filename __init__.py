"""
AMFOrA - Automatic Macroscopic Fabric and Orientation Analysis

A Python package for automated analysis of ceramic sherd fabric, including:
- Inclusion detection and measurement using individual color channels for improved detection accuracy across varied pastes
- Void detection and analysis  
- Orientation analysis
- Color analysis
- Enhanced edge detection and masking

Version: 1.0.0
Author: Alec Iacobucci
"""

from .core.detection import *

from .core.analysis import *

from .core.visualization import *

from .core.statistics import *

__version__ = "1.0.0"
__author__ = "Alec Iacobucci"


