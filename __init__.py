"""
AMACFA+ - Automatic Macroscopic Fabric and Orientation Analysis (Enhanced)

A Python package for automated analysis of ceramic sherd fabric, including:
- Inclusion detection and measurement
- Void detection and analysis  
- Orientation analysis
- Color analysis
- Enhanced edge detection and masking

Version: 0.2.0
Author: Alec Iacobucci
"""

from .core.detection import *

from .core.analysis import *

from .core.visualization import *

from .core.statistics import *

__version__ = "0.2.0"
__author__ = "Alec Iacobucci"


