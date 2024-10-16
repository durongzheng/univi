# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "0.0.1"

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

from univi.data.explorer.explorer import Explorer
from univi.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from univi.utils import ASSETS, SETTINGS
from univi.utils.checks import check_yolo as checks
from univi.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
