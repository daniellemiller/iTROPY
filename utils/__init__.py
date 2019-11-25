# import itropy objects
from .fig_utils import *
from .stats_utils import *
from .utils import *

__docformat__ = "restructuredtext"


# Let users know if they're missing any module
hard_dependencies = ("numpy", "RNA", "tqdm", "Bio", "sklearn", "statsmodels", "seaborn", "matplotlib", "keras")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append("{0}: {1}".format(dependency, str(e)))

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies))

__all__ = ["pipeline", "dropplot", "profileplot"]

__doc__ = "type here doc"