from gantools.utils import in_ipynb

if not in_ipynb():
    import matplotlib
    matplotlib.use('Agg')

from .plot import *
from .colorize import colorize

from . import audio
