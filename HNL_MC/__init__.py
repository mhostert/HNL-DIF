import numpy
import os
import pyximport

#CYTHON -- MAC OS X FIX --	FOLLOW https://github.com/cython/cython/issues/1725
numpy_path = numpy.get_include()
os.environ['CFLAGS'] = "-I" + numpy_path
pyximport.install(
	language_level=3,
    pyimport=False,
    setup_args={'include_dirs': numpy.get_include()}
    )

# Handling four vectors
from . import fourvec # python only
from . import Cfourvec as Cfv # cython

from . import hnl_tools
from . import model
from . import rates
from . import nuH_integrands
from . import nuH_MC
from . import exp

from . import printer
from . import plot_style

