import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.tri as tri

fsize=11
rc('text', usetex=True)
params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,1.4*2.3617)	}
rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
rcParams.update(params)
axes_form  = [0.18,0.18,0.78,0.74]

################################################
def interp_grid(x,y,z, fine_gridx=50, fine_gridy=50, log=False):

    if log:
        xi = np.logspace(np.min(np.log10(x)), np.max(np.log10(x)), fine_gridx)
        yi = np.logspace(np.min(np.log10(y)), np.max(np.log10(y)), fine_gridy)
    else: 
        xi = np.linspace(np.min(x), np.max(x), fine_gridx)
        yi = np.linspace(np.min(y), np.max(y), fine_gridy)

    # Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)

    Xi, Yi = np.meshgrid(xi, yi)
    Zi = interpolator(Xi, Yi)

    return Xi, Yi, Zi


