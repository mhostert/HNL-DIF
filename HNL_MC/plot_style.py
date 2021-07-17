import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.tri as tri

fsize=11
rc('text', usetex=False)
params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,1.4*2.3617)	}
rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
rcParams.update(params)
axes_form  = [0.18,0.18,0.78,0.74]
matplotlib.rcParams['hatch.linewidth'] = 0.3  # previous pdf hatch linewidth

def get_fig():
    fig = plt.figure()
    ax = fig.add_axes(axes_form)
    return fig,ax

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


def my_histogram(ax, df, var, color='black',label=r'new', density=True, ls='-', var_range=(0,1), units = 1, hatch=''):

    out = ax.hist(df[var, '']*units, 
               bins=10, 
               range=var_range,
               weights=df['weight',], 
                facecolor=color,
              ls=ls,
               edgecolor=color,
               histtype='step',
                density=density,
                 lw=0.5, zorder=10)

    out = ax.hist(df[var, '']*units, 
               label=label,
               bins=10, 
               range=var_range,
               weights=df['weight',], 
                facecolor='None',
                hatch=hatch,
               edgecolor=color,
                density=density,
                 lw=0.0,
                 alpha =1)    