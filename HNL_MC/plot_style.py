import numpy as np
import scipy 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.tri as tri

from .const import *

fsize=11
rc('text', usetex=False)
params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,1.4*2.3617)	}
rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
rcParams.update(params)
axes_form  = [0.18,0.18,0.78,0.74]
matplotlib.rcParams['hatch.linewidth'] = 0.3  # previous pdf hatch linewidth

def std_fig():
    fig = plt.figure()
    ax = fig.add_axes(axes_form)
    return fig,ax

################################################
def interp_grid(x,y,z, fine_gridx=False, fine_gridy=False, logx=False, logy=False):

    if not fine_gridx:
        fine_gridx = len(x)
    if not fine_gridy:
        fine_gridy = len(y)


    if logx:
        xi = np.logspace(np.min(np.log10(x)), np.max(np.log10(x)), fine_gridx)
    else: 
        xi = np.linspace(np.min(x), np.max(x), fine_gridx)
    if logy:
        yi = np.logspace(np.min(np.log10(y)), np.max(np.log10(y)), fine_gridy)
    else:
        yi = np.linspace(np.min(y), np.max(y), fine_gridy)

    Xi, Yi = np.meshgrid(xi, yi)


    # Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
    

    # triang = tri.Triangulation(x, y)
    # interpolator = tri.LinearTriInterpolator(triang, z)
    # Zi = interpolator(Xi, Yi)


    Zi = scipy.interpolate.griddata((x, y), z,\
                                    (xi[None,:], yi[:,None]),\
                                    method='linear', rescale =True)
    # Zi_g = scipy.ndimage.filters.gaussian_filter(Zi, 0.8, mode='nearest', order = 0, cval=0)

    return Xi, Yi, Zi


def my_histogram(ax, df, var, color='black',label=r'new', density=True, ls='-', var_range=(0,1), units = 1, hatch=''):

    out = ax.hist(df[var, '']*units, 
               bins=30, 
               range=var_range,
               weights=df['weight',], 
                facecolor=color,
              ls=ls,
               edgecolor=color,
               histtype='step',
                density=density,
                 lw=1, zorder=10)

    out = ax.hist(df[var, '']*units, 
               label=label,
               bins=30, 
               range=var_range,
               weights=df['weight',], 
                facecolor='None',
                hatch=hatch,
               edgecolor=color,
                density=density,
                 lw=0.0,
                 alpha =1)    


def error_histogram(ax, df, var, color='black', label=r'new', density=True, ls='-', var_range=(0,1), units = 1, hatch='', cumulative=False):

    w = df['weight',]
    x = df[var, '']*units
    # if cumulative:
    #     var_range = (np.min(df[var, '']), np.max(df[var, '']))

    prediction, bin_edges = np.histogram(x,
             range=var_range,
             bins=50,
             weights=w,
            )

    errors2 = np.histogram(x,
                 range=var_range,
                 bins=50,
                 weights=w**2,
                )[0]
    if cumulative=='cum_sum':
        prediction = np.cumsum(prediction)
        errors2 = np.cumsum(errors2)
    elif cumulative=='cum_sum_prior_to':
        prediction = np.cumsum(prediction[::-1])[::-1]
        errors2 = np.cumsum(errors2[::-1])[::-1]

    area = np.sum(prediction)
    prediction /= area
    errors = np.sqrt(errors2)/area

    # plotting
    ax.plot(bin_edges,
             np.append(prediction, [0]),
             ds='steps-post',
             label=label,
             color=color,
             lw=0.8)

    for edge_left, edge_right, pred, err in zip(bin_edges[:-1], bin_edges[1:], prediction, errors):
        ax.add_patch(
            patches.Rectangle(
            (edge_left, pred-err),
            edge_right-edge_left,
            2 * err,
            color=color,
            hatch=hatch,
            fill=False,
            linewidth=0,
            alpha=0.5,
            )
            )


def get_PS191_limit(x, nevent_for_new_limit):
    units = 1e-3 # GeV units
    this_file = 'Nlimits/digitized/PS-191/UeUmu_K.dat'
    m4, Umu4sq = np.genfromtxt(this_file, unpack=True)
    fK = interpolate.interp1d(m4*units, Umu4sq, kind='linear', bounds_error=False, fill_value=1, assume_sorted=False)    

    this_file = 'Nlimits/digitized/PS-191/UeUmu_pi.dat'
    m4, Umu4sq = np.genfromtxt(this_file, unpack=True)
    fpi = interpolate.interp1d(m4*units, Umu4sq, kind='linear', bounds_error=False, fill_value=1, assume_sorted=False)    

    NCscaling = np.sqrt(gL**2 + gR**2 + gR*gL)
    combined = np.amin([fK(x),fpi(x)],axis=0)

    no_bkg = combined/NCscaling
    w_bkg = combined/NCscaling*np.sqrt(nevent_for_new_limit/2.3) # needs some attention
    
    return no_bkg, w_bkg

def rescale_muboone(mN,Usqr):
    units = 1e-3 # GeV units
    this_file = 'Nlimits/digitized/PS-191/UeUmu_K.dat'
    m4, Umu4sq = np.genfromtxt(this_file, unpack=True)
    fK = interpolate.interp1d(m4*units, Umu4sq, kind='linear', bounds_error=False, fill_value=1, assume_sorted=False)    

    this_file = 'Nlimits/digitized/PS-191/UeUmu_pi.dat'
    m4, Umu4sq = np.genfromtxt(this_file, unpack=True)
    fpi = interpolate.interp1d(m4*units, Umu4sq, kind='linear', bounds_error=False, fill_value=1, assume_sorted=False)    

    NCscaling = np.sqrt(gL**2 + gR**2 + gR*gL)
    combined = np.amin([fK(x),fpi(x)],axis=0)

    no_bkg = combined/NCscaling
    w_bkg = combined/NCscaling*np.sqrt(nevent_for_new_limit/2.3) # needs some attention
    
    return no_bkg, w_bkg




