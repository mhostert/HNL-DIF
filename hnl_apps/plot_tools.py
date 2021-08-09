import numpy as np
import numpy.ma as ma
import scipy 

import matplotlib.pyplot as plt
from matplotlib.pyplot import *

from matplotlib import rc, rcParams
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib.font_manager import *

from .const import *
from .hnl_tools import *
from .exp import *

fsize=11
rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,1.4*2.3617)	}
rc('text', usetex=True)
rc('font',**{'family':'serif', 'serif': ['Computer Modern Roman']})
rcParams.update(rcparams)
matplotlib.rcParams['hatch.linewidth'] = 0.3

axes_form  = [0.158,0.14,0.81,0.80]
def std_fig(ax_form=axes_form, rasterized=True):
    fig = plt.figure()
    ax = fig.add_axes(axes_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)
    return fig,ax

################################################
def interp_grid(x,y,z, fine_gridx=False, fine_gridy=False, logx=False, logy=False, method='interpolate', smear_stddev=False):

    # default
    if not fine_gridx:
        fine_gridx = 100
    if not fine_gridy:
        fine_gridy = 100

    # log scale x
    if logx:
        xi = np.logspace(np.min(np.log10(x)), np.max(np.log10(x)), fine_gridx)
    else: 
        xi = np.linspace(np.min(x), np.max(x), fine_gridx)
    
    # log scale y
    if logy:
        y = -np.log(y)
        yi = np.logspace(np.min(np.log10(y)), np.max(np.log10(y)), fine_gridy)

    else:
        yi = np.linspace(np.min(y), np.max(y), fine_gridy)

    
    Xi, Yi = np.meshgrid(xi, yi)
    if logy:
        Yi = np.exp(-Yi)

    # triangulation
    if method=='triangulation':
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Zi = interpolator(Xi, Yi)
    
    elif method=='interpolate':
        Zi = scipy.interpolate.griddata((x, y), z,\
                                        (xi[None,:], yi[:,None]),\
                                        method='linear', rescale =True)        
    else:
        print(f"Method {method} not implemented.")
    
    # gaussian smear -- not recommended
    if smear_stddev:
            Zi = scipy.ndimage.filters.gaussian_filter(Zi, smear_stddev, mode='nearest', order = 0, cval=0)
    
    return Xi, Yi, Zi

################################################
# AUXIALIARY FOR RECASTS OF CONSTRAINTS

def get_PS191_limit(x, nevent_for_new_limit):
    
    units = 1e-3 # GeV units
    this_file = 'Nlimits/digitized/PS-191/UeUmu_K.dat'
    m4, Umu4sq = np.genfromtxt(this_file, unpack=True)
    fK = interpolate.interp1d(m4*units, Umu4sq, kind='linear', bounds_error=False, fill_value=1, assume_sorted=False)    

    this_file = 'Nlimits/digitized/PS-191/UeUmu_pi.dat'
    m4, Umu4sq = np.genfromtxt(this_file, unpack=True)
    fpi = interpolate.interp1d(m4*units, Umu4sq, kind='linear', bounds_error=False, fill_value=1, assume_sorted=False)    

    NCscaling = np.sqrt(1/4*(1-4*s2w+8*s2w**2))
    combined = np.amin([fK(x),fpi(x)],axis=0)

    no_bkg = combined/NCscaling
    w_bkg = no_bkg*np.sqrt(nevent_for_new_limit/2.3) # needs some attention
    
    return no_bkg, w_bkg


def rescale_muboone_to_SBN(x, channel='nu_e_e', flavor_struct=[0,1,0], dipoles={}, dark_coupl={}, detector=muboone_numi_absorber):

    this_file = 'digitized/muboone_kelly_machado/PS_eff_avg.dat'
    m4, Umu4sq = np.genfromtxt(this_file, unpack=True)
    Umu4sq *= np.sqrt(2) # from Majorana --> Dirac (only approximate -- neglects effect on efficiencies)
    m4 *= 1e-3
    f = interpolate.interp1d(np.log10(m4), np.log10(Umu4sq), kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=False)    
    
    L = detector.prop['baseline']
    d = detector.prop['length']
    ratio_of_areas = detector.prop['area']/muboone_numi_absorber.prop['area']

    y=[]
    for mN in x:    
    
        # COM energy
        EN = (m_neutral_kaon**2 + mN**2-m_mu**2)/2/m_neutral_kaon
        gamma = EN/mN
        beta = np.sqrt(1.0-1.0/(gamma)**2)

        usqr=10**f(np.log10(mN))

        ctau0_new=c_LIGHT*get_lifetime((mN,usqr), flavor_struct=flavor_struct, dipoles=dipoles, dark_coupl=dark_coupl)
        ctau0_old=c_LIGHT*get_lifetime((mN,usqr), flavor_struct=flavor_struct)

        br_new = get_brs((mN,usqr), channel=channel, flavor_struct=flavor_struct, dipoles=dipoles, dark_coupl=dark_coupl)
        br_old = get_brs((mN,usqr), channel='nu_e_e', flavor_struct=flavor_struct)

        prob_old = d/(usqr*beta*gamma) * (br_old/ctau0_old)
        prob_new = prob_decay_in_interval(L, d, ctau0_new, gamma)*br_new
        
        prob_new = ma.masked_array(data=prob_new,
                        mask = ~(prob_new > 0),
                        fill_value=np.nan)  

        ratio_old_new = prob_old/prob_new.filled()/ratio_of_areas

        y.append(usqr**2*ratio_old_new)

    return np.array(y)

def rescale_muboone_to_SBN_for_dip_vs_usqr(x, mN=0.250, channel='nu_e_e', flavor_struct=[0,1,0], dipoles={}, dark_coupl={}, detector=muboone_numi_absorber):

    this_file = 'digitized/muboone_kelly_machado/PS_eff_avg.dat'
    m4, Umu4sq = np.genfromtxt(this_file, unpack=True)
    m4 *= 1e-3
    Umu4sq *= np.sqrt(2) # from Majorana --> Dirac (only approximate -- neglects effect on efficiencies)
    f = interpolate.interp1d(np.log10(m4), np.log10(Umu4sq), kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=False)    
   
    L = detector.prop['baseline']
    d = detector.prop['length']
    ratio_of_areas = detector.prop['area']/muboone_numi_absorber.prop['area']

    # COM energy
    EN = (m_neutral_kaon**2 + mN**2-m_mu**2)/2/m_neutral_kaon
    gamma = EN/mN
    beta = np.sqrt(1.0-1.0/(gamma)**2)
    
    usqr=10**f(np.log10(mN))
    
    ctau0_old=c_LIGHT*get_lifetime((mN,usqr), flavor_struct=flavor_struct)
    br_old = get_brs((mN,usqr), channel='nu_e_e', flavor_struct=flavor_struct)
    prob_old = d/(usqr*beta*gamma) * (br_old/ctau0_old)
    
    y=[]
    for dmu in x:    

        # update dipole paramters
        dipoles['dip_mu4'] =  dmu

        ctau0_new=c_LIGHT*get_lifetime((mN,usqr), flavor_struct=flavor_struct, dipoles=dipoles, dark_coupl=dark_coupl)
        br_new = get_brs((mN,usqr), channel=channel, flavor_struct=flavor_struct, dipoles=dipoles, dark_coupl=dark_coupl)
        prob_new = prob_decay_in_interval(L, d, ctau0_new, gamma)*br_new

        prob_new = ma.masked_array(data=prob_new,
                        mask = ~(prob_new > 0),
                        fill_value=np.nan)  

        ratio_old_new = prob_old/prob_new.filled()/ratio_of_areas

        y.append(usqr**2*ratio_old_new)

    return np.array(y)    



################################################
# AUXIALIARY FOR KINEMATICAL DISTRIBUTION PLOTS
my_list_of_colors = ['royalblue','deeppink','black','darkorange']
my_list_of_dashes = ['-']*4
my_hatches = ['///////////////']*4


my_kin_vars = ['easy',
            'ee_momentum',
            'eplus',
            'eminus',
            'e_smallest',
            'e_largest',
            'ee_theta',
            'ee_beam_theta',
            'ee_beam_costheta',
            'radius_plus',
            'radius_minus',
            'distance_between_circles_at_10cm',
            'distance_between_circles_at_20cm',
            'distance_between_circles_at_50cm',
            'race_to_b=1cm',
            'ee_mass',]

my_units= [1,
            1,
            1,
            1,
            1,
            1,
            rad_to_deg,
            rad_to_deg,
            1,
            1e-2,
            1e-2,
            1,
            1,
            1,
            1e-2,
            1e3,]

my_var_ranges= [
            (-1,1),
            (0,7),
            (0,7),
            (0,7),
            (0,7),
            (0,7),
            (0,90),
            (0,90),
            (0.98,1.0),
            (0,100),
            (0,100),
            (0,10),
            (0,20),
            (0,50),
            (0,10),
            (0,250),]

my_xlabels  = [r'$(E_{e^+} - E_{e^-})/(E_{e^+} + E_{e^-})$', 
            r'$E_{ee}/$GeV', 
            r'$E_{e^+}/$GeV', 
            r'$E_{e^-}/$GeV', 
            r'$E_{e}^{\rm min}/$GeV', 
            r'$E_{e}^{\rm max}/$GeV', 
            r'$\Delta \theta_{ee}\,(^\circ)$', 
            r'$\theta_{ee}^{\rm beam} \, (^\circ)$', 
            r'$\cos(\theta_{ee}^{\rm beam})$', 
            r'$e^+$ curvature radius (m)', 
            r'$e^-$ curvature radius (m)', 
            r'distance between bent trajectories at 10 cm', 
            r'distance between bent trajectories at 20 cm', 
            r'distance between bent trajectories at 50 cm', 
            r'travel distance to b=1 (cm)', 
            r'$e^+e^-$ invariant mass (MeV)',]

def my_histogram(ax, df, var, color='black',label=r'new', density=True, ls='-', var_range=(0,1), units = 1, hatch='///////'):

    out = ax.hist(df[var, '']*units, 
               bins=30, 
               range=var_range,
               weights=df['weight',''], 
                facecolor=color,
              ls=ls,
               edgecolor=color,
               histtype='step',
                density=density,
                 lw=1.5, zorder=1)

    out = ax.hist(df[var, '']*units, 
               label=label,
               bins=30, 
               range=var_range,
               weights=df['weight',''], 
                facecolor='None',
                hatch=hatch,
               edgecolor=color,
                density=density,
                 lw=0.0,
                 zorder=1,
                 alpha =1)


def error_histogram(ax, df, var, color='black', label=r'new', density=True, ls='-', var_range=(0,1), units = 1, hatch='', cumulative=False):

    w = df['weight','']
    x = df[var, '']*units
    # if cumulative:
        # var_range = (np.min(df[var, ''])*units, np.max(df[var, ''][df[var, '']/df[var, '']==1])*units)
        # range_cut_eff = np.sum(w[(var_range[0] < x) & (x < var_range[1])])/np.sum(w)

    prediction, bin_edges = np.histogram(x,
             range=var_range,
             bins=20,
             weights=w,
            )

    errors2 = np.histogram(x,
                 range=var_range,
                 bins=20,
                 weights=w**2,
                )[0]

    area = np.sum(prediction)
    prediction /= area
    errors = np.sqrt(errors2)/area

    if cumulative=='cum_sum':
        prediction = np.cumsum(prediction)
        errors2 = np.cumsum(errors2)
    elif cumulative=='cum_sum_prior_to':
        prediction = np.cumsum(prediction[::-1])[::-1]
        errors2 = np.cumsum(errors2[::-1])[::-1]


    # plotting
    ax.plot(bin_edges,
             np.append(prediction, [0]),
             ds='steps-post',
             label=label,
             color=color,
             lw=1.5, 
             rasterized=True,
             zorder=1)

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
            rasterized=True,
            zorder=1,
            )
            )
