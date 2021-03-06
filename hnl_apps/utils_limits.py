import numpy as np
from scipy.special import erf, erfinv
from . import hnl_tools, exp
from matplotlib import pyplot as plt

masses_t2k = np.linspace(140, 490, 36)
masses_all = np.linspace(10, 490, 49)

pot_scale_factor = (6.29+12.34)/12.34
pot_scale_upgrade_factor = 200/12.34
volume_upgrade_factor = (1 + 2*180*70*212 /(170*196*56*3)/2)

main_folder = 'nd280-heavy-neutrino-search-2018_main/'
flux_filename = 'flux.npy'
eff_filename = 'efficiency.npy'

flux = np.load(main_folder+flux_filename)
eff = np.load(main_folder+eff_filename)

nd280_limits_marg = np.loadtxt(main_folder + 'limits_combined/limits_marginalisation.dat')
nd280_limits_prof = np.loadtxt(main_folder + 'limits_combined/limits_profiling.dat')

modes_majorana = np.array([0, 4, 10, 12, 16, 20])
weights_modes_majorana = np.array([1, 1, 0.5, 1, 1, 1])

modes_dirac = np.array([0, 10, 12, 16])
weights_modes_dirac = np.array([1, 0.25, 0.5, 1])

modes_latex_strings = {
    0: r'$K^+ \rightarrow \mu^+ N \rightarrow \mu^+(\mu^- \pi^+)$',
    4: r'$K^+ \rightarrow \mu^+ N \rightarrow \mu^+(\mu^+ \pi^-)$',
    10: r'$K^+ \rightarrow \mu^+ N \rightarrow \mu^+(e^+ e^- \nu_{\mu})$',
    12: r'$K^+ \rightarrow \mu^+ N \rightarrow \mu^+(\mu^- \mu^+ \nu_{\mu})$',
    16: r'$K^+ \rightarrow \mu^+ N \rightarrow \mu^+(\mu^- e^+ \nu_e)$',
    20: r'$K^+ \rightarrow \mu^+ N \rightarrow \mu^+(\mu^+ e^- \nu_e)$',
}

#Bayesian limit with flat prior
def bayesian_upper_limit(n_events, cl=0.9):
    # n_events is the prediction for U^2 = 1
    eta = np.sqrt(n_events)
    return erfinv(cl * erf(eta)) / eta

def frequentist_upper_limit(n_events, masses, threshold=2.3):
    # n_events is the prediction for U^2 = 1
    U2_grid = np.geomspace(1e-10, 1, 100)
    n_events_grid = np.outer(n_events, U2_grid**2)
    aux = plt.contour(masses, U2_grid, n_events_grid.T, levels=(threshold,), alpha=0)
    return aux.allsegs[0][0][:, 0], aux.allsegs[0][0][:, 1]


def n_events_T2K(eff, flux, channels=np.arange(10), modes=np.arange(24), mode_weights=None):
    if mode_weights is None:
        mode_weights = np.ones(len(modes))
    assert len(modes) == len(mode_weights)
    channels = np.asarray(channels)
    modes = np.asarray(modes)
    mode_weights = np.asarray(mode_weights)
    
    nu_channels = channels[channels<5]
    antinu_channels = channels[channels>=5]
    
    n_events = 0
    if len(nu_channels)>0:
        n_events += eff[:, modes, :][:, :, nu_channels] * flux[:, modes, np.newaxis]*mode_weights.reshape(1, len(mode_weights), 1)
    if len(antinu_channels)>0:
        n_events += eff[:, modes, :][:, :, antinu_channels] * flux[:, modes+24, np.newaxis]*mode_weights.reshape(1, len(mode_weights), 1)
    return n_events.sum(axis=(1, 2))

def n_events_T2K_mass_couplings(masses, mixings, eff, channels=np.arange(10), modes=np.arange(24), mode_weights=None):
    if mode_weights is None:
        mode_weights = np.ones(len(modes))
    assert len(modes) == len(mode_weights)
    channels = np.asarray(channels)
    modes = np.asarray(modes)
    mode_weights = np.asarray(mode_weights)

    n_events = []
    
    for mixing in mixings:
        hnl_flux = []
        for mass in masses:
            hnl_flux.append(hnl_tools.get_event_rate_mode((mass/1000, mixing), 
                                                      modes=['nu_e_e', 'nu_e_mu', 'mu_pi', 'nu_mu_mu'],
                                                           flavor_struct=[0.0, 1, 0.0], 
                                                           exp_setup=exp.nd280_fhc))
        hnl_flux = np.squeeze(np.asarray(hnl_flux))
        flux_fit = np.zeros((len(masses_all), flux.shape[1]))

        flux_fit[:, 0] = hnl_flux[:, 2]
        flux_fit[:, 4] = hnl_flux[:, 2]
        flux_fit[:, 10] = hnl_flux[:, 0]
        flux_fit[:, 12] = hnl_flux[:, 3]
        flux_fit[:, 16] = hnl_flux[:, 1]
        flux_fit[:, 20] = hnl_flux[:, 1]
        
        n_events.append(n_events_T2K(eff, flux_fit, channels, modes, mode_weights))

    masses_grid, mixings_grid = np.meshgrid(masses, mixings)
    return masses_grid, mixings_grid, np.asarray(n_events)
