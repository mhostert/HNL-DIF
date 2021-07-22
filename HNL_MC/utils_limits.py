import numpy as np
from scipy.special import erf, erfinv

masses_t2k = np.linspace(140, 490, 36)
masses_all = np.linspace(10, 490, 49)

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

# Extrapolation of efficiencies
def get_T2K_efficiency():
    eff_fit = np.zeros(shape=(len(masses_all), 24, 10))
    for mode in modes_majorana:
        for i in range(10):
            poly = np.polynomial.polynomial.Polynomial.fit(np.append(0.0,masses_t2k[:stop_index_fit]), 
                                                           np.append(0.0,eff[:stop_index_fit, mode, i]), 
                                                           deg=1)
            if (poly(1)-poly(0)) < 0:
                poly = np.polynomial.polynomial.Polynomial.fit(np.append(0.0,masses_t2k[:stop_index_fit]), 
                                                               np.append(0.0,eff[:stop_index_fit, mode, i]), 
                                                               deg=0)

        eff_fit[:, mode, i] = poly(masses_all)
    return eff_fit



#Bayesian limit with flat prior
def bayesian_upper_limit(n_events, cl=0.9):
    eta = np.sqrt(n_events)
    return erfinv(cl * erf(eta)) / eta

def n_events_T2K(eff, flux, channels=np.arange(10), modes=np.arange(24), mode_weights=np.ones(24)):
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