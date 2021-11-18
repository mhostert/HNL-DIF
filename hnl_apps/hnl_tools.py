import numpy as np
import numpy.ma as ma

from particle import Particle
from particle import literals as lp

from . import exp
from . import model

from .const import *

# ratio between the matrix elements of M -> N ell / M -> nu ell
def Fm(a,b):
	return a+b-(a-b)**2

# Schrock's function
def rho(a,b):
	return kallen_sqrt(1,a,b)*Fm(a,b)

# Approximate neutrino->HNL correction rescaling
# r_i := m_i^2/M_meson^2
def Rapp(ralpha,rN):
	R = ma.masked_array(data=rho(ralpha,rN)/ralpha/(1.0-ralpha)**2,
						mask = ~(np.sqrt(ralpha)+np.sqrt(rN) < 1.0),
						fill_value=0.0)		
	return R.filled()

def get_Rapp(mN, parent=lp.K_plus, daughter=lp.mu_plus):
	ralpha = (daughter.mass/parent.mass)**2
	rN = (mN/parent.mass*1e3)**2
	return Rapp(ralpha,rN)

# Approximation for HNL flux from neutrino flux
def dphi_dEN_app(dphi_dEnu, EN, Ualpha4SQR, mN, parent=lp.K_plus, daughter=lp.mu_plus):
	return Ualpha4SQR*dphi_dEnu(EN)*get_Rapp(mN, parent=parent, daughter=daughter)

# probability of decay
def prob_decay_in_interval(L, d, ctau0, gamma):
	beta = np.sqrt(1.0-1.0/(gamma)**2)
	# return 
	ell = ctau0*gamma*beta
	tol = (d/ell > 1e-5)
	# probability -- if decay length very long, expand exponentials
	P = ma.masked_array(data=np.exp(-L/ell) * (1 - np.exp(-d/ell)),
						mask = ~tol,
						fill_value=d/ell)		
	return P.filled()

# to be pooled
def get_lifetime(args, flavor_struct=[1.0,1.0,1.0], dipoles={}, dark_coupl = {}):

	M4, USQR = args

	mixings = {'Ue4SQR': USQR*flavor_struct[0], 'Umu4SQR': USQR*flavor_struct[1], 'Utau4SQR': USQR*flavor_struct[2]}
	my_hnl = model.hnl_model(m4=M4, mixings=mixings, dipoles=dipoles, dark_coupl = dark_coupl)
	my_hnl.set_high_level_variables()
	my_hnl.compute_rates()
	
	return my_hnl.ctau0/c_LIGHT

# to be pooled
def get_brs(args, channel='nu_e_e', flavor_struct=[1.0,1.0,1.0], dipoles={}, dark_coupl = {}):

	M4, USQR = args

	mixings = {'Ue4SQR': USQR*flavor_struct[0], 'Umu4SQR': USQR*flavor_struct[1], 'Utau4SQR': USQR*flavor_struct[2]}
	my_hnl = model.hnl_model(m4=M4, mixings=mixings, dipoles=dipoles, dark_coupl = dark_coupl)
	my_hnl.set_high_level_variables()
	my_hnl.compute_rates()
	
	return my_hnl.brs[channel]

# to be pooled
def get_gamma_nuee(args, flavor_struct=[1.0,1.0,1.0], dipoles={}, dark_coupl = {}):

	M4, USQR = args
	
	mixings = {'Ue4SQR': USQR*flavor_struct[0], 'Umu4SQR': USQR*flavor_struct[1], 'Utau4SQR': USQR*flavor_struct[2]}
	my_hnl = model.hnl_model(m4=M4, mixings=mixings, dipoles=dipoles, dark_coupl = dark_coupl)
	my_hnl.set_high_level_variables()
	my_hnl.compute_rates()
	
	return my_hnl.rates['nu_e_e']

# to be pooled
def get_event_rate(args, flavor_struct=[1.0,1.0,1.0], dipoles={}, dark_coupl={}, detector=exp.nd280, eff=True, modes=['nu_e_e']):

	M4, USQR = args
	
	mixings = {'Ue4SQR': USQR*flavor_struct[0], 'Umu4SQR': USQR*flavor_struct[1], 'Utau4SQR': USQR*flavor_struct[2]}
	my_hnl = model.hnl_model(m4=M4, mixings=mixings, dipoles=dipoles, dark_coupl = dark_coupl)
	my_hnl.set_high_level_variables()
	my_hnl.compute_rates()

	
	# iterate over the type of neutrino flux available 
	tot_events = 0
	for this_flavor in detector.prop['flavors']:
		fK = detector.get_flux_func(parent=lp.K_plus, nuflavor = this_flavor)
		fN = lambda pN: dphi_dEN_app(fK, pN, Ualpha4SQR=USQR, mN=M4, parent=lp.K_plus, daughter=lp.mu_plus)

		x = np.linspace(1e-2,10.0,1000)
		dx = (x[1]-x[0])
		
		# integrate N spectrum
		events = 0
		for pN in x:
			#boost
			gamma = np.sqrt(pN**2+my_hnl.m4**2)/my_hnl.m4
			# dN/dpN
			events += prob_decay_in_interval(detector.prop['baseline'], 	
											detector.prop['length'], 
											my_hnl.ctau0, 
											gamma)*fN(pN)		
		tot_events += events*dx

	tot_events *= detector.prop['pots']*detector.prop['area']
	
	events_out = np.zeros(len(modes))
	for i, mode in enumerate(modes):
		events_out[i] = tot_events * my_hnl.brs[mode]
		if eff:
			events_out[i] *= detector.prop[f'eff_{mode}'](M4)
	if len(modes) == 1:
		return events_out[0]
	else:
		return events_out

def get_event_rate_w_mixing_and_dipole(args, m4=0.250, flavor_struct=[1.0,1.0,1.0], dark_coupl = {}, detector = exp.nd280):
	return get_event_rate((m4, args[1]), dipoles={'dip_mu4': args[0]}, flavor_struct=flavor_struct, dark_coupl=dark_coupl, detector=detector) 


## deprecated
# def get_event_rate_mode(args, modes, flavor_struct=[1.0,1.0,1.0], dipoles={}, dark_coupl = {}, detector = exp.nd280):

	M4, USQR = args
	
	mixings = {'Ue4SQR': USQR*flavor_struct[0], 'Umu4SQR': USQR*flavor_struct[1], 'Utau4SQR': USQR*flavor_struct[2]}
	my_hnl = model.hnl_model(m4=M4, mixings=mixings, dipoles=dipoles, dark_coupl = dark_coupl)
	my_hnl.set_high_level_variables()
	my_hnl.compute_rates()

	
	# iterate over the type of neutrino flux available 
	tot_events = 0
	for this_flavor in detector.prop['flavors']:
		fK = detector.get_flux_func(parent=lp.K_plus, nuflavor = this_flavor)
		fN = lambda pN: dphi_dEN_app(fK, pN, Ualpha4SQR=USQR, mN=M4, parent=lp.K_plus, daughter=lp.mu_plus)

		x = np.linspace(1e-2,10.0,1000)
		dx = (x[1]-x[0])
		
		# integrate N spectrum
		events = 0
		for pN in x:
			#boost
			gamma = np.sqrt(pN**2+my_hnl.m4**2)/my_hnl.m4
			# dN/dpN
			events += prob_decay_in_interval(detector.prop['baseline'], 	
											detector.prop['length'], 
											my_hnl.ctau0, 
											gamma)*fN(pN)

		tot_events += events

	norm = detector.prop['pots']*detector.prop['area']
	tot_events *= norm*dx
    
	events_out = np.zeros(len(modes))
	for i, mode in enumerate(modes):
		events_out[i] = tot_events * my_hnl.brs[mode]
	return events_out