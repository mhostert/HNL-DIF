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

# Schrock function
def rho(a,b):
	return kallen_sqrt(1,a,b)*Fm(a,b)

# Approximate neutrino->HNL correction rescaling
# r_i = m_i/M_meson
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
	# return np.exp( -L/ctau0/gamma/beta ) * ( 1 - np.exp( -d/ctau0/gamma/beta ) )
	return d/ctau0/gamma/beta

# to be pooled
def get_lifetime(args, flavor_struct=[1.0,1.0,1.0]):

	M4, USQR = args
	
	my_hnl = model.hnl_model(m4=M4, mixings=USQR*np.array(flavor_struct))
	my_hnl.set_high_level_variables()
	my_hnl.compute_rates()
	
	return my_hnl.ctau0

# to be pooled
def get_event_rate(args, flavor_struct=[1.0,1.0,1.0], exp_setup = exp.ND280_FHC):

	M4, USQR = args
	
	my_hnl = model.hnl_model(m4=M4, mixings=USQR*np.array(flavor_struct))
	my_hnl.set_high_level_variables()
	my_hnl.compute_rates()

	my_exp = exp.experiment(exp_setup)
	
	# iterate over the type of neutrino flux available 
	tot_events = 0
	for this_flavor in my_exp.prop['flavors']:
		fK = my_exp.get_flux_func(parent=lp.K_plus, nuflavor = this_flavor)
		fN = lambda pN: dphi_dEN_app(fK, pN, Ualpha4SQR=USQR, mN=M4, parent=lp.K_plus, daughter=lp.mu_plus)

		x = np.linspace(1e-2,10.0,1000)
		dx = (x[1]-x[0])
		
		# integrate N spectrum
		events = 0
		norm = my_exp.prop['pots']*my_exp.prop['area']*my_exp.prop['eff'](M4)
		for pN in x:
			#boost
			gamma = np.sqrt(pN**2+my_hnl.m4**2)/my_hnl.m4
			# dN/dpN
			events += prob_decay_in_interval(my_exp.prop['baseline'], 	
											my_exp.prop['length'], 
											my_hnl.ctau0, 
											gamma)*\
			fN(pN)*(my_hnl.brs['nu_e_e']*0.10+my_hnl.brs['nu_e_mu']*0.15+my_hnl.brs['nu_mu_mu']*0.15+my_hnl.brs['mu_pi']*0.3)\
			*2 # for majorana

		tot_events += events*norm*dx

	return tot_events

def get_event_rate_mode(args, modes, flavor_struct=[1.0,1.0,1.0], exp_setup = exp.ND280_FHC):

	M4, USQR = args
	
	my_hnl = model.hnl_model(m4=M4, mixings=USQR*np.array(flavor_struct))
	my_hnl.set_high_level_variables()
	my_hnl.compute_rates()

	my_exp = exp.experiment(exp_setup)
	
	# iterate over the type of neutrino flux available 
	tot_events = 0
	for this_flavor in my_exp.prop['flavors']:
		fK = my_exp.get_flux_func(parent=lp.K_plus, nuflavor = this_flavor)
		fN = lambda pN: dphi_dEN_app(fK, pN, Ualpha4SQR=USQR, mN=M4, parent=lp.K_plus, daughter=lp.mu_plus)

		x = np.linspace(1e-2,10.0,1000)
		dx = (x[1]-x[0])
		
		# integrate N spectrum
		events = 0
		norm = my_exp.prop['pots']*my_exp.prop['area']
		for pN in x:
			#boost
			gamma = np.sqrt(pN**2+my_hnl.m4**2)/my_hnl.m4
			# dN/dpN
			events += prob_decay_in_interval(my_exp.prop['baseline'], 	
											my_exp.prop['length'], 
											my_hnl.ctau0, 
											gamma)*fN(pN)

		tot_events += events
    
	tot_events *= norm*dx
    
	events = np.zeros(len(modes))
	for i, mode in enumerate(modes):
		events[i] = tot_events * my_hnl.brs[mode]
	return events
