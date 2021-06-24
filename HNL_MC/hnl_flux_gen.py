import numpy as np
from particle import Particle
from particle import literals as lp

# phase space function
def kallen(a,b,c):
	return (a-b-c)**2 - 4*b*c
def kallen_sqrt(a,b,c):
	return np.sqrt(kallen(a,b,c))

# ratio between the matrix elements of M -> N ell / M -> nu ell
def Fm(a,b):
	return a+b-(a-b)**2

# Schrock function
def rho(a,b):
	return kallen_sqrt(1,a,b)*Fm(a,b)

# Approximate neutrino->HNL correction rescaling
# r_i = m_i/M_meson
def Rapp(ralpha,rN):
	# avoiding eval of bad exp
	if (np.sqrt(ralpha)+np.sqrt(rN) < 1):
		return rho(ralpha,rN)/ralpha/(1.0-ralpha)**2
	else:
		return 0.0

vec_Rapp = np.vectorize(Rapp)
def get_Rapp(mN, parent=lp.K_plus, daughter=lp.mu_plus):
	ralpha = (daughter.mass/parent.mass)**2
	rN = (mN/parent.mass*1e3)**2
	return vec_Rapp(ralpha,rN)

# Approximation for HNL flux from neutrino flux
def dphi_dEN_app(dphi_dEnu, EN, Ualpha4SQR, mN, parent=lp.K_plus, daughter=lp.mu_plus):
	return Ualpha4SQR*dphi_dEnu(EN)*get_Rapp(mN, parent=parent, daughter=daughter)

################ 
# decay rates


