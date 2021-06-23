import numpy as np
from scipy import interpolate

#############################################
# PDG2020 values for constants and SM masses
#############################################

alphaQED = 1.0/137.03599908421 # Fine structure constant at q2 -> 0
e = np.sqrt((4*np.pi)/alphaQED)
Gf =1.16637876e-5 # Fermi constant (GeV^-2)

## MASSES in GeV
m_proton = 0.93827208816 # GeV
m_neutron = 0.93956542052 # GeV
m_avg = (m_proton+m_neutron)/2. # GeV

m_W = 80.37912 # GeV
m_Z = 91.187621 # GeV

m_e =  0.5109989500015e-3 # GeV
m_mu =  0.1134289257 # GeV
m_tau =  1.77682 # GeV

# speed of light cm/s
c_LIGHT = 29979245800

# constants for normalization
invm2_to_incm2=1e-4
fb_to_cm2 = 1e-39
NAvo = 6.02214076*1e23
tons_to_nucleons = NAvo*1e6/m_avg
rad_to_deg = 180.0/np.pi
invGeV2_to_cm2 = 3.89379372e-28 # hbar c = 197.3269804e-16 GeV.cm
invGeV_to_cm = np.sqrt(invGeV2_to_cm2)
invGeV_to_s = invGeV_to_cm/c_LIGHT
hb = 6.582119569e-25 # hbar in Gev s

def get_decay_rate_in_s(G):
	return 1.0/G*invGeV_to_s
def get_decay_rate_in_cm(G):
	return 1.0/G*invGeV_to_cm
