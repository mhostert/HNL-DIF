import numpy as np
from scipy import interpolate

from particle import Particle
from particle import literals as lp

#############################################
# PDG2020 values for constants and SM masses
#############################################


################################################
## MASSES in GeV
m_proton = 0.93827208816 # GeV
m_neutron = 0.93956542052 # GeV
m_avg = (m_proton+m_neutron)/2. # GeV

m_W = 80.37912 # GeV
m_Z = 91.187621 # GeV

m_e =  0.5109989500015e-3 # GeV
m_mu =  0.1134289257 # GeV
m_tau =  1.77682 # GeV

# charged hadrons
Mcharged_pion = 0.1396
Mcharged_rho = 0.7758

# neutral hadrons
Mneutral_pion = 0.135
Mneutral_eta = 0.5478
Mneutral_rho = 0.7755

Mneutral_B = 5.27958
Mcharged_B = 5.27958

Mneutral_kaon = 0.497611
Mcharged_kaon = 0.4937
Mcharged_kaonstar = 0.892

print(Mcharged_pion - m_mu)
################################################
# QED
alphaQED = 1.0/137.03599908421 # Fine structure constant at q2 -> 0
e = np.sqrt((4*np.pi)/alphaQED)

################################################
# weak sector
Gf =1.16637876e-5 # Fermi constant (GeV^-2)
gweak = np.sqrt(Gf*m_W**2*8/np.sqrt(2))
s2w = 0.22343 # On-shell
sw = np.sqrt(s2w)
cw = np.sqrt(1. - s2w)

################################################
# higgs -- https://pdg.lbl.gov/2019/reviews/rpp2018-rev-higgs-boson.pdf
vev_EW = 1/np.sqrt(np.sqrt(2)*Gf)
m_H = 125.10
lamb_quartic = (m_H/vev_EW)**2/2
m_h_potential = - lamb_quartic*vev_EW**2


################################################
# FORM FACTOR CONSTANTS
gA = 1.26
tau3 = 1

MAG_N = -1.913
MAG_P = 2.792

################################################
# Mesons
fcharged_pion = 0.1307
fcharged_kaon = 0.1598
fcharged_rho = 0.220

fneutral_pion = 0.130
fneutral_kaon = 0.164
fneutral_B = 0.1909
fcharged_B = 0.1871
fneutral_Bs = 0.2272
fneutral_eta = 0.210

Fneutral_pion = fneutral_pion/np.sqrt(2.0)
Fneutral_kaon = fneutral_kaon/np.sqrt(2.0)
Fneutral_B = fneutral_B/np.sqrt(2.0)
Fcharged_B = fcharged_B/np.sqrt(2.0)
Fneutral_Bs = fneutral_Bs/np.sqrt(2.0)
Fneutral_eta = fneutral_eta/np.sqrt(2.0)


################################################
# CKM elements
# PDG2019
lamCKM = 0.22453;
ACKM = 0.836;
rhoBARCKM = 0.122;
etaBARCKM = 0.355;
rhoCKM = rhoBARCKM/(1-lamCKM*lamCKM/2.0);
etaCKM = etaBARCKM/(1-lamCKM*lamCKM/2.0);

s12 = lamCKM;
s23 = ACKM*lamCKM**2;
s13e = ACKM*lamCKM**3*(rhoCKM + 1j*etaCKM)*np.sqrt(1.0 - ACKM**2*lamCKM**4)/(np.sqrt(1.0 - lamCKM**2)*(1.0 - ACKM**2*lamCKM**4*(rhoCKM + 1j*etaCKM) ))
c12 = np.sqrt(1 - s12**2);
c23 = np.sqrt(1 - s23**2);
c13 = np.sqrt(1 - abs(s13e)**2);

Vud = c12*c13;
Vus = s12*c13;
Vub = np.conj(s13e);
Vcs = c12*c23-s12*s23*s13e;
Vcd = -s12*c23-c12*s23*s13e;
Vcb = s23*c13;
Vts = -c12*s23-s12*c23*s13e;
Vtd = s12*s23-c12*c23*s13e;
Vtb = c23*c13;


################################################
# speed of light cm/s
c_LIGHT = 29979245800

################################################
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



################
# DM velocity at FREEZOUT
V_DM_FREEZOUT = 0.3 # c 
# SIGMAV_FREEZOUT = 4.8e-26 # cm^3/s
SIGMAV_FREEZOUT = 6e-26 # cm^3/s
GeV2_to_cm3s = invGeV2_to_cm2*c_LIGHT*1e2




################################################
# auxiliary functions
def get_decay_rate_in_s(G):
	return 1.0/G*invGeV_to_s
def get_decay_rate_in_cm(G):
	return 1.0/G*invGeV_to_cm

################################################
# auxiliary functions -- scikit-hep particles
def in_same_doublet(p1,p2):
	if p1.pdgid.abspid in [11,13,15]:
		return (p2.pdgid.abspid - p1.pdgid.abspid == 1 or p2.pdgid.abspid - p1.pdgid.abspid == 0)
	elif p1.pdgid.abspid in [12,14,16]:
		return (p1.pdgid.abspid - p2.pdgid.abspid == 1 or p1.pdgid.abspid - p2.pdgid.abspid == 0)
	else:
		return None

def in_e_doublet(p):
	return p.pdgid.abspid in [11,12]

def in_mu_doublet(p):
	return p.pdgid.abspid in [13,14]

def in_tau_doublet(p):
	return p.pdgid.abspid in [15,16]
