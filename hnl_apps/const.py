import numpy as np
from scipy import interpolate
from pathlib import Path
local_dir = Path(__file__).parent


#####################################
# particle names and props
from particle import Particle
from particle import literals as lp

N4 = 'N4'
N5 = 'N5'
N6 = 'N6'

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
m_charged_pion = 0.1396
m_charged_rho = 0.7758

# neutral hadrons
m_neutral_pion = 0.135
m_neutral_eta = 0.5478
m_neutral_rho = 0.7755

m_neutral_B = 5.27958
m_charged_B = 5.27958

m_neutral_kaon = 0.497611
m_charged_kaon = 0.4937
m_charged_kaonstar = 0.892



################################################
# QED
alphaQED = 1.0/137.03599908421 # Fine structure constant at q2 -> 0
eQED = np.sqrt((4*np.pi)/alphaQED)

################################################
# weak sector
Gf =1.16637876e-5 # Fermi constant (GeV^-2)
gweak = np.sqrt(Gf*m_W**2*8/np.sqrt(2))
s2w = 0.22343 # On-shell
sw = np.sqrt(s2w)
cw = np.sqrt(1. - s2w)
gL = -1/2 + s2w
gR = s2w

################################################
# higgs -- https://pdg.lbl.gov/2019/reviews/rpp2018-rev-higgs-boson.pdf
vev_EW = 1/np.sqrt(np.sqrt(2)*Gf)
m_H = 125.10
lamb_quartic = (m_H/vev_EW)**2/2
m_h_potential = - lamb_quartic*vev_EW**2

################################################
# Mesons
f_charged_pion = 0.1307 # GeV
f_charged_kaon = 0.1598 # GeV
f_charged_rho = 0.220 # GeV
f_charged_Kstar = 0.178 # GeV^2
f_charged_D = 0.212 # GeV
f_charged_Ds = 0.249 # GeV

f_neutral_pion = 0.130 # GeV
f_neutral_kaon = 0.164 # GeV
f_neutral_Kstar = 0.178 # GeV^2
f_neutral_B = 0.1909 # GeV
f_charged_B = 0.1871 # GeV
f_neutral_Bs = 0.2272 # GeV
f_neutral_eta = 0.210 # GeV
f_neutral_D = 0.212 # GeV

f_neutral_rho = 0.171 # GeV^2
f_neutral_omega = 0.155 # GeV^2
f_neutral_phi = 0.232 # GeV^2

F_neutral_pion = f_neutral_pion/np.sqrt(2.0)
F_neutral_kaon = f_neutral_kaon/np.sqrt(2.0)
F_neutral_B = f_neutral_B/np.sqrt(2.0)
F_charged_B = f_charged_B/np.sqrt(2.0)
F_neutral_Bs = f_neutral_Bs/np.sqrt(2.0)
F_neutral_eta = f_neutral_eta/np.sqrt(2.0)


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


################################################
# auxiliary functions
def get_decay_rate_in_s(G):
	return 1.0/G*invGeV_to_s
def get_decay_rate_in_cm(G):
	return 1.0/G*invGeV_to_cm

# phase space function
def kallen(a,b,c):
	return (a-b-c)**2 - 4*b*c
def kallen_sqrt(a,b,c):
	return np.sqrt(kallen(a,b,c))

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

def get_doublet(p):
	if in_e_doublet(p):
		return 0
	elif in_mu_doublet(p):
		return 1
	elif in_tau_doublet(p):
		return 2
	else:
		print(f"Could not find doublet of {p.name}.")
		return 0
def same_doublet(p1,p2):
	return get_doublet(p1) == get_doublet(p2)
def same_particle(p1,p2):
	return p1.pdgid.abspid == p2.pdgid.abspid
def is_particle(p):
	return p.pdgid.abspid > 0 
def is_antiparticle(p):
	return p.pdgid.abspid < 0 

