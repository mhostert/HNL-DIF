import numpy as np
from scipy import interpolate
import scipy.stats
import matplotlib.pyplot as plt
from scipy.integrate import quad

from particle import Particle
from particle import literals as lp

# import source 
from .const import *

neutrino4 = 1
def tau_GeV_to_s(decay_rate):
	return 1./decay_rate/1.52/1e24

def L_GeV_to_cm(decay_rate):
	return 1./decay_rate/1.52/1e24*2.998e10

def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c

def I1_2body(x,y):
	return ((1+x-y)*(1+x) - 4*x)*np.sqrt(lam(1.0,x,y))


class N_decay:
	def __init__(self,params):

		self.params=params

	def compute_rates(self):
		
		# Bsm parameters
		params = self.params
		
		##################
		# Neutrino 4
		mh = self.params.m4
		rates = {}
		neutrinos = [lp.nu_e, lp.nu_mu, lp.nu_tau]

		# channels with 3 neutrinos in final state
		rates['R_nu_nu_nu'] = 0.0
		for nu_a in neutrinos:
			for nu_b in neutrinos:
				rates['R_nu_nu_nu'] += nui_nuj_nuk_nuk(params, neutrino4, nu_a, nu_b, nu_b)

		# channels with 1 neutrino in final states
		rates['R_nu_e_e'] = 0
		rates['R_nu_mu_mu'] = 0
		rates['R_nu_e_mu'] = 0
		rates['R_nu_pi'] = 0 
		rates['R_nu_eta'] = 0
		for nu_a in neutrinos:
			# dileptons -- already contains the Delta L = 2 channel
			if mh > 2*lp.e_minus.mass/1e3:
				rates['R_nu_e_e'] = nui_nuj_ell1_ell2(params, neutrino4, nu_a, lp.e_minus, lp.e_plus)
			if mh > lp.e_minus.mass/1e3 + lp.mu_minus.mass/1e3:
				rates['R_nu_e_mu'] = nui_nuj_ell1_ell2(params, neutrino4, nu_a, lp.e_minus, lp.mu_plus)
			if mh > 2*lp.mu_minus.mass/1e3:
				rates['R_nu_mu_mu'] = nui_nuj_ell1_ell2(params, neutrino4, nu_a, lp.mu_minus, lp.mu_plus)
			# pseudoscalar -- neutral current
			if mh > lp.pi_0.mass/1e3:
				rates['R_nu_pi'] = nui_nu_P(params, neutrino4, nu_a, lp.pi_0)
			if mh > lp.eta.mass/1e3:
				rates['R_nu_eta'] = nui_nu_P(params, neutrino4, nu_a, lp.eta)
			

		# CC-only channels	
		# pseudoscalar -- factor of 2 for delta L=2 channel 
		if mh > lp.e_minus.mass/1e3+lp.pi_plus.mass/1e3:
			rates['R_e_pi'] = 2*nui_l_P(params, neutrino4, lp.e_minus, lp.pi_plus)
		if mh > lp.e_minus.mass/1e3+lp.K_plus.mass/1e3:
			rates['R_e_K'] = 2*nui_l_P(params, neutrino4, lp.e_minus, lp.K_plus)
		
		# pseudoscalar -- already contain the Delta L = 2 channel
		if mh > lp.mu_minus.mass/1e3+lp.pi_plus.mass/1e3:
			rates['R_mu_pi'] = 2*nui_l_P(params, neutrino4, lp.mu_minus, lp.pi_plus)
		if mh > lp.mu_minus.mass/1e3+lp.K_plus.mass/1e3:
			rates['R_mu_K'] = 2*nui_l_P(params, neutrino4, lp.mu_minus, lp.K_plus)
	
		self.rates = rates			

	def total_rate(self):
		self.rate_total = np.sum(self.rates.values())
		return self.rate_total

	def compute_BR(self):
		brs = {}
		for channel in rates.keys():
			brs[f'B{channel}'] = self.rates[channel]/self.rate_total
		self.brs = brs


def nui_l_P(params, initial_neutrino, final_lepton, final_hadron):

	mh = params.m4
	mp = final_hadron.mass/1e3
	ml = final_lepton.mass/1e3
	
	# Mixing required for CC N-like
	CC_mixing = np.sqrt(in_tau_doublet(final_lepton)*params.Utau4**2
					+ in_mu_doublet(final_lepton)*params.Umu4**2
					+ in_e_doublet(final_lepton)*params.Ue4**2)
	

	if (final_hadron==lp.pi_plus):
		Vqq = Vud
		fp  = Fcharged_pion
	elif(final_hadron==lp.K_plus):
		Vqq = Vus
		fp  = Fcharged_kaon
	else:
		print(f"Meson {final_hadron.name} no supported.")

	return np.where( mh-mp-ml > 0, (Gf*fp*CC_mixing*Vqq)**2 * mh**3/(16*np.pi) * I1_2body((ml/mh)**2, (mp/mh)**2), 0.0)



def nui_nu_P(params, initial_neutrino, final_neutrino, final_hadron):

	mh = params.m4
	mp = final_hadron.mass/1e3
	
	NC_mixing = np.sqrt(in_tau_doublet(final_neutrino)*params.Utau4**2
					+ in_mu_doublet(final_neutrino)*params.Umu4**2
					+ in_e_doublet(final_neutrino)*params.Ue4**2)

	if (final_hadron==lp.pi_0):
		fp  = Fneutral_pion
	elif(final_hadron==lp.eta):
		fp  = Fneutral_eta


	return np.where( mh - mp > 0, (Gf*fp*NC_mixing)**2*mh**3/(64*np.pi)*(1-(mp/mh)**2)**2, 0)


###############################
# New containing all terms!
def nui_nuj_ell1_ell2(params, initial_neutrino, final_neutrino, final_lepton1, final_lepton2):
	
	################################
	# MASSES
	mh = params.m4
	mf = 0.0

	# NC
	if final_lepton2==final_lepton1:
		
		NCflag=1

		# Which outgoing neutrino?
		Cih = 1./2 * np.sqrt(in_e_doublet(final_neutrino)*params.ce4**2
				+ in_mu_doublet(final_neutrino)*params.cmu4**2
				+ in_tau_doublet(final_neutrino)*params.ctau4**2)

		Dih = 1./2 * np.sqrt(in_e_doublet(final_neutrino)*params.de4**2
				+ in_mu_doublet(final_neutrino)*params.dmu4**2
				+in_tau_doublet(final_neutrino)*params.dtau4**2)
			
		# NC + CC
		if in_same_doublet(final_neutrino,final_lepton1):
			
			# Mixing required for CC N-like
			CC_mixing1 = np.sqrt(in_tau_doublet(final_lepton1)*params.Utau4**2
							+ in_mu_doublet(final_lepton1)*params.Umu4**2
							+ in_e_doublet(final_lepton1)*params.Ue4**2)
			# Mixing required for CC N-like
			CC_mixing2 = np.sqrt(in_tau_doublet(final_lepton2)*params.Utau4**2
							+ in_mu_doublet(final_lepton2)*params.Umu4**2
							+ in_e_doublet(final_lepton2)*params.Ue4**2)
		else:
			CC_mixing1 = 0
			CC_mixing2 = 0
	# CC
	else:
		NCflag=0
		Cih = 0
		Dih = 0

		if in_e_doublet(final_lepton1) and in_mu_doublet(final_lepton2):
			CC_mixing1 = params.Umu4 
			CC_mixing2 = params.Ue4 
		elif in_e_doublet(final_lepton1) and in_tau_doublet(final_lepton2):
			CC_mixing1 = params.Utau4
			CC_mixing2 = params.Ue4
		elif in_mu_doublet(final_lepton1) and in_tau_doublet(final_lepton2):
			CC_mixing1 = params.Umuon4 
			CC_mixing2 = params.Utau4
		else:
			CC_mixing1 = 0
			CC_mixing2 = 0

	#######################################
	#######################################
	### WATCH OUT FOR THE MINUS SIGN HERE -- IMPORTANT FOR INTERFERENCE
	## Put requires mixings in CCflags
	CCflag1 = CC_mixing1
	CCflag2 = -CC_mixing2

	##############################
	# CHARGED LEPTON MASSES (GeV)
	mm = final_lepton1.mass/1e3
	mp = final_lepton2.mass/1e3

	# couplings and masses 	LEADING ORDER!
	Cv = params.ceV
	Ca = params.ceA
	Dv = params.deV
	Da = params.deA
	MZBOSON = m_Z
	MZPRIME = params.Mzprime
	MW = m_W
	g = gweak

	def DGammaDuDt(t,u,mh,mf,mp,mm,NCflag,CCflag1,CCflag2,Cv,Ca,Dv,Da,Cih,Dih,MZBOSON,MZPRIME,MW):
		pi = np.pi
		return (u*((-256*(Ca*Ca)*(Cih*Cih)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (256*(Cih*Cih)*(Cv*Cv)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (64*(Ca*Ca)*(Cih*Cih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (256*(Da*Da)*(Dih*Dih)*mf*mh*mm*mp*(NCflag*NCflag))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (256*(Dih*Dih)*(Dv*Dv)*mf*mh*mm*mp*(NCflag*NCflag))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*(Da*Da)*(Dih*Dih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Dih*Dih)*(Dv*Dv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Ca*Ca)*(Cih*Cih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Da*Da)*(Dih*Dih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Dih*Dih)*(Dv*Dv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (512*Ca*Cih*Da*Dih*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (512*Cih*Cv*Dih*Dv*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (128*Ca*Cih*Da*Dih*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Cih*Cv*Dih*Dv*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Ca*Cih*Da*Dih*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Cih*Cv*Dih*Dv*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (32*Ca*CCflag1*Cih*(g*g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (32*CCflag1*Cih*Cv*(g*g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (8*Ca*CCflag1*Cih*(g*g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(g*g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (32*CCflag1*Da*Dih*(g*g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*CCflag1*Dih*Dv*(g*g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(MW*MW - u)) - (8*CCflag1*Da*Dih*(g*g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(g*g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(g*g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(g*g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(g*g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(g*g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (CCflag1*CCflag1*(g*g*g*g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(g*g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(g*g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(g*g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(g*g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (CCflag1*CCflag1*(g*g*g*g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(g*g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(g*g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(g*g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(g*g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (CCflag2*CCflag2*(g*g*g*g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (CCflag2*CCflag2*(g*g*g*g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (32*Ca*CCflag2*Cih*(g*g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Cih*Cv*(g*g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(g*g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(g*g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (32*CCflag2*Da*Dih*(g*g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Dih*Dv*(g*g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(g*g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(g*g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*Ca*CCflag2*Cih*(g*g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(g*g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Da*Dih*(g*g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(g*g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (2*CCflag1*CCflag2*(g*g*g*g)*mf*mh*(-(mm*mm) - mp*mp + t))/((MW*MW - u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(g*g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(g*g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(g*g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(g*g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(g*g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(g*g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(g*g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(g*g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (CCflag1*CCflag1*(g*g*g*g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MW*MW - u)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(g*g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(g*g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(g*g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(g*g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) - (CCflag2*CCflag2*(g*g*g*g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(g*g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Cih*Cv*(g*g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(g*g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Dih*Dv*(g*g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))))/(512.*mh*(pi*pi*pi)*((t + u)*(t + u)))	
	
	def Sqrt(x):
		return np.sqrt(x)

	tminus = lambda u: ((mh*mh - mm*mm)*(-(mf*mf) + mp*mp) + (mf*mf + mh*mh + mm*mm + mp*mp)*u - u*u - mh*mh*Sqrt(1 + (mf*mf*mf*mf)/(u*u) - (2*(mf*mf)*(mp*mp))/(u*u) + (mp*mp*mp*mp)/(u*u) - (2*(mf*mf))/u - (2*(mp*mp))/u)*u*Sqrt(1 - (2*(mm*mm))/(mh*mh) + (mm*mm*mm*mm)/(mh*mh*mh*mh) - (2*u)/(mh*mh) - (2*(mm*mm)*u)/(mh*mh*mh*mh) + (u*u)/(mh*mh*mh*mh)))/(2.*u)
	tplus = lambda u: ((mh*mh - mm*mm)*(-(mf*mf) + mp*mp) + (mf*mf + mh*mh + mm*mm + mp*mp)*u - u*u + mh*mh*Sqrt(1 + (mf*mf*mf*mf)/(u*u) - (2*(mf*mf)*(mp*mp))/(u*u) + (mp*mp*mp*mp)/(u*u) - (2*(mf*mf))/u - (2*(mp*mp))/u)*u*Sqrt(1 - (2*(mm*mm))/(mh*mh) + (mm*mm*mm*mm)/(mh*mh*mh*mh) - (2*u)/(mh*mh) - (2*(mm*mm)*u)/(mh*mh*mh*mh) + (u*u)/(mh*mh*mh*mh)))/(2.*u)


	integral, error = scipy.integrate.dblquad(	DGammaDuDt,
												(mf+mp)**2,
												(mh-mm)**2, 
												tminus,
												tplus,
												args=(mh,mf,mp,mm,NCflag,CCflag1,CCflag2,Cv,Ca,Dv,Da,Cih,Dih,MZBOSON,MZPRIME,MW),\
												epsabs=1.49e-08, epsrel=1.49e-08)

	return integral




###############################
def nui_nuj_nuk_nuk(params, initial_neutrino, final_neutrinoj, final_neutrinok1, final_neutrinok2):
	mh = params.m4
	
	NC_mixing = np.sqrt(in_tau_doublet(final_neutrinoj)*params.Utau4**2
					+ in_mu_doublet(final_neutrinoj)*params.Umu4**2
					+ in_e_doublet(final_neutrinoj)*params.Ue4**2)

	return Gf**2/288/np.pi**3*mh**5 * NC_mixing**2

	
	# Cki = 0
	# Ckh = 0
	# Dki = 0
	# Dkh = 0
	# Cki2 = 0
	# Ckh2 = 0
	# Dki2 = 0
	# Dkh2 = 0
	# symmetry_factor = 1

	################################
	# COUPLINGS

	# mh = params.m4
	# mm = 0.0
	# mp = 0.0
	# # Which outgoing neutrino?
	# if final_neutrinoj==neutrino_light:
	# 	Cih = params.clight4/2.0
	# 	Dih = params.dlight4/2.0
	# 	mf = 0.0

	# 	if final_neutrinok1==final_neutrinok2:
	# 		Ckk = params.clight/2.0
	# 		Dkk = params.dlight/2.0
	# 		if final_neutrinoj==final_neutrinok2:
	# 			Cki = params.clight/2.0
	# 			Ckh = params.clight4/2.0
	# 			Dki = params.dlight/2.0
	# 			Dkh = params.dlight4/2.0

	# 			Cki2 = params.clight/2.0
	# 			Ckh2 = params.clight4/2.0
	# 			Dki2 = params.dlight/2.0
	# 			Dkh2 = params.dlight4/2.0
	# 			symmetry_factor = 2.0


	# if final_neutrinoj==neutrino_electron:
	# 	Cih = params.ce4/2.0
	# 	Dih = params.de4/2.0*params.Ue4*params.Ue4

	# 	mf = 0.0
	# if final_neutrinoj==neutrino_muon:
	# 	Cih = params.cmu4/2.0
	# 	Dih = params.dmu4/2.0*params.Umu4*params.Umu4
	# 	mf = 0.0
	# if final_neutrinoj==neutrino_tau:
	# 	Cih = params.ctau4/2.0
	# 	Dih = params.dtau4/2.0*params.Utau4*params.Utau4
	# 	mf = 0.0
	# if final_neutrinoj==neutrino4:
	# 	print('ERROR! (nu4 -> nu4 l l) is kinematically not allowed!')

	# # couplings and masses 	LEADING ORDER!
	# MZBOSON = Mz
	# MZPRIME = params.Mzprime


	# def DGammaDuDt(t,u,mh,mf,mp,mm,Cih,Dih,Ckk,Dkk,Ckh,Dkh,Cki,Dki,Ckh2,Dkh2,Cki2,Dki2,MZBOSON,MZPRIME):
	# 	pi = np.pi
	# 	return (u*((-1024*(Cih*Cih)*(Ckk*Ckk)*mf*mh*mm*mp)/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (256*(Cih*Cih)*(Ckk*Ckk)*mm*mp*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (1024*(Dih*Dih)*(Dkk*Dkk)*mf*mh*mm*mp)/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (256*(Dih*Dih)*(Dkk*Dkk)*mm*mp*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (256*(Cih*Cih)*(Ckk*Ckk)*mf*mh*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (256*(Dih*Dih)*(Dkk*Dkk)*mf*mh*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (512*(Ckh2*Ckh2)*(Cki2*Cki2)*mf*mh*mm*mp)/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) - (1024*Cih*Ckh2*Cki2*Ckk*mf*mh*mm*mp)/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) - (256*Cih*Ckh2*Cki2*Ckk*mm*mp*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) + (256*Cih*Ckh2*Cki2*Ckk*mf*mh*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) - (512*(Dkh2*Dkh2)*(Dki2*Dki2)*mf*mh*mm*mp)/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) - (1024*Dih*Dkh2*Dki2*Dkk*mf*mh*mm*mp)/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) - (256*Dih*Dkh2*Dki2*Dkk*mm*mp*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) + (256*Dih*Dkh2*Dki2*Dkk*mf*mh*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) + (128*(Cih*Cih)*(Ckk*Ckk)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (128*(Dih*Dih)*(Dkk*Dkk)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Ckh2*Ckh2)*(Cki2*Cki2)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) + (128*Cih*Ckh2*Cki2*Ckk*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) + (64*(Dkh2*Dkh2)*(Dki2*Dki2)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) + (128*Dih*Dkh2*Dki2*Dkk*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) - (512*(Ckh*Ckh)*(Cki*Cki)*mf*mh*mm*mp)/((-(MZBOSON*MZBOSON) + u)*(-(MZBOSON*MZBOSON) + u)) + (64*(Ckh*Ckh)*(Cki*Cki)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((-(MZBOSON*MZBOSON) + u)*(-(MZBOSON*MZBOSON) + u)) + (1024*Cih*Ckh*Cki*Ckk*mf*mh*mm*mp)/((MZBOSON*MZBOSON - t)*(-(MZBOSON*MZBOSON) + u)) + (256*Cih*Ckh*Cki*Ckk*mm*mp*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(MZBOSON*MZBOSON) + u)) - (256*Cih*Ckh*Cki*Ckk*mf*mh*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(MZBOSON*MZBOSON) + u)) + (256*Ckh*Ckh2*Cki*Cki2*mm*mp*(mf*mf + mh*mh - t))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(-(MZBOSON*MZBOSON) + u)) - (256*Ckh*Ckh2*Cki*Cki2*mf*mh*(-(mm*mm) - mp*mp + t))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(-(MZBOSON*MZBOSON) + u)) - (128*Cih*Ckh*Cki*Ckk*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(MZBOSON*MZBOSON) + u)) - (512*(Dkh*Dkh)*(Dki*Dki)*mf*mh*mm*mp)/((-(MZPRIME*MZPRIME) + u)*(-(MZPRIME*MZPRIME) + u)) + (64*(Dkh*Dkh)*(Dki*Dki)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((-(MZPRIME*MZPRIME) + u)*(-(MZPRIME*MZPRIME) + u)) + (1024*Dih*Dkh*Dki*Dkk*mf*mh*mm*mp)/((MZPRIME*MZPRIME - t)*(-(MZPRIME*MZPRIME) + u)) + (256*Dih*Dkh*Dki*Dkk*mm*mp*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(-(MZPRIME*MZPRIME) + u)) - (256*Dih*Dkh*Dki*Dkk*mf*mh*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(-(MZPRIME*MZPRIME) + u)) + (256*Dkh*Dkh2*Dki*Dki2*mm*mp*(mf*mf + mh*mh - t))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(-(MZPRIME*MZPRIME) + u)) - (256*Dkh*Dkh2*Dki*Dki2*mf*mh*(-(mm*mm) - mp*mp + t))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(-(MZPRIME*MZPRIME) + u)) - (128*Dih*Dkh*Dki*Dkk*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(-(MZPRIME*MZPRIME) + u)) + (128*(Cih*Cih)*(Ckk*Ckk)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (128*(Dih*Dih)*(Dkk*Dkk)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Ckh2*Ckh2)*(Cki2*Cki2)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) + (128*Cih*Ckh2*Cki2*Ckk*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) + (64*(Dkh2*Dkh2)*(Dki2*Dki2)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) + (128*Dih*Dkh2*Dki2*Dkk*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) + (64*(Ckh*Ckh)*(Cki*Cki)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((-(MZBOSON*MZBOSON) + u)*(-(MZBOSON*MZBOSON) + u)) - (128*Cih*Ckh*Cki*Ckk*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(MZBOSON*MZBOSON) + u)) + (64*(Dkh*Dkh)*(Dki*Dki)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((-(MZPRIME*MZPRIME) + u)*(-(MZPRIME*MZPRIME) + u)) - (128*Dih*Dkh*Dki*Dkk*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(-(MZPRIME*MZPRIME) + u)) - (256*Ckh*Cki*mf*mp*(mh*mh + mm*mm - u)*(Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u)))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*((-(MZBOSON*MZBOSON) + u)*(-(MZBOSON*MZBOSON) + u))) + (256*Ckh*Cki*mh*mm*(-(mf*mf) - mp*mp + u)*(Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u)))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*((-(MZBOSON*MZBOSON) + u)*(-(MZBOSON*MZBOSON) + u))) + (256*Cih*Ckk*mf*mp*(mh*mh + mm*mm - u)*(Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u)))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(-(MZBOSON*MZBOSON) + u)) - (256*Cih*Ckk*mh*mp*(mh*mh + mp*mp - t - u)*(Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u)))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(-(MZBOSON*MZBOSON) + u)) - (256*Cih*Ckk*mh*mm*(-(mf*mf) - mp*mp + u)*(Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u)))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(-(MZBOSON*MZBOSON) + u)) + (256*Cih*Ckk*mf*mm*(-(mf*mf) - mm*mm + t + u)*(Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u)))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(-(MZBOSON*MZBOSON) + u)) - (512*mf*mh*mm*mp*((Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u))*(Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u))))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*((-(MZBOSON*MZBOSON) + u)*(-(MZBOSON*MZBOSON) + u))) + (128*(mf*mf + mh*mh - t)*(-(mm*mm) - mp*mp + t)*((Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u))*(Ckh*Cki*(mf*mf) - Ckh2*Cki2*(mf*mf) + Ckh*Cki*(mm*mm) - Ckh2*Cki2*(mp*mp) - Ckh*Cki*(MZBOSON*MZBOSON) + Ckh2*Cki2*(MZBOSON*MZBOSON) + Ckh*Cki*(mh*mh + mp*mp - t - u) - Ckh2*Cki2*(-(mf*mf) - mp*mp + u))))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*((-(MZBOSON*MZBOSON) + u)*(-(MZBOSON*MZBOSON) + u))) + (256*Ckh2*Cki2*mh*mp*(mh*mh + mp*mp - t - u)*(-(Ckh*Cki*(mf*mf)) + Ckh2*Cki2*(mf*mf) - Ckh*Cki*(mm*mm) + Ckh2*Cki2*(mp*mp) + Ckh*Cki*(MZBOSON*MZBOSON) - Ckh2*Cki2*(MZBOSON*MZBOSON) - Ckh*Cki*(mh*mh + mp*mp - t - u) + Ckh2*Cki2*(-(mf*mf) - mp*mp + u)))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(-(MZBOSON*MZBOSON) + u)) - (256*Ckh2*Cki2*mf*mm*(-(mf*mf) - mm*mm + t + u)*(-(Ckh*Cki*(mf*mf)) + Ckh2*Cki2*(mf*mf) - Ckh*Cki*(mm*mm) + Ckh2*Cki2*(mp*mp) + Ckh*Cki*(MZBOSON*MZBOSON) - Ckh2*Cki2*(MZBOSON*MZBOSON) - Ckh*Cki*(mh*mh + mp*mp - t - u) + Ckh2*Cki2*(-(mf*mf) - mp*mp + u)))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(-(MZBOSON*MZBOSON) + u)) - (256*Dkh*Dki*mf*mp*(mh*mh + mm*mm - u)*(Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u)))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*((-(MZPRIME*MZPRIME) + u)*(-(MZPRIME*MZPRIME) + u))) + (256*Dkh*Dki*mh*mm*(-(mf*mf) - mp*mp + u)*(Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u)))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*((-(MZPRIME*MZPRIME) + u)*(-(MZPRIME*MZPRIME) + u))) + (256*Dih*Dkk*mf*mp*(mh*mh + mm*mm - u)*(Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u)))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(-(MZPRIME*MZPRIME) + u)) - (256*Dih*Dkk*mh*mp*(mh*mh + mp*mp - t - u)*(Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u)))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(-(MZPRIME*MZPRIME) + u)) - (256*Dih*Dkk*mh*mm*(-(mf*mf) - mp*mp + u)*(Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u)))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(-(MZPRIME*MZPRIME) + u)) + (256*Dih*Dkk*mf*mm*(-(mf*mf) - mm*mm + t + u)*(Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u)))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(-(MZPRIME*MZPRIME) + u)) - (512*mf*mh*mm*mp*((Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u))*(Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u))))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*((-(MZPRIME*MZPRIME) + u)*(-(MZPRIME*MZPRIME) + u))) + (128*(mf*mf + mh*mh - t)*(-(mm*mm) - mp*mp + t)*((Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u))*(Dkh*Dki*(mf*mf) - Dkh2*Dki2*(mf*mf) + Dkh*Dki*(mm*mm) - Dkh2*Dki2*(mp*mp) - Dkh*Dki*(MZPRIME*MZPRIME) + Dkh2*Dki2*(MZPRIME*MZPRIME) + Dkh*Dki*(mh*mh + mp*mp - t - u) - Dkh2*Dki2*(-(mf*mf) - mp*mp + u))))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*((-(MZPRIME*MZPRIME) + u)*(-(MZPRIME*MZPRIME) + u))) + (256*Dkh2*Dki2*mh*mp*(mh*mh + mp*mp - t - u)*(-(Dkh*Dki*(mf*mf)) + Dkh2*Dki2*(mf*mf) - Dkh*Dki*(mm*mm) + Dkh2*Dki2*(mp*mp) + Dkh*Dki*(MZPRIME*MZPRIME) - Dkh2*Dki2*(MZPRIME*MZPRIME) - Dkh*Dki*(mh*mh + mp*mp - t - u) + Dkh2*Dki2*(-(mf*mf) - mp*mp + u)))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(-(MZPRIME*MZPRIME) + u)) - (256*Dkh2*Dki2*mf*mm*(-(mf*mf) - mm*mm + t + u)*(-(Dkh*Dki*(mf*mf)) + Dkh2*Dki2*(mf*mf) - Dkh*Dki*(mm*mm) + Dkh2*Dki2*(mp*mp) + Dkh*Dki*(MZPRIME*MZPRIME) - Dkh2*Dki2*(MZPRIME*MZPRIME) - Dkh*Dki*(mh*mh + mp*mp - t - u) + Dkh2*Dki2*(-(mf*mf) - mp*mp + u)))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(-(MZPRIME*MZPRIME) + u)) - (128*(Ckh2*Ckh2)*(Cki2*Cki2)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) - (256*Cih*Ckh2*Cki2*Ckk*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZBOSON*MZBOSON - t - u)) - (128*(Dkh2*Dkh2)*(Dki2*Dki2)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) - (256*Dih*Dkh2*Dki2*Dkk*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(mf*mf + mh*mh + mm*mm + mp*mp - MZPRIME*MZPRIME - t - u)) + (128*(Ckh*Ckh)*(Cki*Cki)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((-(MZBOSON*MZBOSON) + u)*(-(MZBOSON*MZBOSON) + u)) - (256*Cih*Ckh*Cki*Ckk*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(MZBOSON*MZBOSON) + u)) + (128*(Dkh*Dkh)*(Dki*Dki)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((-(MZPRIME*MZPRIME) + u)*(-(MZPRIME*MZPRIME) + u)) - (256*Dih*Dkh*Dki*Dkk*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(-(MZPRIME*MZPRIME) + u))))/(512.*mh*(pi*pi*pi)*((t + u)*(t + u)))

	# def Sqrt(x):
	# 	return np.sqrt(x)

	# tminus = lambda u: ((mh*mh - mm*mm)*(-(mf*mf) + mp*mp) + (mf*mf + mh*mh + mm*mm + mp*mp)*u - u*u - mh*mh*Sqrt(1 + (mf*mf*mf*mf)/(u*u) - (2*(mf*mf)*(mp*mp))/(u*u) + (mp*mp*mp*mp)/(u*u) - (2*(mf*mf))/u - (2*(mp*mp))/u)*u*Sqrt(1 - (2*(mm*mm))/(mh*mh) + (mm*mm*mm*mm)/(mh*mh*mh*mh) - (2*u)/(mh*mh) - (2*(mm*mm)*u)/(mh*mh*mh*mh) + (u*u)/(mh*mh*mh*mh)))/(2.*u)
	# tplus = lambda u: ((mh*mh - mm*mm)*(-(mf*mf) + mp*mp) + (mf*mf + mh*mh + mm*mm + mp*mp)*u - u*u + mh*mh*Sqrt(1 + (mf*mf*mf*mf)/(u*u) - (2*(mf*mf)*(mp*mp))/(u*u) + (mp*mp*mp*mp)/(u*u) - (2*(mf*mf))/u - (2*(mp*mp))/u)*u*Sqrt(1 - (2*(mm*mm))/(mh*mh) + (mm*mm*mm*mm)/(mh*mh*mh*mh) - (2*u)/(mh*mh) - (2*(mm*mm)*u)/(mh*mh*mh*mh) + (u*u)/(mh*mh*mh*mh)))/(2.*u)


	# integral, error = scipy.integrate.dblquad(	DGammaDuDt,
	# 											(mf+mp)**2,
	# 											(mh-mm)**2, 
	# 											tminus,
	# 											tplus,
	# 											args=(mh,mf,mp,mm,Cih,Dih,Ckk,Dkk,Ckh,Dkh,Cki,Dki,Ckh2,Dkh2,Cki2,Dki2,MZBOSON,MZPRIME),\
	# 											epsabs=1.49e-08, epsrel=1.49e-08)

	# return integral

