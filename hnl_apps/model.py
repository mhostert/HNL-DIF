import numpy as np
import warnings

from .const import * 
from .rates import *


class hnl_model():

	def __init__(self, m4, mixings = {'Umu4SQR': 1.0}, dipoles = {}, dark_coupl={}, HNLtype="dirac"):

		# Dirac or Majorana
		self.HNLtype	= HNLtype

		################################################################
		## std HNL parameters -- N5 not supported atm
		self.Ue4	= np.sqrt(mixings['Ue4SQR']) if 'Ue4SQR' in mixings else mixings['Ue4'] if 'Ue4' in mixings else 0.0
		self.Umu4	= np.sqrt(mixings['Umu4SQR']) if 'Umu4SQR' in mixings else mixings['Umu4'] if 'Umu4' in mixings else 0.0
		self.Utau4	= np.sqrt(mixings['Utau4SQR']) if 'Utau4SQR' in mixings else mixings['Utau4'] if 'Utau4' in mixings else 0.0
		self.Ue5	= np.sqrt(mixings['Ue5SQR']) if 'Ue5SQR' in mixings else mixings['Ue5'] if 'Ue5' in mixings else 0.0
		self.Umu5	= np.sqrt(mixings['Umu5SQR']) if 'Umu5SQR' in mixings else mixings['Umu5'] if 'Umu5' in mixings else 0.0
		self.Utau5	= np.sqrt(mixings['Utau5SQR']) if 'Utau5SQR' in mixings else mixings['Utau5'] if 'Utau5' in mixings else 0.0

		self.m4		= m4
		self.m5		= 1e10

		################################################################
		## dark Z' parameters 
		# convert from four-fermion notation assuming decoupled Z'
		if (not 'GX' in dark_coupl) and (not 'gprime' in dark_coupl):
			self.GX			= 0.0
			self.mzprime	= 1e10
			self.gprime		= 0.0
			self.chi		= 0.0
			self.epsilon	= 0.0
			self.UD4		= 0.0
			self.UD5		= 0.0

		elif 'GX' in dark_coupl:
			if dark_coupl.keys() > {'GX'}:
				warnings.warn("Warning: overriding dark couplings with GX.")

			self.GX = dark_coupl['GX']
			# defining couplings that recover the correct GX
			self.mzprime	= 5.0
			self.chi		= 1e-3 # small to avoid spoiling SM couplings
			self.epsilon 	= cw*self.chi
			self.gprime		= self.mzprime**2*dark_coupl['GX']/np.sqrt(2)/eQED/self.epsilon
			self.UD4		= 1.0
			self.UD5		= 0.0

		else:
			# kinetic mixing parameters
			if 'chi' in dark_coupl:
				self.chi		= dark_coupl['chi']
				self.epsilon 	= cw*self.chi
			elif 'epsilon' in dark_coupl:
				self.epsilon 	= dark_coupl['epsilon']
				self.chi 		= self.epsilon/cw

			self.gprime		= dark_coupl['gprime']
			self.UD4		= dark_coupl['UD4']
			self.UD5		= 0.0
			self.mzprime	= dark_coupl['mzprime']
			self.GX = np.sqrt(2)*self.gprime*self.epsilon*eQED/self.mzprime**2

		################################################################
		## transition dipole parameters -- note that dip = mu_{tr} /2

		self.dip_e4		= dipoles['dip_e4'] if 'dip_e4' in dipoles else 0  # GeV^-1
		self.dip_mu4	= dipoles['dip_mu4'] if 'dip_mu4' in dipoles else 0  # GeV^-1
		self.dip_tau4	= dipoles['dip_tau4'] if 'dip_tau4' in dipoles else 0  # GeV^-1

		self.dij        = np.sqrt(self.dip_e4**2+self.dip_mu4**2+self.dip_tau4**2)

		# cut on the e+e- invariant mass for dipole decays
		self.cut_ee     = dipoles['cut_ee'] if 'cut_ee' in dipoles else 2*m_e 


		################################################################
		## ALP parameters 
		self.inv_f_alp 	= dark_coupl['inv_f_alp']	if 'inv_f_alp'	in dark_coupl else 0.0
		self.m_alp		= dark_coupl['m_alp'] 	if 'm_alp' 	in dark_coupl else 1e10
		self.c_N        = dark_coupl['c_N'] 	if 'c_N' 	in dark_coupl else 0.0
		self.c_e        = dark_coupl['c_e'] 	if 'c_e' 	in dark_coupl else 0.0
		self.g_gamma_gamma = alphaQED/np.pi * self.c_e * self.inv_f_alp * np.abs(1 - F_alp_loop(self.m_alp**2/4/m_e**2)) if (self.c_e*self.inv_f_alp >= 0 and self.m_alp < 1e10) else 0.0



	def set_high_level_variables(self):
		self.Ue1 = np.sqrt(1.0-self.Ue4**2-self.Ue5**2)
		self.Umu1 = np.sqrt(1.0-self.Umu4**2-self.Umu5**2)
		self.Utau1 = np.sqrt(1.0-self.Utau4**2-self.Utau5**2)
		self.UD1 = np.sqrt(self.Ue4**2+self.Umu4**2+self.Utau4**2)

		self.Uactive4SQR = self.Ue4**2+self.Umu4**2+self.Utau4**2
		self.Uactive5SQR = self.Ue5**2+self.Umu5**2+self.Utau5**2

		self.alphaD = self.gprime**2/4.0/np.pi

	  	########################################################
		# all the following is true to leading order in chi
		
		# Neutrino couplings #### CHECK THE SIGN IN THE SECOND TERM
		self.ce5 = gweak/2/cw* (self.Ue5) + self.UD5*(-self.UD4*self.Ue4 -self.UD5*self.Ue5)*self.gprime*sw*self.chi
		self.cmu5 = gweak/2/cw* (self.Umu5) + self.UD5*(-self.UD4*self.Umu4 -self.UD5*self.Umu5)*self.gprime*sw*self.chi
		self.ctau5 = gweak/2/cw* (self.Utau5) + self.UD5*(-self.UD4*self.Utau4 -self.UD5*self.Utau5)*self.gprime*sw*self.chi
		
		self.de5 = self.UD5*(-self.UD4*self.Ue5 - self.UD5*self.Ue5)*self.gprime
		self.dmu5 = self.UD5*(-self.UD4*self.Umu5 - self.UD5*self.Umu5)*self.gprime
		self.dtau5 = self.UD5*(-self.UD4*self.Utau5 - self.UD5*self.Utau5)*self.gprime

		self.ce4 = gweak/2/cw* (self.Ue4) + self.UD4*(-self.UD4*self.Ue4 -self.UD5*self.Ue5)*self.gprime*sw*self.chi
		self.cmu4 = gweak/2/cw* (self.Umu4) + self.UD4*(-self.UD4*self.Umu4 -self.UD5*self.Umu5)*self.gprime*sw*self.chi
		self.ctau4 = gweak/2/cw* (self.Utau4) + self.UD4*(-self.UD4*self.Utau4 -self.UD5*self.Utau5)*self.gprime*sw*self.chi
		
		self.de4 = self.UD4*(-self.UD4*self.Ue4 - self.UD5*self.Ue4)*self.gprime
		self.dmu4 = self.UD4*(-self.UD4*self.Umu4 - self.UD5*self.Umu4)*self.gprime
		self.dtau4 = self.UD4*(-self.UD4*self.Utau4 - self.UD5*self.Utau4)*self.gprime

		self.clight4 = np.sqrt(self.ce4**2+self.cmu4**2+self.ctau4**2)
		self.dlight4 = np.sqrt(self.de4**2+self.dmu4**2+self.dtau4**2)

		self.clight5 = np.sqrt(self.ce5**2+self.cmu5**2+self.ctau5**2)
		self.dlight5 = np.sqrt(self.de5**2+self.dmu5**2+self.dtau5**2)

		self.c45 = gweak/2/cw* (np.sqrt(self.Uactive4SQR*self.Uactive5SQR)) + self.UD5*self.UD4*self.gprime*sw*self.chi
		self.c44 = gweak/2/cw* (np.sqrt(self.Uactive4SQR*self.Uactive4SQR)) + self.UD4*self.UD4*self.gprime*sw*self.chi
		self.c55 = gweak/2/cw* (np.sqrt(self.Uactive5SQR*self.Uactive5SQR)) + self.UD5*self.UD5*self.gprime*sw*self.chi
		self.d45 = self.UD5*self.UD4*self.gprime
		self.d44 = self.UD4*self.UD4*self.gprime
		self.d55 = self.UD5*self.UD5*self.gprime

		# Kinetic mixing
		##############
		tanchi = np.tan(self.chi)
		sinof2chi  = 2*tanchi/(1.0+tanchi*tanchi)
		cosof2chi  = (1.0 - tanchi*tanchi)/(1.0+tanchi*tanchi)
		s2chi = (1.0 - cosof2chi)/2.0
		self.tanof2beta = np.sqrt(s2w) *  sinof2chi / ( (self.mzprime/vev_EW)**2 - cosof2chi - (1.0-s2w)*s2chi )

		######################
		if self.tanof2beta != 0:
			self.tbeta = (-1 + np.sign(self.tanof2beta) * np.sqrt( 1 + self.tanof2beta*self.tanof2beta) )
		else:
			self.tbeta = 0.0

		self.sinof2beta = 2 * self.tbeta/(1.0+self.tbeta*self.tbeta)
		self.cosof2beta = (1.0-self.tbeta*self.tbeta)/(1.0+self.tbeta*self.tbeta)
		######################

		self.sbeta = np.sqrt( (1 - self.cosof2beta)/2.0)
		self.cbeta = np.sqrt( (1 + self.cosof2beta)/2.0)

		# Charged leptons
		self.ceV = self.cbeta*(2*s2w - 0.5) - 3.0/2.0*self.sbeta*sw*tanchi
		self.ceA = -(self.cbeta - self.sbeta*sw*tanchi)/2.0
		self.ceV = gweak/(2*cw) * (2*s2w - 0.5)
		self.ceA = gweak/(2*cw) * (-1.0/2.0)

		# quarks
		self.cuV = self.cbeta*(0.5 - 4*s2w/3.0) + 5.0/6.0*self.sbeta*sw*tanchi
		self.cuA = (self.cbeta + self.sbeta*sw*tanchi)/2.0

		self.cdV = self.cbeta*(-0.5 + 2*s2w/3.0) - 1.0/6.0*self.sbeta*sw*tanchi
		self.cdA = -(self.cbeta + self.sbeta*sw*tanchi)/2.0

		# if not self.minimal:
		self.deV = 3.0/2.0 * self.cbeta * s2w * tanchi + self.sbeta*(0.5 + 2*s2w)
		self.deA = (-self.sbeta - self.cbeta * s2w * tanchi)/2.0
		# self.deV = gweak/(2*cw) * 2*sw*cw**2*self.chi
		# self.deA = gweak/(2*cw) * 0

		self.duV = self.sbeta*(0.5 - 4*s2w/3.0) - 5.0/6.0*self.cbeta*sw*tanchi
		self.duA = (self.sbeta + self.cbeta*sw*tanchi)/2.0

		self.ddV = self.sbeta*(-0.5 + 2*s2w/3.0) + 1.0/6.0*self.cbeta*sw*tanchi
		self.ddA = -(self.sbeta + self.cbeta*sw*tanchi)/2.0

		self.gVproton = -3.0/2.0*sw*self.chi*(1-8.0/9.0*s2w)#2*self.duV +self.ddV
		self.gAproton = 2*self.duA + self.ddA
		self.gVneutron = 2*self.ddV + self.duV
		self.gAneutron = 2*self.ddA + self.duA

	def compute_rates(self):
		
		################################################################
		# ALP decays
		self.rates_alp = {}
		self.rates_alp['e_e'] = 0
		self.rates_alp['mu_mu'] = 0
		self.rates_alp['gamma_gamma'] = 0

		self.rates_alp['gamma_gamma'] = a_to_gamma_gamma(self)
		if self.m_alp > (lp.e_minus.mass+lp.e_plus.mass)/1e3:
			self.rates_alp['e_e'] = a_to_ell_ell(self, lp.e_minus)
		if self.m_alp > (lp.mu_minus.mass+lp.mu_plus.mass)/1e3:
			self.rates_alp['mu_mu'] = a_to_ell_ell(self, lp.mu_minus)
		
		# total decay rate
		self.rate_alp_total = sum(self.rates_alp.values())

		# total decay rate
		self.lifetime = get_decay_rate_in_s(self.rate_alp_total)
		self.ctau0 = get_decay_rate_in_cm(self.rate_alp_total)

		# Branchin ratios
		self.alp_brs = {}
		if self.rate_alp_total == 0.0:
			for channel in self.rates_alp.keys():
				self.alp_brs[channel] = self.rates_alp[channel]*0.0
		else:
			for channel in self.rates_alp.keys():
				self.alp_brs[channel] = self.rates_alp[channel]/self.rate_alp_total


		##################
		# Neutrino 4
		mh = self.m4
		self.rates = {}
		neutrinos = [lp.nu_e, lp.nu_mu, lp.nu_tau]

		# channels with 3 neutrinos in final state
		self.rates['nu_nu_nu'] = 0.0
		for nu_a in neutrinos:
			self.rates['nu_nu_nu'] += nui_nuj_nuk_nuk(self, N4, nu_a)

		# channels with 1 neutrino in final states
		self.rates['nu_gamma'] = 0
		self.rates['nu_e_e'] = 0
		self.rates['nu_mu_mu'] = 0
		self.rates['nu_tau_tau'] = 0
		self.rates['nu_e_mu'] = 0
		self.rates['nu_e_tau'] = 0
		self.rates['nu_mu_tau'] = 0
		self.rates['nu_pi'] = 0 
		self.rates['nu_eta'] = 0
		self.rates['nu_D'] = 0
		self.rates['nu_rho'] = 0
		self.rates['nu_omega'] = 0
		self.rates['nu_phi'] = 0
		self.rates['nu_Kstar'] = 0
		self.rates['e_pi'] = 0
		self.rates['e_K'] = 0
		self.rates['e_Kstar'] = 0
		self.rates['e_D'] = 0
		self.rates['e_Ds'] = 0
		self.rates['mu_pi'] = 0
		self.rates['mu_K'] = 0
		self.rates['mu_Kstar'] = 0
		self.rates['mu_D'] = 0
		self.rates['mu_Ds'] = 0
		self.rates['tau_pi'] = 0
		self.rates['tau_K'] = 0
		self.rates['tau_Kstar'] = 0
		self.rates['tau_D'] = 0
		self.rates['tau_Ds'] = 0

		for nu_a in neutrinos:			# nu gamma 
			self.rates['nu_gamma'] += nui_nuj_gamma(self, N4, nu_a)
			# dileptons -- already contains the Delta L = 2 channel
			# for different flavors, we sum the LNV and LNC channels.
			if mh > 2*lp.e_minus.mass/1e3:
				self.rates['nu_e_e'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.e_minus, lp.e_plus)
			if mh > lp.e_minus.mass/1e3 + lp.mu_minus.mass/1e3:
				self.rates['nu_e_mu'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.e_minus, lp.mu_plus)
				self.rates['nu_e_mu'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.mu_minus, lp.e_plus)
			if mh > 2*lp.mu_minus.mass/1e3:
				self.rates['nu_mu_mu'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.mu_minus, lp.mu_plus)
			
			if mh > lp.e_minus.mass/1e3 + lp.tau_minus.mass/1e3:
				self.rates['nu_e_tau'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.e_minus, lp.tau_plus)
				self.rates['nu_e_tau'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.tau_minus, lp.e_plus)
			if mh > lp.mu_minus.mass/1e3 + lp.tau_minus.mass/1e3:
				self.rates['nu_mu_tau'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.mu_minus, lp.tau_plus)
				self.rates['nu_mu_tau'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.tau_minus, lp.mu_plus)
			if mh > 2*lp.tau_minus.mass/1e3:
				self.rates['nu_tau_tau'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.tau_minus, lp.tau_plus)
			
			# pseudoscalar -- neutral current
			if mh > lp.pi_0.mass/1e3:
				self.rates['nu_pi'] += nui_nu_P(self, N4, nu_a, lp.pi_0)
			if mh > lp.eta.mass/1e3:
				self.rates['nu_eta'] += nui_nu_P(self, N4, nu_a, lp.eta)
			if mh > lp.D_0.mass/1e3:
				self.rates['nu_D'] += nui_nu_P(self, N4, nu_a, lp.D_0)

			if mh > lp.rho_770_0.mass/1e3:
				self.rates['nu_rho'] += nui_nu_V(self, N4, nu_a, lp.rho_770_0)
			if mh > lp.omega_782.mass/1e3:
				self.rates['nu_omega'] += nui_nu_V(self, N4, nu_a, lp.omega_782)
			if mh > lp.Kst_892_0.mass/1e3:
				self.rates['nu_Kstar'] += nui_nu_V(self, N4, nu_a, lp.Kst_892_0)
			if mh > lp.phi_1020.mass/1e3:
				self.rates['nu_phi'] += nui_nu_V(self, N4, nu_a, lp.phi_1020)

		# CC-only channels	
		# pseudoscalar -- already contain the Delta L = 2 channel
		if mh > lp.e_minus.mass/1e3+lp.pi_plus.mass/1e3:
			self.rates['e_pi'] += nui_l_P(self, N4, lp.e_minus, lp.pi_plus)
		if mh > lp.e_minus.mass/1e3+lp.K_plus.mass/1e3:
			self.rates['e_K'] += nui_l_P(self, N4, lp.e_minus, lp.K_plus)
		if mh > lp.e_minus.mass/1e3+lp.Kst_892_plus.mass/1e3:
			self.rates['e_Kstar'] += nui_l_V(self, N4, lp.e_minus, lp.Kst_892_plus)
		if mh > lp.e_minus.mass/1e3+lp.D_plus.mass/1e3:
			self.rates['e_D'] += nui_l_P(self, N4, lp.e_minus, lp.D_plus)
		if mh > lp.e_minus.mass/1e3+lp.D_s_plus.mass/1e3:
			self.rates['e_Ds'] += nui_l_P(self, N4, lp.e_minus, lp.D_s_plus)

		if mh > lp.mu_minus.mass/1e3+lp.pi_plus.mass/1e3:
			self.rates['mu_pi'] += nui_l_P(self, N4, lp.mu_minus, lp.pi_plus)
		if mh > lp.mu_minus.mass/1e3+lp.K_plus.mass/1e3:
			self.rates['mu_K'] += nui_l_P(self, N4, lp.mu_minus, lp.K_plus)
		if mh > lp.mu_minus.mass/1e3+lp.Kst_892_plus.mass/1e3:
			self.rates['mu_Kstar'] += nui_l_V(self, N4, lp.mu_minus, lp.Kst_892_plus)
		if mh > lp.mu_minus.mass/1e3+lp.D_plus.mass/1e3:
			self.rates['mu_D'] += nui_l_P(self, N4, lp.mu_minus, lp.D_plus)
		if mh > lp.mu_minus.mass/1e3+lp.D_s_plus.mass/1e3:
			self.rates['mu_Ds'] += nui_l_P(self, N4, lp.mu_minus, lp.D_s_plus)

		if mh > lp.tau_minus.mass/1e3+lp.pi_plus.mass/1e3:
			self.rates['tau_pi'] += nui_l_P(self, N4, lp.tau_minus, lp.pi_plus)
		if mh > lp.tau_minus.mass/1e3+lp.K_plus.mass/1e3:
			self.rates['tau_K'] += nui_l_P(self, N4, lp.tau_minus, lp.K_plus)
		if mh > lp.tau_minus.mass/1e3+lp.Kst_892_plus.mass/1e3:
			self.rates['tau_Kstar'] += nui_l_V(self, N4, lp.tau_minus, lp.Kst_892_plus)
		if mh > lp.tau_minus.mass/1e3+lp.D_plus.mass/1e3:
			self.rates['tau_D'] += nui_l_P(self, N4, lp.tau_minus, lp.D_plus)
		if mh > lp.tau_minus.mass/1e3+lp.D_s_plus.mass/1e3:
			self.rates['tau_Ds'] += nui_l_P(self, N4, lp.tau_minus, lp.D_s_plus)

		# vector mesons
		if mh > lp.e_minus.mass/1e3+lp.Kst_892_plus.mass/1e3:
			self.rates['e_Kstar'] += nui_l_V(self, N4, lp.e_minus, lp.Kst_892_plus)
				# vector mesons
		if mh > lp.mu_minus.mass/1e3+lp.Kst_892_plus.mass/1e3:
			self.rates['mu_Kstar'] += nui_l_V(self, N4, lp.mu_minus, lp.Kst_892_plus)
				# vector mesons
		if mh > lp.tau_minus.mass/1e3+lp.Kst_892_plus.mass/1e3:
			self.rates['tau_Kstar'] += nui_l_V(self, N4, lp.tau_minus, lp.Kst_892_plus)
		


		# total decay rate
		self.rate_total = sum(self.rates.values())

		# total decay rate
		self.lifetime = get_decay_rate_in_s(self.rate_total)
		self.ctau0 = get_decay_rate_in_cm(self.rate_total)

		# Branchin ratios
		brs = {}
		if self.rate_total == 0.0:
			for channel in self.rates.keys():
				brs[channel] = self.rates[channel]*0.0
		else:
			for channel in self.rates.keys():
				brs[channel] = self.rates[channel]/self.rate_total
			self.brs = brs