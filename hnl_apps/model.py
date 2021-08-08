import numpy as np

from .const import * 
from .rates import *


class hnl_model():

	def __init__(self, m4, mixings = [0.0,0.0,0.0], dipoles = [0.0,0.0,0.0], dark_coupl=[0,0,0,0], GX = 0.0, minimal=True, HNLtype="dirac"):

		self.minimal = minimal
		
		if self.minimal:
			self.gprime		= dark_coupl[0]
			self.chi		= dark_coupl[1]
			self.Ue4		= np.sqrt(mixings[0])
			self.Umu4		= np.sqrt(mixings[1])
			self.Utau4		= np.sqrt(mixings[2])
			self.Ue5		= 0.0
			self.Umu5		= 0.0
			self.Utau5		= 0.0
			self.UD4		= dark_coupl[2]
			self.UD5		= 0.0
			self.m4			= m4
			self.m5			= 1e10
			self.Mzprime	= dark_coupl[3]
			self.HNLtype	= HNLtype

			self.GX         = GX

			self.dip_e4		= dipoles[0]
			self.dip_mu4		= dipoles[1]
			self.dip_tau4		= dipoles[2]
			self.dij        = np.sqrt(self.dip_e4**2+self.dip_mu4**2+self.dip_tau4**2)
			self.cut_ee     = 2*m_e

		else:
			print("Non-minimal model not supported")



	def set_high_level_variables(self):
		self.Ue1 = np.sqrt(1.0-self.Ue4*self.Ue4-self.Ue5*self.Ue5)
		self.Umu1 = np.sqrt(1.0-self.Umu4*self.Umu4-self.Umu5*self.Umu5)
		self.Utau1 = np.sqrt(1.0-self.Utau4*self.Utau4-self.Utau5*self.Utau5)
		self.UD1 = np.sqrt(self.Ue4*self.Ue4+self.Umu4*self.Umu4+self.Utau4*self.Utau4)

		self.Uactive4SQR = self.Ue4*self.Ue4+self.Umu4*self.Umu4+self.Utau4*self.Utau4
		self.Uactive5SQR = self.Ue5*self.Ue5+self.Umu5*self.Umu5+self.Utau5*self.Utau5

		self.alphaD = self.gprime*self.gprime/4.0/np.pi

	  ########################################################
		# all the following is true to leading order in chi


		# Neutrino couplings ## CHECK THE SIGN IN THE SECOND TERM????
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

		tanof2beta = np.sqrt(s2w) *  sinof2chi / ( (self.Mzprime/vev_EW)**2 - cosof2chi - (1.0-s2w)*s2chi )

		self.epsilon = cw*self.chi

		######################
		if tanof2beta != 0:
			tbeta = (-1 + np.sign(tanof2beta) * np.sqrt( 1 + tanof2beta*tanof2beta) )
		else:
			tbeta = 0.0

		sinof2beta = 2 * tbeta/(1.0+tbeta*tbeta)
		cosof2beta = (1.0-tbeta*tbeta)/(1.0+tbeta*tbeta)
		######################

		sbeta = np.sqrt( (1 - cosof2beta)/2.0)
		cbeta = np.sqrt( (1 + cosof2beta)/2.0)

		# Charged leptons
		self.ceV = cbeta*(2*s2w - 0.5) - 3.0/2.0*sbeta*sw*tanchi
		self.ceA = -(cbeta - sbeta*sw*tanchi)/2.0
		self.ceV = gweak/(2*cw) * (2*s2w - 0.5)
		self.ceA = gweak/(2*cw) * (-1.0/2.0)

		# quarks
		self.cuV = cbeta*(0.5 - 4*s2w/3.0) + 5.0/6.0*sbeta*sw*tanchi
		self.cuA = (cbeta + sbeta*sw*tanchi)/2.0

		self.cdV = cbeta*(-0.5 + 2*s2w/3.0) - 1.0/6.0*sbeta*sw*tanchi
		self.cdA = -(cbeta + sbeta*sw*tanchi)/2.0

		# if not self.minimal:
		self.deV = 3.0/2.0 * cbeta * s2w * tanchi + sbeta*(0.5 + 2*s2w)
		self.deA = (-sbeta - cbeta * s2w * tanchi)/2.0
		# self.deV = gweak/(2*cw) * 2*sw*cw**2*self.chi
		# self.deA = gweak/(2*cw) * 0

		self.duV = sbeta*(0.5 - 4*s2w/3.0) - 5.0/6.0*cbeta*sw*tanchi
		self.duA = (sbeta + cbeta*sw*tanchi)/2.0

		self.ddV = sbeta*(-0.5 + 2*s2w/3.0) + 1.0/6.0*cbeta*sw*tanchi
		self.ddA = -(sbeta + cbeta*sw*tanchi)/2.0

		self.gVproton = -3.0/2.0*sw*self.chi*(1-8.0/9.0*s2w)#2*self.duV +self.ddV
		self.gAproton = 2*self.duA + self.ddA
		self.gVneutron = 2*self.ddV + self.duV
		self.gAneutron = 2*self.ddA + self.duA

	def compute_rates(self):
		
		##################
		# Neutrino 4
		mh = self.m4
		rates = {}
		neutrinos = [lp.nu_e, lp.nu_mu, lp.nu_tau]

		# channels with 3 neutrinos in final state
		rates['nu_nu_nu'] = 0.0
		for nu_a in neutrinos:
			rates['nu_nu_nu'] += nui_nuj_nuk_nuk(self, N4, nu_a)

		# channels with 1 neutrino in final states
		rates['nu_gamma'] = 0
		rates['nu_e_e'] = 0
		rates['nu_mu_mu'] = 0
		rates['nu_e_mu'] = 0
		rates['nu_pi'] = 0 
		rates['nu_eta'] = 0
		rates['e_pi'] = 0
		rates['e_K'] = 0
		rates['mu_pi'] = 0
		rates['mu_K'] = 0

		for nu_a in neutrinos:			# nu gamma 
			rates['nu_gamma'] += nui_nuj_gamma(self, N4, nu_a)
			# dileptons -- already contains the Delta L = 2 channel
			if mh > 2*lp.e_minus.mass/1e3:
				rates['nu_e_e'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.e_minus, lp.e_plus)
			if mh > lp.e_minus.mass/1e3 + lp.mu_minus.mass/1e3:
				rates['nu_e_mu'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.e_minus, lp.mu_plus)
				rates['nu_e_mu'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.mu_minus, lp.e_plus)
			if mh > 2*lp.mu_minus.mass/1e3:
				rates['nu_mu_mu'] += nui_nuj_ell1_ell2(self, N4, nu_a, lp.mu_minus, lp.mu_plus)
			# pseudoscalar -- neutral current
			if mh > lp.pi_0.mass/1e3:
				rates['nu_pi'] += nui_nu_P(self, N4, nu_a, lp.pi_0)
			if mh > lp.eta.mass/1e3:
				rates['nu_eta'] += nui_nu_P(self, N4, nu_a, lp.eta)
			

		# CC-only channels	
		# pseudoscalar -- factor of 2 for delta L=2 channel 
		if mh > lp.e_minus.mass/1e3+lp.pi_plus.mass/1e3:
			rates['e_pi'] = nui_l_P(self, N4, lp.e_minus, lp.pi_plus)
		if mh > lp.e_minus.mass/1e3+lp.K_plus.mass/1e3:
			rates['e_K'] = nui_l_P(self, N4, lp.e_minus, lp.K_plus)
		
		# pseudoscalar -- already contain the Delta L = 2 channel
		if mh > lp.mu_minus.mass/1e3+lp.pi_plus.mass/1e3:
			rates['mu_pi'] = nui_l_P(self, N4, lp.mu_minus, lp.pi_plus)
		if mh > lp.mu_minus.mass/1e3+lp.K_plus.mass/1e3:
			rates['mu_K'] = nui_l_P(self, N4, lp.mu_minus, lp.K_plus)
	
		self.rates = rates			

		# total decay rate
		self.rate_total = sum(self.rates.values())

		# total decay rate
		self.lifetime = get_decay_rate_in_s(self.rate_total)
		self.ctau0 = get_decay_rate_in_cm(self.rate_total)

		# Branchin ratios
		brs = {}
		for channel in self.rates.keys():
			brs[channel] = self.rates[channel]/self.rate_total
		self.brs = brs


