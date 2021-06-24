import numpy as np
from . import const 

class hnl_model():

	def __init__(self, m4, mixings = [0.0,0.0,0.0], minimal=True, HNLtype="Majorana"):

		self.minimal = minimal
		
		if self.minimal:
			self.gprime		= 0.0
			self.chi		= 0.0
			self.Ue4		= mixings[0]
			self.Umu4		= mixings[1]
			self.Utau4		= mixings[2]
			self.Ue5		= 0.0
			self.Umu5		= 0.0
			self.Utau5		= 0.0
			self.UD4		= 0.0
			self.UD5		= 0.0
			self.m4			= m4
			self.m5			= 1e10
			self.Mzprime	= 1e10
			self.Dirac		= HNLtype

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
		self.ce5 = const.gweak/2/const.cw* (self.Ue5) + self.UD5*(-self.UD4*self.Ue4 -self.UD5*self.Ue5)*self.gprime*const.sw*self.chi
		self.cmu5 = const.gweak/2/const.cw* (self.Umu5) + self.UD5*(-self.UD4*self.Umu4 -self.UD5*self.Umu5)*self.gprime*const.sw*self.chi
		self.ctau5 = const.gweak/2/const.cw* (self.Utau5) + self.UD5*(-self.UD4*self.Utau4 -self.UD5*self.Utau5)*self.gprime*const.sw*self.chi
		
	
		self.de5 = self.UD5*(-self.UD4*self.Ue5 - self.UD5*self.Ue5)*self.gprime
		self.dmu5 = self.UD5*(-self.UD4*self.Umu5 - self.UD5*self.Umu5)*self.gprime
		self.dtau5 = self.UD5*(-self.UD4*self.Utau5 - self.UD5*self.Utau5)*self.gprime


		self.ce4 = const.gweak/2/const.cw* (self.Ue4) + self.UD4*(-self.UD4*self.Ue4 -self.UD5*self.Ue5)*self.gprime*const.sw*self.chi
		self.cmu4 = const.gweak/2/const.cw* (self.Umu4) + self.UD4*(-self.UD4*self.Umu4 -self.UD5*self.Umu5)*self.gprime*const.sw*self.chi
		self.ctau4 = const.gweak/2/const.cw* (self.Utau4) + self.UD4*(-self.UD4*self.Utau4 -self.UD5*self.Utau5)*self.gprime*const.sw*self.chi
		
		self.de4 = self.UD4*(-self.UD4*self.Ue4 - self.UD5*self.Ue4)*self.gprime
		self.dmu4 = self.UD4*(-self.UD4*self.Umu4 - self.UD5*self.Umu4)*self.gprime
		self.dtau4 = self.UD4*(-self.UD4*self.Utau4 - self.UD5*self.Utau4)*self.gprime

		self.clight4 = np.sqrt(self.ce4**2+self.cmu4**2+self.ctau4**2)
		self.dlight4 = np.sqrt(self.de4**2+self.dmu4**2+self.dtau4**2)

		self.clight5 = np.sqrt(self.ce5**2+self.cmu5**2+self.ctau5**2)
		self.dlight5 = np.sqrt(self.de5**2+self.dmu5**2+self.dtau5**2)

		self.c45 = const.gweak/2/const.cw* (np.sqrt(self.Uactive4SQR*self.Uactive5SQR)) + self.UD5*self.UD4*self.gprime*const.sw*self.chi
		self.c44 = const.gweak/2/const.cw* (np.sqrt(self.Uactive4SQR*self.Uactive4SQR)) + self.UD4*self.UD4*self.gprime*const.sw*self.chi
		self.c55 = const.gweak/2/const.cw* (np.sqrt(self.Uactive5SQR*self.Uactive5SQR)) + self.UD5*self.UD5*self.gprime*const.sw*self.chi
		self.d45 = self.UD5*self.UD4*self.gprime
		self.d44 = self.UD4*self.UD4*self.gprime
		self.d55 = self.UD5*self.UD5*self.gprime

		self.clight = const.gweak/2/const.cw
		self.dlight = 0.0


		# BSM-like couplings
		# self.de4 = self.UD4*self.Ue1*self.UD1 * (self.gprime/const.gweak)
		# self.dmu4 = self.UD4*self.Umu1*self.UD1 * (self.gprime/const.gweak)
		# self.dtau4 = self.UD4*self.Utau1*self.UD1 * (self.gprime/const.gweak)

		# self.de5 = self.UD5*self.Ue1*self.UD1* self.gprime/const.gweak
		# self.dmu5 = self.UD5*self.Umu1*self.UD1* self.gprime/const.gweak
		# self.dtau5 = self.UD5*self.Utau1*self.UD1* self.gprime/const.gweak

		# SM-like couplings
		# self.ce4 = self.UD4*self.Ue1*self.UD1 * (0.5 - self.gprime/const.gweak*const.sw*const.cw*self.chi)
		# self.cmu4 = self.UD4*self.Umu1*self.UD1 * (0.5 - self.gprime/const.gweak*const.sw*const.cw*self.chi)
		# self.ctau4 = self.UD4*self.Utau1*self.UD1 * (0.5 - self.gprime/const.gweak*const.sw*const.cw*self.chi)

		# self.ce5 = self.UD5*self.Ue1*self.UD1 * (0.5 - self.gprime/const.gweak*const.sw*const.cw*self.chi)
		# self.cmu5 = self.UD5*self.Umu1*self.UD1 * (0.5 - self.gprime/const.gweak*const.sw*const.cw*self.chi)
		# self.ctau5 = self.UD5*self.Utau1*self.UD1 * (0.5 - self.gprime/const.gweak*const.sw*const.cw*self.chi)


		# Kinetic mixing
		##############
		tanchi = np.tan(self.chi)

		sinof2chi  = 2*tanchi/(1.0+tanchi*tanchi)
		cosof2chi  = (1.0 - tanchi*tanchi)/(1.0+tanchi*tanchi)

		s2chi = (1.0 - cosof2chi)/2.0

		tanof2beta = np.sqrt(const.s2w) *  sinof2chi / ( (self.Mzprime/const.vev_EW)**2 - cosof2chi - (1.0-const.s2w)*s2chi )

		self.epsilon = const.cw*self.chi

		######################
		# CHECK ME! I dont think this reproduces the SM limit when tanof2beta = 0
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
		# self.ceV = cbeta*(2*const.s2w - 0.5) - 3.0/2.0*sbeta*const.sw*tanchi
		# self.ceA = -(cbeta - sbeta*const.sw*tanchi)/2.0
		self.ceV = const.gweak/(2*const.cw) * (2*const.s2w - 0.5)
		self.ceA = const.gweak/(2*const.cw) * (-1.0/2.0)

		# quarks
		self.cuV = cbeta*(0.5 - 4*const.s2w/3.0) + 5.0/6.0*sbeta*const.sw*tanchi
		self.cuA = (cbeta + sbeta*const.sw*tanchi)/2.0

		self.cdV = cbeta*(-0.5 + 2*const.s2w/3.0) - 1.0/6.0*sbeta*const.sw*tanchi
		self.cdA = -(cbeta + sbeta*const.sw*tanchi)/2.0

		# if not self.minimal:
		# self.deV = 3.0/2.0 * cbeta * const.s2w * tanchi + sbeta*(0.5 + 2*const.s2w)
		# self.deA = (-sbeta - cbeta * const.s2w * tanchi)/2.0
		self.deV = const.gweak/(2*const.cw) * 2*const.sw*const.cw**2*self.chi
		self.deA = const.gweak/(2*const.cw) * 0

		self.duV = sbeta*(0.5 - 4*const.s2w/3.0) - 5.0/6.0*cbeta*const.sw*tanchi
		self.duA = (sbeta + cbeta*const.sw*tanchi)/2.0

		self.ddV = sbeta*(-0.5 + 2*const.s2w/3.0) + 1.0/6.0*cbeta*const.sw*tanchi
		self.ddA = -(sbeta + cbeta*const.sw*tanchi)/2.0

		self.gVproton = -3.0/2.0*const.sw*self.chi*(1-8.0/9.0*const.s2w)#2*self.duV +self.ddV
		self.gAproton = 2*self.duA + self.ddA
		self.gVneutron = 2*self.ddV + self.duV
		self.gAneutron = 2*self.ddA + self.duA
