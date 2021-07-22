import numpy as np
import vegas as vg
import gvar as gv

import random

from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

from . import const 
from . import model 
from . import nuH_integrands as integrands

# Integration parameters
NINT = 20
NEVAL = 1e4

NINT_warmup = 20
NEVAL_warmup = 1e3

def Power(x,n):
	return x**n
def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c


class MC_events:
	def __init__(self, HNLtype= "MAJORANA", EN=1.0, ENmin = 0.0, ENmax=10.0, mh=0.150, mf=0.0, mp=const.m_e, mm=const.m_e, helicity=-1, BSMparams=None, convolve_flux=True):
		
		self.params = BSMparams
		
		# set target properties
		if ENmin <= mh:
			self.ENmin = mh
		else:
			self.ENmin = ENmin
		self.ENmax = ENmax
		self.EN = EN
		self.mh = mh
		self.mf = mf
		self.mp = mp
		self.mm = mm		
		self.helicity = helicity
		self.HNLtype = HNLtype
		self.convolve_flux = convolve_flux

	def get_MC_events(self):

		# Model params
		params = self.params

		if self.convolve_flux:
			DIM =5
		else:

			DIM =4
		
		#########################################################################
		# BATCH SAMPLE INTEGRAN OF INTEREST		
		batch_f = integrands.N_to_nu_ell_ell(dim=DIM, MC_case=self)
		integ = vg.Integrator(DIM*[[0.0, 1.0]])

		##########################################################################
		# COMPUTE TOTAL INTEGRAL
		# Sample the integrand to adapt integrator
		integ(batch_f, nitn=NINT_warmup, neval=NEVAL_warmup)

		# Sample again, now saving result
		result = integ(batch_f,  nitn = NINT, neval = NEVAL, minimize_mem = False)

		# final integral of ( dGamma/du/dt/dc3/dphi34 * flux(EN) )
		####################################################################
		integral = result.mean
		##########################################################################


		##############################################
		## Get samples from the MC integral

		SAMPLES = [[] for i in range(DIM)]
		weights = []
		integral = 0.0
		variance = 0.0
		for x, wgt, hcube in integ.random_batch(yield_hcube=True):
			
			wgt_fx = wgt*batch_f(x)
			weights = np.concatenate((weights,wgt_fx))
			for i in range(DIM):
				SAMPLES[i] = np.concatenate((SAMPLES[i],x[:,i]))

			for i in range(hcube[0], hcube[-1] + 1):
				idx = (hcube == i)
				nwf = np.sum(idx)
				wf  = wgt_fx[idx]

				sum_wf = np.sum(wf)
				sum_wf2 = np.sum(wf ** 2) # sum of (wgt * f(x)) ** 2

				integral += sum_wf
				variance += (sum_wf2 * nwf - sum_wf ** 2) / (nwf - 1.)

		final_integral = gv.gvar(integral, variance ** 0.5)
		
		# print final_integral
		# print "integral = %s, Q = %.2f"%(result, result.Q)
		P1LAB_decay, P2LAB_decay, P3LAB_decay, P4LAB_decay = integrands.N_to_nu_ell_ell_phase_space(samples=SAMPLES, MC_case=self)

		return P1LAB_decay, P2LAB_decay, P3LAB_decay, P4LAB_decay, weights



