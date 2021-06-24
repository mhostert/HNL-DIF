import numpy as np
from scipy import interpolate
from particle import Particle
from particle import literals as lp

from . import const

# experiment 
ND280_FHC    = "nd280_FHC"
ND280_RHC = "nd280_RHC"

######################################################
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    
#   nus/cm^2/GeV/POT      
######################################################
class experiment():

	def __init__(self,EXP_FLAG):

		self.EXP_FLAG = EXP_FLAG

		#########################
		# Experimental Parameters

		if self.EXP_FLAG == ND280_FHC:

			self.prop = {
				
				"name" : "nd280/FHC",
				"flux_norm" : 1.0/1e21/0.05,
				"emin" : 0.05,
				"emax" : 10.0,
				"pots" : 12.34e20,
				"flavors" : [lp.nu_mu],
			}

		elif self.EXP_FLAG == ND280_RHC:
			self.prop = {
				
				"name" : "nd280/RHC",
				"flux_norm" : 1.0/1e21/0.05,
				"emin" : 0.05,
				"emax" : 10.0,
				"pots" : 6.29e20,
				"flavors" : [lp.nu_mu_bar],
			}
		
		else:
			print('ERROR! No experiment chosen.')


	def get_flux_func(self, parent=lp.pi_plus, nuflavor=lp.nu_mu):
		# nus/cm^2/50 MeV/1e21 POT

		fluxfile = f"./fluxes/{self.prop['name']}/{nuflavor.name}_{parent.name}.dat"
		try:
			e,f = np.genfromtxt(fluxfile, unpack = True)
			flux = interpolate.interp1d(e, f*self.prop['flux_norm'], fill_value=0.0, bounds_error=False)
			return flux
		except:
		    print(f"Fluxfile not found: {fluxfile}")


