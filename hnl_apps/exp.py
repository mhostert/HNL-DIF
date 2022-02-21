import numpy as np
from scipy import interpolate
from particle import literals as lp

from pathlib import Path
local_dir = Path(__file__).parent

from . import const


# ratio between current and full t2k
POT_RATIO_FULL_T2K = 10.7


# experiments
flag_nd280    = "nd280"
flag_nd280_flat = "nd280_flat"
flag_nd280_fhc = "nd280_FHC"
flag_nd280_rhc = "nd280_RHC"
flag_ps191 = "PS191"
flag_ps191_proposal = "PS191_proposal"

flag_sbnd_numi_absorber = "SBND_NuMI_absorber"
flag_muboone_numi_absorber = "microBooNE_NuMI_absorber"
flag_icarus_numi_absorber = "ICARUS_NuMI_absorber"


class experiment():

	def __init__(self,EXP_FLAG):

		self.EXP_FLAG = EXP_FLAG

		#########################
		# Experimental Parameters

		if self.EXP_FLAG == flag_nd280:

			##############################################################################
			# Including extrapolation and interpolation of t2k efficiencies by hand
			masses_t2k = np.linspace(140, 490, 36)
			main_folder = f'{local_dir}/../nd280-heavy-neutrino-search-2018_main/'
			eff_filename = 'efficiency.npy'
			eff = np.load(main_folder+eff_filename)

			nuPOT = 12.34e20
			nubarPOT = 6.29e20
			avg_eff_nu_nubar = (eff[:, 10, :5]*nuPOT + eff[:, 10, 5:]*nubarPOT)/(nuPOT+nubarPOT)
			summed_avg_eff = np.sum(avg_eff_nu_nubar, axis=1)

			stop_index_fit=17
			func_extrap = np.polynomial.polynomial.Polynomial.fit(masses_t2k[:stop_index_fit], 
			                                           summed_avg_eff[:stop_index_fit], 
			                                           deg=1)
			func_interp = interpolate.interp1d(masses_t2k, summed_avg_eff, bounds_error=False, fill_value=0)
			
			self.eff = summed_avg_eff
			self.masses_t2k = masses_t2k

			self.prop = {
				"name" : "nd280/FHC",
				# nus/cm^2/50 MeV/1e21 POT
				"flux_norm" : 1.0/1e21/0.05,
				"area" : 1.7e2*1.96e2, # cm^2
				"length" : 56*3, # cm
				"baseline" : 280e2, # cm
				"eff_nu_e_e" : lambda x: np.heaviside(x*1e3-140,0)*func_interp(x*1e3) + np.heaviside(-x*1e3+140,0)*func_extrap(x*1e3),
				"emin" : 0.05,
				"emax" : 10.0,
				"pots" : 12.34e20+6.29e20, ## approximately the same flux.
				"flavors" : [lp.nu_mu,lp.nu_mu_bar],
			}

		elif self.EXP_FLAG == flag_nd280_fhc:

			##############################################################################
			# Including extrapolation and interpolation of t2k efficiencies by hand
			masses_t2k = np.linspace(140, 490, 36)
			main_folder = f'{local_dir}/../nd280-heavy-neutrino-search-2018_main/'
			eff_filename = 'efficiency.npy'
			eff = np.load(main_folder+eff_filename)

			nuPOT = 12.34e20
			nubarPOT = 6.29e20
			eff_nu = eff[:, 10, :5]
			summed_eff = np.sum(eff_nu, axis=1)

			stop_index_fit=17
			func_extrap = np.polynomial.polynomial.Polynomial.fit(masses_t2k[:stop_index_fit], 
			                                           summed_eff[:stop_index_fit], 
			                                           deg=1)
			func_interp = interpolate.interp1d(masses_t2k, summed_eff, bounds_error=False, fill_value=0)


			self.prop = {
				"name" : "nd280/FHC",
				# nus/cm^2/50 MeV/1e21 POT
				"flux_norm" : 1.0/1e21/0.05,
				"area" : 1.7e2*1.96e2, # cm^2
				"length" : 56*3, # cm
				"baseline" : 280e2, # cm
				"eff_nu_e_e" : lambda x: np.heaviside(x*1e3-140,0)*func_interp(x*1e3) + np.heaviside(-x*1e3+140,0)*func_extrap(x*1e3),
				"emin" : 0.05,
				"emax" : 10.0,
				"pots" : 12.34e20,
				"flavors" : [lp.nu_mu,lp.nu_mu_bar],
			}
		
		elif self.EXP_FLAG == flag_nd280_rhc:

			##############################################################################
			# Including extrapolation and interpolation of t2k efficiencies by hand
			masses_t2k = np.linspace(140, 490, 36)
			main_folder = f'{local_dir}/../nd280-heavy-neutrino-search-2018_main/'
			eff_filename = 'efficiency.npy'
			eff = np.load(main_folder+eff_filename)

			eff_nubar = eff[:, 10, 5:]
			summed_eff = np.sum(eff_nubar, axis=1)

			stop_index_fit=17
			func_extrap = np.polynomial.polynomial.Polynomial.fit(masses_t2k[:stop_index_fit], 
			                                           summed_eff[:stop_index_fit], 
			                                           deg=1)
			func_interp = interpolate.interp1d(masses_t2k, summed_eff, bounds_error=False, fill_value=0)

			self.prop = {
				"name" : "nd280/RHC",
				# nus/cm^2/50 MeV/1e21 POT
				"flux_norm" : 1.0/1e21/0.05,
				"area" : 1.7e2*1.96e2, # cm^2
				"length" : 56*3, # cm
				"baseline" : 280e2, # cm
				"eff_nu_e_e" : lambda x: np.heaviside(x*1e3-140,0)*func_interp(x*1e3) + np.heaviside(-x*1e3+140,0)*func_extrap(x*1e3),
				"emin" : 0.05,
				"emax" : 10.0,
				"pots" : 6.29e20,
				"flavors" : [lp.nu_mu_bar,lp.nu_mu],
			}

		elif self.EXP_FLAG == flag_sbnd_numi_absorber:
			self.prop = {
				"name" : "nd280/RHC",
				"area" : (4e2*5e2), # cm^2
				"length" : np.sqrt(4**2+4**2)*1e2, # cm
				"baseline" : 426.82e2, # cm
			}

		elif self.EXP_FLAG == flag_muboone_numi_absorber:
			self.prop = {
				"name" : "nd280/RHC",
				"area" : 10.36e2*2.5635e2, # cm^2
				"length" : 3e2, # cm
				"baseline" : 101.51e2, # cm
			}
		
		elif self.EXP_FLAG == flag_icarus_numi_absorber:
			self.prop = {
				"name" : "nd280/RHC",
				"area" : 19.9e2*3.6e2, # cm^2
				"length" : np.sqrt(3.6**2+3.9**2)*1e2, # cm
				"baseline" : 109.26e2, # cm
			}
				
		elif self.EXP_FLAG == flag_ps191:
			self.prop = {
				"name" : "ps191",
				"flux_norm" : 1.0,
				"area" : 6e2*3e2, # cm^2
				"length" : 12e2, # cm
				"baseline" : 128e2, # cm
				"eff_nu_e_e" : lambda x: 0.3,
				"eff_nu_e_mu" : lambda x: 0.3,
				"eff_nu_mu_mu" : lambda x: 0.3,
				"eff_mu_pi" : lambda x: 0.3,
				"emin" : 0.05,
				"emax" : 7.0,
				"pots" : 0.89e19,
				"flavors" : [lp.nu_mu],
			}

		elif self.EXP_FLAG == flag_ps191_proposal:
			self.prop = {
				"name" : "ps191_proposal",
				"flux_norm" : 1.0/0.89e19/6e2/3e2/0.2*33,
				"area" : 6e2*3e2, # cm^2
				"length" : 12e2, # cm
				"baseline" : 123e2, # cm
				"eff_nu_e_e" : lambda x: 0.3,
				"emin" : 0.05,
				"emax" : 7.0,
				"pots" : 0.89e19,
				"flavors" : [lp.nu_mu],
			}

		elif self.EXP_FLAG == flag_nd280_flat:

			##############################################################################
			self.prop = {
				"name" : "nd280/FHC",
				# nus/cm^2/50 MeV/1e21 POT
				"flux_norm" : 1.0/1e21/0.05,
				"area" : 1.7e2*1.96e2, # cm^2
				"length" : 56*3, # cm
				"baseline" : 280e2, # cm
				"eff_nu_e_e" : lambda x: 1.0,
				"emin" : 0.05,
				"emax" : 10.0,
				"pots" : 12.34e20+6.29e20, ## approximately the same flux.
				"flavors" : [lp.nu_mu,lp.nu_mu_bar],
			}

		else:
			print(f'Could not find experiment {self.EXP_FLAG}.')

	######################################################
	# ALL FLUXES ARE NORMALIZED TO UNITS OF
	#   nus/cm^2/GeV/POT      
	######################################################
	def get_flux_func(self, parent=lp.K_plus, nuflavor=lp.nu_mu):

		fluxfile = f"{local_dir}/fluxes/{self.prop['name']}/{nuflavor.name}_{parent.name}.dat"
		try:
			e,f = np.genfromtxt(fluxfile, unpack = True)
		except IOError:
			print(f"Fluxfile {fluxfile} not found or not implemented. Skipping this decay.")
			return None
		else:		    
			flux = interpolate.interp1d(e, f*self.prop['flux_norm'], fill_value=0.0, bounds_error=False)
			return flux


nd280    = experiment(flag_nd280)
nd280_flat    = experiment(flag_nd280_flat)
nd280_fhc = experiment(flag_nd280_fhc)
nd280_rhc = experiment(flag_nd280_rhc)
ps191 = experiment(flag_ps191)
ps191_proposal = experiment(flag_ps191_proposal)

sbnd_numi_absorber = experiment(flag_sbnd_numi_absorber)
muboone_numi_absorber = experiment(flag_muboone_numi_absorber)
icarus_numi_absorber = experiment(flag_icarus_numi_absorber)
