import numpy as np
import vegas as vg
# import gvar as gv
import pandas as pd 
import os
# import random

from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

# import matplotlib.pyplot as plt
# from matplotlib import rc, rcParams
# from matplotlib.pyplot import *
# from matplotlib.legend_handler import HandlerLine2D

import fourvec
import hist_plot
import const 
import model 
import nuH_integrands as integrands
import nuH_MC as MC

FOLDER=''

def plot(MHEAVY,MLIGHT,HNLtype=const.MAJORANA,color='dodgerblue',lepton_mass=const.Me,HEL=-1):
	# PEDRO's benchmark point
	alphaD 		= 0.25
	epsilon 	= np.sqrt(2.0*1.0e-10/const.alphaQED)
	UE4 		= 0.0
	UMU4 		= np.sqrt(9.0e-7)
	UTAU4 		= 0.0
	UD4 		= np.sqrt(1.0 - (UE4**2 + UMU4**2 + UTAU4**2)) # using unitarity here
	GPRIME 		= 0.05
	CHI 		= np.sqrt(1e-6)
	MZPRIME = 0.0

	#########################
		# Set BSM parameters

	BSMparams = model.model_params()	

	BSMparams.gprime = GPRIME
	BSMparams.chi = CHI
	BSMparams.Ue4 = UE4
	BSMparams.Umu4 = UMU4 
	BSMparams.Utau4 = UTAU4
	BSMparams.Ue5 = 0.0 
	BSMparams.Umu5 = 0.0
	BSMparams.Utau5 = 0.0
	BSMparams.UD4 = np.sqrt(1.0 - UMU4**2)
	BSMparams.UD5 = 0.0
	BSMparams.m4 = MLIGHT
	BSMparams.m5 = MHEAVY
	BSMparams.Mzprime = MZPRIME
	BSMparams.Dirac = 1.0 

	BSMparams.set_high_level_variables()

	EnuH= 70.0 # GeV

	h_plus = MC.MC_events(HNLtype = HNLtype, EN=EnuH, mh=BSMparams.m5, mf=BSMparams.m4, mp=lepton_mass, mm=lepton_mass, helicity=HEL, BSMparams=BSMparams)
	h_minus = MC.MC_events(HNLtype = HNLtype, EN=EnuH, mh=BSMparams.m5, mf=BSMparams.m4, mp=lepton_mass, mm=lepton_mass, helicity=HEL, BSMparams=BSMparams)

	def Combine_MC_output(case0, case1, Ifactor_case0=1.0, Ifactor_case1=1.0, case0_flag=0, case1_flag=1):
		phnlp, pnuprimep, pe1p, pe2p, wp, Ip = case0
		phnlm, pnuprimem, pe1m, pe2m, wm, Im = case1

		phnl = np.array( np.append(phnlm, phnlp, axis=0) )
		pnuprime = np.array( np.append(pnuprimem, pnuprimep, axis=0) )
		pe1 = np.array( np.append(pe1m, pe1p, axis=0) )
		pe2 = np.array( np.append(pe2m, pe2p, axis=0) )
		w	= np.array( np.append(wp*Ifactor_case0, wm*Ifactor_case1, axis=0) )
		I	= Ip*Ifactor_case0 + Im*Ifactor_case1
		flags = np.append(np.ones(np.shape(phnlp)[0])*case0_flag,  np.ones(np.shape(phnlm)[0])*case1_flag )

		return phnl, pnuprime, pe1, pe2, w, I, flags



	bag = Combine_MC_output(h_plus.get_MC_events(),\
								h_minus.get_MC_events(), \
								Ifactor_case0=1.0, \
								Ifactor_case1 = 1.0,\
								case0_flag=0, \
								case1_flag=1)

	phnl, pnu, plm, plp, w, I, regime = bag
	size_samples=np.shape(regime)[0]

	###############################################
	# SAVE ALL EVENTS AS A PANDAS DATAFRAME
	columns = [['plm', 'plp', 'pnu', 'pHad'], ['t', 'x', 'y', 'z']]
	columns_index = pd.MultiIndex.from_product(columns)
	aux_data = [plm[:, 0],
			plm[:, 1],
			plm[:, 2],
			plm[:, 3],
			plp[:, 0],
			plp[:, 1],
			plp[:, 2],
			plp[:, 3],
			pnu[:, 0],
			pnu[:, 1],
			pnu[:, 2],
			pnu[:, 3],
			pnu[:, 0]*0,
			pnu[:, 1]*0,
			pnu[:, 2]*0,
			pnu[:, 3]*0,]
	
	aux_df = pd.DataFrame(np.stack(aux_data, axis=-1), columns=columns_index)
	aux_df['weight', ''] = w
	aux_df['regime', ''] = regime

	PATH_data = 'data/'
	# Create target Directory if it doesn't exist
	if not os.path.exists(PATH_data):
	    os.makedirs(PATH_data)
	if PATH_data[-1] != '/':
		PATH_data += '/'
	out_file_name = PATH_data+f"cc_MC_m5_{BSMparams.m5:.8g}_m4_{BSMparams.m4:.8g}_mlepton_{lepton_mass:.8g}_hel_{HEL}_{HNLtype[:3]}.pckl"

	aux_df.to_pickle(out_file_name)

	# pe1=plp
	# pe2=plm
	# pep=plp
	# pem=plm


	# Eshower = np.array([ fourvec.dot4(pe1[i],fourvec.k0)+fourvec.dot4(pe2[i],fourvec.k0) for i in range(size_samples)])
	# invmassSQR = np.array([ fourvec.dot4(pe1[i]+pe2[i],pe1[i]+pe2[i]) for i in range(size_samples)])

	# # electron kinematics
	# Eep = np.array([ fourvec.dot4(pep[i],fourvec.k0) for i in range(size_samples)])
	# Eem = np.array([ fourvec.dot4(pem[i],fourvec.k0) for i in range(size_samples)])
	# EN = np.array([ fourvec.dot4(phnl[i],fourvec.k0) for i in range(size_samples)])

	# Tep = np.array([ fourvec.dot4(pep[i],fourvec.k0)-lepton_mass for i in range(size_samples)])
	# Tem = np.array([ fourvec.dot4(pem[i],fourvec.k0)-lepton_mass for i in range(size_samples)])
	# TN = np.array([ fourvec.dot4(phnl[i],fourvec.k0)-MLIGHT for i in range(size_samples)])


	# # angle of separation between ee
	# cosdelta_ee = np.array([ fourvec.dot3(pep[i],pem[i])/np.sqrt( fourvec.dot3(pem[i],pem[i]))/np.sqrt( fourvec.dot3(pep[i],pep[i]) ) for i in range(size_samples)])
	# theta_ee = np.arccos(cosdelta_ee)*180.0/np.pi

	# # two individual angles
	# costheta_ep = np.array([ fourvec.dot3(pep[i],fourvec.kz)/np.sqrt( fourvec.dot3(pep[i],pep[i])) for i in range(size_samples)])
	# theta_ep = np.arccos(costheta_ep)*180.0/np.pi

	# costheta_em = np.array([ fourvec.dot3(pem[i],fourvec.kz)/np.sqrt( fourvec.dot3(pem[i],pem[i])) for i in range(size_samples)])
	# theta_em = np.arccos(costheta_em)*180.0/np.pi

	# # this is the angle of the combination of ee with the neutrino beam
	# costheta_comb = np.array([ fourvec.dot3(pem[i]+pep[i],fourvec.kz)/np.sqrt( fourvec.dot3(pem[i]+pep[i],pem[i]+pep[i])) for i in range(size_samples)])
	# theta_comb = np.arccos(costheta_comb)*90.0/np.pi



	# title = r"$E_{\nu_4}= %.2f\,$ GeV, $m_{\nu_4} = %.0f\,$ MeV, $M_{Z^\prime} = %.2f\,$ GeV"%(EnuH,MLIGHT*1e3,MZPRIME)

	# hist_plot.histogram1D("plots/"+FOLDER+"/cdelta_theta.pdf", [cosdelta_ee, w, I], 0.9, 1.0, r"$\cos(\Delta \theta_{ee})$", title, 100)
	# hist_plot.histogram1D("plots/"+FOLDER+"/delta_theta.pdf", [theta_ee, w, I], 0.0, 20.0, r"$\Delta \theta_{ee}$ ($^\circ$)", title, 100)

# fsize = 9
# rc('text', usetex=True)
# params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
# 				'figure.figsize':(1.2*3.7,1.4*2.3617)	}
# rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
# rcParams.update(params)
# D=0.83;I=0.13
# axes_form1  = [0.15,I,0.78,D/3]
# axes_form2  = [0.15,I+D/3,0.78,D/3]
# axes_form3  = [0.15,I+2*D/3,0.78,D/3]
# fig = plt.figure()
# ax1  = fig.add_axes(axes_form1)
# ax2  = fig.add_axes(axes_form2)
# ax3  = fig.add_axes(axes_form3)


# ax2.set_xticks([])
# ax3.set_xticks([])
# ax1.set_xlabel(r"$E_{e^+}+E_{e^-}$ (GeV)",fontsize=fsize)


mass = 5.0 # GeV 
# plot(mass,0.00,color='dodgerblue',lepton_mass=const.Me, HNLtype=const.DIRAC,HEL=-1)
plot(mass,0.00,color='dodgerblue',lepton_mass=const.Mmu, HNLtype=const.DIRAC,HEL=-1)
# plot(mass,0.00,color='dodgerblue',lepton_mass=const.Me, HNLtype=const.DIRAC,HEL=+1)
plot(mass,0.00,color='dodgerblue',lepton_mass=const.Mmu, HNLtype=const.DIRAC,HEL=+1)

# plot(mass, 0.00, color='dodgerblue', lepton_mass=const.Me, HNLtype=const.MAJORANA, HEL=-1)
plot(mass, 0.00, color='dodgerblue', lepton_mass=const.Mmu, HNLtype=const.MAJORANA, HEL=-1)
# plot(mass, 0.00, color='dodgerblue', lepton_mass=const.Me, HNLtype=const.MAJORANA, HEL=+1)
plot(mass, 0.00, color='dodgerblue', lepton_mass=const.Mmu, HNLtype=const.MAJORANA, HEL=+1)