import os
import sys
import numpy as np
import pandas as pd

from . import const
from . import pdg
from dark_news.decayer import decay_position
#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv
from dark_news.geom import old_geometry_muboone
from dark_news.fourvec import dot4

def print_events_to_pandas(PATH_data, bag, BSMparams, l_decay_proper=0.0, out_file_name='samples'):
	# events
	pN   = bag['P_outgoing_HNL']
	pnu   = bag['P_out_nu']
	pZ   = bag['P_em']+bag['P_ep']
	plm  = bag['P_em']
	plp  = bag['P_ep']
	pHad = bag['P_outgoing_target']
	w = bag['w']
	w_decay = bag['w_decay']
	I = bag['I']
	I_decay = bag['I_decay']
	m4 = bag['m4_scan']
	mzprime = bag['mzprime_scan']
	regime = bag['flags']

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
			pHad[:, 0],
			pHad[:, 1],
			pHad[:, 2],
			pHad[:, 3],
			]
	aux_df = pd.DataFrame(np.stack(aux_data, axis=-1), columns=columns_index)

	aux_df['weight', ''] = w
	aux_df['weight_decay', ''] = w_decay
	aux_df['regime', ''] = regime
	aux_df['m4', ''] = m4
	aux_df['mzprime', ''] = mzprime


	# Create target Directory if it doesn't exist
	if not os.path.exists(PATH_data):
	    os.makedirs(PATH_data)
	if PATH_data[-1] != '/':
		PATH_data += '/'
	full_file_name = PATH_data+out_file_name

	aux_df.to_pickle(full_file_name)


#######
# not relevant anymore. 
def print_unweighted_events_to_HEPEVT(PATH_data, bag, TOT_EVENTS, BSMparams, l_decay_proper=0.0):
	# events
	pN   = bag['P_outgoing_HNL']
	pnu  = bag['P_out_nu']
	pZ   = bag['P_em']+bag['P_ep']
	plm  = bag['P_em']
	plp  = bag['P_ep']
	pHad = bag['P_outgoing_target']
	Mhad = np.sqrt(dot4(pHad, pHad))
	w = bag['w']
	I = bag['I']
	regime = bag['flags']

	# Accept/reject method -- samples distributed according to their weights
	AllEntries = np.array(range(np.shape(plm)[0]))
	AccEntries = np.random.choice(AllEntries, size=TOT_EVENTS, replace=True, p=w/np.sum(w))
	pN, plp, plm, pnu, pHad, w, regime  = pN[AccEntries], plp[AccEntries], plm[AccEntries], pnu[AccEntries], pHad[AccEntries], w[AccEntries], regime[AccEntries]

	# sample size (# of events)
	size = np.shape(plm)[0]

	# get scattering positions
	t,x,y,z = old_geometry_muboone(size)

	# decay events
	t_decay,x_decay,y_decay,z_decay = decay_position(pN, l_decay_proper_cm=l_decay_proper)

	###############################################
	# SAVE ALL EVENTS AS A HEPEVT .dat file
	hepevt_file_name = PATH_data+"MC_m4_"+format(BSMparams.m4,'.8g')+"_mzprime_"+format(BSMparams.Mzprime,'.8g')+".dat"
	# Open file in write mode
	f = open(hepevt_file_name,"w+") 
	
	# f.write("%i\n",TOT_EVENTS)
	# loop over events
	for i in range(TOT_EVENTS):
		f.write("%i 4\n" % i)
		f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.electron,plm[i][1], plm[i][2], plm[i][3], plm[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))
		f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.positron,plp[i][1], plp[i][2], plp[i][3], plp[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))	
		f.write("2 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.numu,pnu[i][1], pnu[i][2], pnu[i][3], pnu[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i], t_decay[i]))
		if (regime[i] == const.COHRH or regime[i] == const.COHLH):
			f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.Argon40,pHad[i][1], pHad[i][2], pHad[i][3], pHad[i][0], Mhad[i],x[i],y[i],z[i],t[i]))
		elif (regime[i] == const.DIFRH or regime[i] == const.DIFLH):
			f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.proton,pHad[i][1], pHad[i][2], pHad[i][3], pHad[i][0], Mhad[i],x[i],y[i],z[i],t[i]))
		else:
			print('Error! Cannot find regime of event ', i)
	f.close()

