import os
import sys
import numpy as np
import pandas as pd

from . import const

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

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
