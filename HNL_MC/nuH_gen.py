import numpy as np
import vegas as vg
import pandas as pd 
import os

from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

from . import const
from . import model 
from . import nuH_integrands as integrands
from . import nuH_MC as MC
from . fourvec import *


def generate_events(MHEAVY, mixings, HNLtype='majorana',lepton_mass=const.m_e, HEL=-1, NCflag=1, CCflag1=1, CCflag2=-1):

	#########################
	# Set BSM parameters

	my_hnl = model.hnl_model(MHEAVY, mixings = mixings, minimal=True, HNLtype=HNLtype)	
	my_hnl.set_high_level_variables()

	EnuH= 2.0 # GeV

	my_MC = MC.MC_events(HNLtype = HNLtype, EN=EnuH, mh=my_hnl.m4, mf=0.0, mp=lepton_mass, mm=lepton_mass, helicity=HEL, BSMparams=my_hnl)
	my_MC.NCflag = NCflag
	my_MC.CCflag1 = CCflag1
	my_MC.CCflag2 = CCflag2

	phnl, pnu, plm, plp, w, I = my_MC.get_MC_events()
	size_samples=np.shape(w)[0]

	###############################################
	# SAVE ALL EVENTS AS A PANDAS DATAFRAME
	columns = [ ['plm', 'plp', 'pnu', 'phnl'], ['t', 'x', 'y', 'z']]

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
				phnl[:, 0],
				phnl[:, 1],
				phnl[:, 2],
				phnl[:, 3]]
	

	aux_df = pd.DataFrame(np.stack(aux_data, axis=-1), columns=columns_index)
	aux_df['weight', ''] = w

	PATH_data = 'data/mc_samples/'
	# Create target Directory if it doesn't exist
	if not os.path.exists(PATH_data):
	    os.makedirs(PATH_data)
	if PATH_data[-1] != '/':
		PATH_data += '/'
	out_file_name = PATH_data+f"MC_m4_{my_hnl.m4:.8g}_mlepton_{lepton_mass:.8g}_hel_{HEL}_{HNLtype[:3]}_{CCflag1}{CCflag2}{NCflag}.pckl"

	aux_df.to_pickle(out_file_name)

def compute_kin_vars(df):
    for comp in ['t','x','y','z']:
        df['pee', comp] = df['plm', comp] + df['plp', comp]
        df['pdark', comp] = df['plm', comp] + df['plp', comp] + df['pnu', comp]
    df['easy', ''] = (df['plp','t']-df['plm','t'])/(df['plm','t']+df['plp','t']-inv_mass(df['plm'])-inv_mass(df['plp']))
    df['eplus', ''] = df['plp','t']
    df['eminus', ''] = df['plm','t']
    df['miss_pt', ''] = np.sqrt(df['pnu','x']**2+df['pnu','y']**2)
    df['ee_mass', ''] = inv_mass(df['pee'])
    df['m_t', ''] = df['miss_pt','']+np.sqrt(df['miss_pt','']**2+df['ee_mass','']**2)
    df['ee_costheta', ''] = costheta(df['plm'], df['plp'])
    df['ee_beam_costheta', ''] = df['pee', 'z']/np.sqrt(dot3_df(df['pee'], df['pee']))
    df['ee_momentum', ''] = np.sqrt(dot3_df(df['pee'], df['pee']))
    df['experimental_t', ''] = (df['plm','t'] - df['plm','z'] + df['plp','t'] - df['plp','z'])**2 +\
                                   df['plm','x']**2 + df['plm','y']**2 + df['plp','x']**2 + df['plp','y']**2

