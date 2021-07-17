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

vertices = {'CConly': [1,-1,0], "NConly" : [0,0,1], 'CCandNC': [1,-1,1], 'NCandCC': [1,-1,1]}
helicities = {'LH': -1, 'RH': +1, 'both': 0}

def generate_events(MHEAVY, mixings, HNLtype='majorana',lepton_mass=const.m_e, HEL='LH', modify_vertex='NCandCC', convolve_flux=True):

	#########################
	# Set BSM parameters

	my_hnl = model.hnl_model(MHEAVY, mixings = mixings, minimal=True, HNLtype=HNLtype)	
	my_hnl.set_high_level_variables()

	if HEL == 'both':

		my_MC = MC.MC_events(HNLtype = HNLtype, mh=my_hnl.m4, mf=0.0, mp=lepton_mass, mm=lepton_mass, helicity=+1, BSMparams=my_hnl, convolve_flux=convolve_flux)
		
		my_MC.CCflag1 = vertices[modify_vertex][0]
		my_MC.CCflag2 = vertices[modify_vertex][1]
		my_MC.NCflag = vertices[modify_vertex][2]
		
		RH = my_MC.get_MC_events()	

		my_MC = MC.MC_events(HNLtype = HNLtype, mh=my_hnl.m4, mf=0.0, mp=lepton_mass, mm=lepton_mass, helicity=-1, BSMparams=my_hnl, convolve_flux=convolve_flux)
		
		my_MC.CCflag1 = vertices[modify_vertex][0]
		my_MC.CCflag2 = vertices[modify_vertex][1]
		my_MC.NCflag = vertices[modify_vertex][2]
		
		LH = my_MC.get_MC_events()

		combined=[]
		for i in range(len(LH)):
			combined.append(np.concatenate((RH[i],LH[i]), axis=0 ))

		phnl, pnu, plm, plp, w = combined 
	else:
		my_MC = MC.MC_events(HNLtype = HNLtype, mh=my_hnl.m4, mf=0.0, mp=lepton_mass, mm=lepton_mass, helicity=helicities[HEL], BSMparams=my_hnl, convolve_flux=convolve_flux)
		
		my_MC.CCflag1 = vertices[modify_vertex][0]
		my_MC.CCflag2 = vertices[modify_vertex][1]
		my_MC.NCflag = vertices[modify_vertex][2]
		
		phnl, pnu, plm, plp, w = my_MC.get_MC_events()
	

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
	out_file_name = PATH_data+f"MC_m4_{my_hnl.m4:.8g}_mlepton_{lepton_mass:.8g}_hel_{HEL}_{HNLtype}_{modify_vertex}.pckl"

	print(out_file_name)
	aux_df.to_pickle(out_file_name)


ND280_B_FIELD = 0.2 #Tesla
# radius in cm, Bfield in Tesla, and p in GeV
def radius_of_curvature(p, Bfield):
	return 1e2*p/0.3/Bfield

def tilted_circle(R,theta,x):
	return np.sqrt(R**2 * np.cos(2*theta) + R**2 - 4*x*R*np.sin(theta)-2*x**2)/np.sqrt(2)-R*np.cos(theta)


def distance_between_particles(theta, R1, R2, x):
	y1 = tilted_circle(R1, theta/2, x)
	y2 = tilted_circle(R2, theta/2, x)
	return np.abs(y1+y2)

def compute_kin_vars(df):
    for comp in ['t','x','y','z']:
        df['pee', comp] = df['plm', comp] + df['plp', comp]
        df['pdark', comp] = df['plm', comp] + df['plp', comp] + df['pnu', comp]
    df['easy', ''] = (df['plp','t']-df['plm','t'])/(df['plm','t']+df['plp','t']-inv_mass(df['plm'])-inv_mass(df['plp']))
    df['ehnl', ''] = df['phnl','t']
    df['eplus', ''] = df['plp','t']
    df['eminus', ''] = df['plm','t']
    df['e_smallest', ''] = np.minimum(df['plm','t'],df['plp','t'])
    df['e_largest', ''] = np.maximum(df['plm','t'],df['plp','t'])
    df['miss_pt', ''] = np.sqrt(df['pnu','x']**2+df['pnu','y']**2)
    df['ee_mass', ''] = inv_mass(df['pee'])
    df['m_t', ''] = df['miss_pt','']+np.sqrt(df['miss_pt','']**2+df['ee_mass','']**2)
    df['ee_costheta', ''] = costheta(df['plm'], df['plp'])
    df['ee_theta', ''] = np.arccos(df['ee_costheta', ''])
    df['ee_beam_costheta', ''] = df['pee', 'z']/np.sqrt(dot3_df(df['pee'], df['pee']))
    df['ee_beam_theta', ''] = np.arccos(df['ee_beam_costheta', ''])
    df['ee_momentum', ''] = np.sqrt(dot3_df(df['pee'], df['pee']))
    df['experimental_t', ''] = (df['plm','t'] - df['plm','z'] + df['plp','t'] - df['plp','z'])**2 +\
                                   df['plm','x']**2 + df['plm','y']**2 + df['plp','x']**2 + df['plp','y']**2
    df['race_to_b=1cm', ''] = 0.5/np.sqrt(1/np.cos(df['ee_theta', '']/2) -1 )    
    df['radius_minus', ''] = radius_of_curvature(df['eminus', ''], ND280_B_FIELD)
    df['radius_plus', ''] = radius_of_curvature(df['eplus', ''], ND280_B_FIELD)
    df['distance_between_circles_at_10cm'] = distance_between_particles(df['ee_theta', ''], df['radius_minus', ''], df['radius_plus', ''],  10)
    df['distance_between_circles_at_20cm'] = distance_between_particles(df['ee_theta', ''], df['radius_minus', ''], df['radius_plus', ''],  20)
    df['distance_between_circles_at_50cm'] = distance_between_particles(df['ee_theta', ''], df['radius_minus', ''], df['radius_plus', ''],  50)



