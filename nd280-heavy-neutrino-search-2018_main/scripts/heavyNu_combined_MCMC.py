from heavyNu_combined_model import *

import sys
import os
import pymc
import numpy as np

from pylab import hist, show
from pymc.Matplot import plot

import mkl

            
# suffix  = name of the output file
# nsteps  = number of Markov Chain Monte Carlo steps
# syst    = boolean to enable/disable the systematics
# profile = define which variables U_alpha^2 to consider
#           0 : Ue2,Umu2,Utau2 all free
#           1 : Utau2 = 0
#           2 : Umu2  = Utau2 = 0
#           3 : Ue2   = Utau2 = 0
# prior   = define the used prior on U_alpha^2
#           0 : flat in U^2
#           1 : flat in U^4
#           2 : flat in U
# mass    = heavy neutrino mass to consider (in MeV/c^2)

def main(suffix, nsteps, syst=0, profile=0, prior=0, mass=270):

    # prevent numpy from multi-threading
    mkl.set_num_threads(1)

    parameters_of_interest = ["Ue2", "Umu2", "Utau2"] 
    enableSyst = syst
    saveBigTraces = False

    # -----------------

    # build the model
    model, cov_flux, cov_eff = make_model(m=mass, syst=enableSyst, saveBigTraces=saveBigTraces, profile=profile, prior=prior)

    # output file
    dbname = 'db/MCMC_'+suffix+'_'+str(mass)+'.hdf5'
    M = pymc.MCMC(model, db='hdf5', dbname=dbname, dbmode='a')
        
    if enableSyst:
        M.use_step_method(pymc.AdaptiveMetropolis, [M.s_flux], delay=50000, cov=1/sqrt(cov_flux.shape[0])*cov_flux)
        M.use_step_method(pymc.AdaptiveMetropolis, [M.s_eff], delay=50000, cov=1/sqrt(cov_eff.shape[0])*cov_eff)

    M.db
    
    # run MCMC
    M.sample(iter=nsteps, thin=10, progress_bar=False)
    M.db.close()

    
if __name__ == '__main__':
    main(suffix=str(sys.argv[1]), nsteps=int(sys.argv[2]), syst=bool(sys.argv[3]), profile=int(sys.argv[4]), prior=int(sys.argv[5]), mass=int(sys.argv[6]))
