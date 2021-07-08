import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from math import log

import ROOT
import pymc
import sys,os

import numpy as np
import numpy.ma as ma
import array

import glob

import scipy.stats
import pylab


# compute the limits
def main(suffix):

	# burh the same steps of the MCMC
    Nburn = 1000
    
    suffix2 = suffix.replace("*","")
    suffix2 = suffix2.replace("/","_")

    file_o = open('results/sensPython_'+suffix2+'.dat', 'w')

    xbins = np.array([0])
    xbins = np.append(xbins, np.logspace(-15, 0, num=450))
    xbins = array.array('d', xbins)

	# loop on masses
    for mass in range(140, 500, 10):
        
        Ue2 = np.array([])
        Umu2 = np.array([])
        Utau2 = np.array([])

        dbname = 'db/MCMC_'+suffix+'_'+str(mass)+'.hdf5'
        for file in glob.glob(dbname):

            try:
                db = pymc.database.hdf5.load(file)
                for i in range(db.nchains()):
                    Ue2_new   = db.trace('Ue2',   chain=i)[Nburn:]
                    Umu2_new  = db.trace('Umu2',  chain=i)[Nburn:]
                    Utau2_new = db.trace('Utau2', chain=i)[Nburn:]
                    Ue2   = np.append(Ue2,   Ue2_new)
                    Umu2  = np.append(Umu2,  Umu2_new)
                    Utau2 = np.append(Utau2, Utau2_new)
            except:
                print('Corrupted file: {0}'.format(file))
                continue
                    
            db.close()

        length = max(len(Ue2),len(Umu2),len(Utau2))
        if (length==0):
             print("Lists are empty !")
             continue

        Ue2   = 10**Ue2
        Umu2  = 10**Umu2
        Utau2 = 10**Utau2
  
        limit_Ue2 = 0
        if (len(Ue2)>0):
            limit_Ue2 = np.percentile(Ue2, 90)
            
        limit_Umu2 = 0
        if (len(Umu2)>0):
            limit_Umu2 = np.percentile(Umu2, 90)
            
        limit_Utau2 = 0
        if (len(Utau2)>0):
            limit_Utau2 = np.percentile(Utau2, 90)

        print('{0} => {1}, {2}, {3}'.format(mass, 
                                            limit_Ue2, 
                                            limit_Umu2,
                                            limit_Utau2))
        
		# write the limits to output file
        file_o.write(str(mass) + ' ' + str(limit_Ue2) + ' ' + str(limit_Umu2) + ' ' + str(limit_Utau2) + '\n')
    
        hist = ROOT.TH3F("hist_"+str(mass), "hist_"+str(mass), 
            len(xbins)-1, np.asarray(xbins,'d'), 
            len(xbins)-1, np.asarray(xbins,'d'), 
            len(xbins)-1, np.asarray(xbins,'d'))
        for i in range(len(Ue2)):
            hist.Fill(Ue2[i], Umu2[i], Utau2[i])

        hist.GetXaxis().SetTitle("U_{e}^{2}")
        hist.GetYaxis().SetTitle("U_{#mu}^{2}")
        hist.GetZaxis().SetTitle("U_{#tau}^{2}")

        f = ROOT.TFile.Open("results/hist_3D_"+suffix2+"_"+str(mass)+".root", "recreate")
        hist.Write()
        f.Close()

    file_o.close()


if __name__ == '__main__':
    main(suffix=str(sys.argv[1]))
