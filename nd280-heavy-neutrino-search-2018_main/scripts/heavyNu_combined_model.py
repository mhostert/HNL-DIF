import ROOT
from pymc import deterministic, potential, stochastic, Poisson, Uniform, Uninformative, distributions
from math import log, sqrt
import numpy as np
import os

# utilitary functions

def VectorToMatrix(vec, ncols):
    mat = ROOT.TMatrixD(vec.GetNrows()/ncols, ncols);
    for i in range(0, mat.GetNrows()):
        for j in range(0, mat.GetNcols()):
            mat[i][j] = vec(i*ncols+j)
    return mat

def TMatrixToArray(mat):
    ncols = mat.GetNcols()
    nrows = mat.GetNrows()
    arr = np.zeros([nrows,ncols])
    for i in range(0, mat.GetNrows()):
        for j in range(0, mat.GetNcols()):
            arr[i][j] = mat[i][j]
    return arr


# make the model

def make_model(m = 270, syst = True, saveBigTraces = False, profile=0, prior=0):

    # -----------------

    f_flux = ROOT.TFile.Open("inputs/vector_flux.root", "read")
    
    flux0 = f_flux.Get("vector_flux_"+str(m))
    flux0 = np.asarray(flux0)
    
    cov_flux = f_flux.Get("covmat_flux_"+str(m))
    cov_flux = TMatrixToArray(cov_flux)
    for i in range(cov_flux.shape[0]):
        for j in range(cov_flux.shape[1]):
            cov_flux[i][j] *= flux0[i]*flux0[j]
        if cov_flux[i][i]==0:
            cov_flux[i][i] = 1
    invcov_flux = np.linalg.inv(cov_flux)
    
    # -----------------
    
    f_eff1 = ROOT.TFile.Open("inputs/effmat_eff.root", "read")
    
    m_eff0 = f_eff1.Get("effmat_eff_"+str(m))
    m_eff0 = TMatrixToArray(m_eff0)
    eff0 = np.reshape(m_eff0, m_eff0.size)
    
    # -----------------
    
    f_eff2 = ROOT.TFile.Open("inputs/covmat_eff.root", "read")
    
    cov_eff = f_eff2.Get("covmat_eff_"+str(m))
    cov_eff = TMatrixToArray(cov_eff)
    
    for i in range(cov_eff.shape[0]):
        for j in range(cov_eff.shape[1]):
            cov_eff[i][j] *= eff0[i]*eff0[j]
        cov_eff[i][i] *= 1.01
        if cov_eff[i][i]==0:
            cov_eff[i][i] = 1e-30
    invcov_eff = np.linalg.inv(cov_eff)
    
    # -----------------
    
    f_bkg = ROOT.TFile.Open("inputs/vector_bkg.root", "read")
    
    bkg0 = f_bkg.Get("vector_bkg")
    bkg0 = np.asarray(bkg0)
    e_bkg = f_bkg.Get("error_bkg")
    e_bkg = np.asarray(e_bkg)
	
    # -----------------
	
    f_data = ROOT.TFile.Open("inputs/vector_data.root", "read")
	
    observed_events = f_data.Get("vector_data")
    observed_events = np.asarray(observed_events)
    
    # -----------------
    
    contribution = np.zeros(32)
    contribution[0]=22;  contribution[1]=12;
    contribution[2]=21;  contribution[3]=11;
    contribution[4]=22;  contribution[5]=12;
    contribution[6]=21;  contribution[7]=11; 
    contribution[8]=23;  contribution[9]=13;
    contribution[10]=24; contribution[11]=14;
    contribution[12]=22; contribution[13]=12;
    contribution[14]=21; contribution[15]=11;
    contribution[16]=22; contribution[17]=12;
    contribution[18]=21; contribution[19]=11;
    contribution[20]=22; contribution[21]=12;
    contribution[22]=21; contribution[23]=11;
    contribution[24]=21; contribution[25]=11;
    contribution[26]=24; contribution[27]=14;
    contribution[28]=12; contribution[29]=11;
    contribution[30]=12; contribution[31]=11

    N_modes = flux0.size/2
    N_channels = bkg0.size
    
    # -----------------
    # DEFINE STOCHASTIC VARIABLES
    # -----------------

    Ue2_0   = Uniform('Ue2',   lower=-16, upper=1, value=-9)
    Umu2_0  = Uniform('Umu2',  lower=-16, upper=1, value=-9)
    Utau2_0 = Uniform('Utau2', lower=-16, upper=1, value=-9)
    
    if (profile == 1):
        Ue2_0   = Uniform('Ue2',   lower=-16, upper=1, value=-9)
        Umu2_0  = Uniform('Umu2',  lower=-16, upper=1, value=-9)
        Utau2_0 = -np.inf
    if (profile == 2):
        Ue2_0   = Uniform('Ue2',   lower=-16, upper=1, value=-9) 
        Umu2_0  = -np.inf
        Utau2_0 = -np.inf
    if (profile == 3):
        Ue2_0   = -np.inf
        Umu2_0  = Uniform('Umu2',  lower=-16, upper=1, value=-9)
        Utau2_0 = -np.inf

    # -----------------
    # PRIORS ON NUISANCE PARAMETERS
    # -----------------
    
    if syst:
    
        @stochastic(name='flux', dtype=float, trace=saveBigTraces)
        def s_flux(value=np.array(flux0), mu=flux0, invcov=invcov_flux):
            x = mu-value
            temp = np.dot(invcov, x)
            return -0.5*np.dot(x, temp)
    
        @stochastic(name='eff', dtype=float, trace=saveBigTraces)
        def s_eff(value=np.array(eff0), mu=eff0, invcov=invcov_eff):
            x = mu-value
            temp = np.dot(invcov, x)
            return -0.5*np.dot(x, temp)

        @stochastic(name='bkg', dtype=float, trace=saveBigTraces)
        def s_bkg(value=np.array(bkg0), bkg=bkg0, ebkg=e_bkg):
            log_bkg = 0
            for i in range(0, len(bkg)):
                if bkg[i] <= 0:
                    continue
                if value[i] <= 0:
                    return -np.inf
                mean = bkg[i]
                std_dev = ebkg[i]
                m     = mean/sqrt( 1+(std_dev*std_dev)/(mean*mean) )
                sigma = sqrt( log( 1+(std_dev*std_dev)/(mean*mean) ) )
                log_bkg += -0.5*(log(value[i]/m)/sigma)**2
            return log_bkg

    else:
        s_flux = flux0
        s_eff  = eff0
        s_bkg  = bkg0

    # -----------------
    # PRIORS ON SIGNAL PARAMETERS
    # -----------------

    if (prior==0):
        if (profile==0):
            @potential
            def prior_U(Ue2=Ue2_0, Umu2=Umu2_0, Utau2=Utau2_0):
                return log(10)*(Ue2+Umu2+Utau2)
        if (profile==1):
            @potential
            def prior_U(Ue2=Ue2_0, Umu2=Umu2_0):
                return log(10)*(Ue2+Umu2)
        if (profile==2):
            @potential
            def prior_U(Ue2=Ue2_0):
                return log(10)*Ue2
        if (profile==3):
            @potential
            def prior_U(Umu2=Umu2_0):
                return log(10)*Umu2

    if (prior==1):
        @potential
        def prior_U(Ue2=Ue2_0, Umu2=Umu2_0, Utau2=Utau2_0):
            return log(10)*2*(Ue2+Umu2+Utau2)
	
    if (prior==2):
        @potential
        def prior_U(Ue2=Ue2_0, Umu2=Umu2_0, Utau2=Utau2_0):
            return log(10)*0.5*(Ue2+Umu2+Utau2)   

    # -----------------
    # DEFINE DETERMINISTIC VARIABLES
    # -----------------
    
    @deterministic(plot=False, trace=saveBigTraces)
    def rate(flux=s_flux,
             v_eff=s_eff,
             bkg=s_bkg,
             contrib=contribution,
             Ue2=10**(Ue2_0), Umu2=10**(Umu2_0), Utau2=10**(Utau2_0)):
    
        out = np.zeros_like(bkg)
        eff = np.reshape(v_eff, (flux.size/2, bkg.size))
    
        # loop on all the observed channels
        for i in range(0, len(out)):
    
            out[i] += bkg[i]
            isRHC_mode = False
            if i>= len(out)/2:
                isRHC_mode = True
            
            # loop on all the expected signal contributions
            for j in range(0, len(contrib)):

                factor=1

                if contrib[j]==0:
                    continue
                
                c_prod  = int(contrib[j]/10)
                c_decay = contrib[j]%10
    
                if (c_prod == 1):
                    factor *= Ue2
                elif (c_prod == 2):
                    factor *= Umu2
                else:
                    print("Error on the used production mode (contrib={0})".format(contrib[j]))
                if (c_decay == 1):
                    factor *= Ue2
                elif (c_decay == 2):
                    factor *= Umu2
                elif (c_decay == 3):
                    factor *= (Ue2+Utau2)/2
                elif (c_decay == 4):
                    factor *= (Umu2+Utau2)/2
                else:
                    print("Error on the used decay mode (contrib={0})".format(contrib[j]))
    
                if isRHC_mode:
                    k_flux = j + len(flux)/2
                else:
                    k_flux = j

                if eff[j,i] > 1e-10 and flux[k_flux] > 1e3: # not consider small variation of eff or flux from 0
                    out[i] += factor*eff[j,i]*flux[k_flux]
 
        return out
    
    events = Poisson('events', mu=rate, value=observed_events, observed=True)
    return locals(), cov_flux, cov_eff
