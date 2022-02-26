import numpy as np

from particle import literals as lp

from .const import *

def tau_GeV_to_s(decay_rate):
    return 1./decay_rate/1.52/1e24

def L_GeV_to_cm(decay_rate):
    return 1./decay_rate/1.52/1e24*2.998e10

def lam(a,b,c):
    return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c

def I1_2body(x,y):
    return (1 - y - x * (2 + y - x) )*kallen_sqrt(1.0,x,y)

def L(x):
    if np.size(x)>1:
        mask = [x>0.05]
        xx = x[mask]
        xs = x[~mask]
        l = np.empty_like(x)
        l[mask] = np.log( (1-3*xx**2 - (1-xx**2)*np.sqrt(1-4*xx**2))/(x**2*(1+np.sqrt(1-4*xx**2))) )
        l[~mask] = np.log(xs**4 + 2*xs**6)
    else:
        if x <0.05:
            l = np.log(x**4 + 2*x**6)
        else:
            l = np.log( (1-3*x**2 - (1-x**2)*np.sqrt(1-4*x**2))/(x**2*(1+np.sqrt(1-4*x**2))) )

    return l    

def f1(x):
    return (1-14*x**2-2*x**4-12*x**6)*np.sqrt(1-4*x**2)+12*x**4*(x**4 - 1) * L(x)
def f2(x):
    return 4*(x**2*(2+10*x**2-12*x**4)*np.sqrt(1-4*x**2) + 6*x**4*(1-2*x**2+2*x**4)*L(x))

##
# a --> gamma gamma coupling (evaluates to complex number)
def F_alp_loop(z):
    if np.size(z)>1:
        z = z.astype(complex)
    else:
        z = complex(z)
    return 1/z * np.arctan(1/np.sqrt(-1 + 1/z))**2

#####################################################################
# all rates have been checked against Ref. arxiv.org/abs/2007.03701 
#####################################################################
def nui_l_P(params, initial_neutrino, final_lepton, final_hadron):

    mh = params.m4
    mp = final_hadron.mass/1e3
    ml = final_lepton.mass/1e3
    
    # Mixing required for CC N-like
    CC_mixing = np.sqrt(in_tau_doublet(final_lepton)*params.Utau4**2
                    + in_mu_doublet(final_lepton)*params.Umu4**2
                    + in_e_doublet(final_lepton)*params.Ue4**2)
    

    if (final_hadron==lp.pi_plus):
        Vqq = Vud
        fp  = f_charged_pion
    elif(final_hadron==lp.K_plus):
        Vqq = Vus
        fp  = f_charged_kaon
    else:
        print(f"Meson {final_hadron.name} no supported.")


    rate = (Gf*fp*CC_mixing*Vqq)**2 * mh**3/(16*np.pi) * I1_2body((ml/mh)**2, (mp/mh)**2)

    if params.HNLtype == 'majorana':
        rate *= 2

    return rate


def nui_nu_P(params, initial_neutrino, final_neutrino, final_hadron):

    mh = params.m4
    mp = final_hadron.mass/1e3
    
    NC_mixing = np.sqrt(in_tau_doublet(final_neutrino)*params.Utau4**2
                    + in_mu_doublet(final_neutrino)*params.Umu4**2
                    + in_e_doublet(final_neutrino)*params.Ue4**2)

    if (final_hadron==lp.pi_0):
        fp  = f_neutral_pion
    elif(final_hadron==lp.eta):
        fp  = f_neutral_eta
    else:
        fp=0
        print(f"Hadron {final_hadron} not supported")

    rate = (Gf*fp*NC_mixing)**2*mh**3/(32*np.pi)*(1-(mp/mh)**2)**2

    if params.HNLtype == 'majorana':
        rate *= 2

    return rate


def nui_nuj_ell1_ell2(params, initial_neutrino, final_neutrino, lepton_minus, lepton_plus):
  
    M = params.m4
    rate = 0.0
    exotic_rate = 0.0

    ## NC (+ CC if allowed + new channels)
    if same_particle(lepton_minus,lepton_plus):

        xb = lepton_minus.mass/1e3/M
        C1 = 1/4*(1- 4*s2w + 8*s2w**2)
        C2 = 1/2* (-s2w+2*s2w**2)
        CC_MIXING = np.sqrt(in_tau_doublet(final_neutrino)*params.Utau4**2
                        + in_mu_doublet(final_neutrino)*params.Umu4**2
                        + in_e_doublet(final_neutrino)*params.Ue4**2)

        # Is CC contribution possible?
        delta = same_doublet(final_neutrino, lepton_minus)

        rate = CC_MIXING**2*Gf**2*M**5/192/np.pi**3*((C1 + 2*s2w*delta)*f1(xb) + (C2 + s2w*delta)*f2(xb))

        if params.HNLtype=='majorana':
            rate *= 2

        ##############################################
        # dipole induced decay
        mh = M
        r = lepton_minus.mass/1e3/M
        reemin = 2*r #params.cut_ee/M
        if reemin < 2*r:
            reemin = 2*r

        dipole = np.sqrt(in_tau_doublet(final_neutrino)*params.dip_tau4**2
        + in_mu_doublet(final_neutrino)*params.dip_mu4**2
        + in_e_doublet(final_neutrino)*params.dip_e4**2)

        # full expression
        g = -1/12 * alphaQED * ( dipole )**( 2 ) * ( mh )**( 3 ) * ( np.pi )**( -2 ) * ( 3 * ( ( 1 + -4 * ( r )**( 2 ) ) )**( 1/2 ) + ( ( r )**( 2 ) * ( ( 1 + -4 * ( r )**( 2 ) ) )**( 1/2 ) * ( -5 + 2 * ( r )**( 2 ) ) + ( np.log( 4 ) + ( 8 * ( r )**( 6 ) * np.log( 1/2 * ( r )**( -1 ) * ( 1 + ( ( 1 + -4 * ( r )**( 2 ) ) )**( 1/2 ) ) ) + -2 * np.log( ( r )**( -1 ) * ( 1 + ( ( 1 + -4 * ( r )**( 2 ) ) )**( 1/2 ) ) ) ) ) ) )
        exotic_rate += g

        ##############################################
        # Z' induced decay
        NCprime_mixing =  np.sqrt(in_tau_doublet(final_neutrino)*params.Utau4**2
                        + in_mu_doublet(final_neutrino)*params.Umu4**2
                    + in_e_doublet(final_neutrino)*params.Ue4**2)

        exotic_rate += NCprime_mixing**2*params.GX**2*M**5/192.0/np.pi**3*(1-4*r**2)
        
        ##############################################
        # alp induced decay (assuming prompt on-shell alp decay!)
        if M > params.m_alp:
            br_a_to_ell_ell = 0.0
            if lepton_minus == lp.e_minus:
                br_a_to_ell_ell = params.alp_brs['e_e'] 
            elif lepton_minus == lp.mu_minus:
                br_a_to_ell_ell = params.alp_brs['mu_mu'] 
            else:
                br_a_to_ell_ell = 0.0

            exotic_rate += params.c_N**2*NCprime_mixing**2*mh**3/128/np.pi*params.inv_f_alp**2 * (1 - params.m_alp**2/mh**2)**2*br_a_to_ell_ell
        ##############################################


        if params.HNLtype=='majorana':
            exotic_rate *= 2


    ## CC only (neglecting the lightest lepton mass)
    elif (same_doublet(final_neutrino,lepton_minus)): 
        
        if lepton_minus.mass > lepton_plus.mass:
            heavy_lepton = lepton_minus
        else:
            heavy_lepton = lepton_plus

        xb = heavy_lepton.mass/1e3/M

        CC_MINUS = np.sqrt(in_tau_doublet(lepton_minus)*params.Utau4**2
                            + in_mu_doublet(lepton_minus)*params.Umu4**2
                            + in_e_doublet(lepton_minus)*params.Ue4**2)

        rate = CC_MINUS**2*Gf**2*M**5/192/np.pi**3*(1 - 8*xb**2 + 8*xb**6 - xb**8 - 12*xb**4*np.log(xb**2))
    
    ## CC only w/ LNV (neglecting the lightest lepton mass)
    elif (same_doublet(final_neutrino,lepton_plus) and params.HNLtype=='majorana'): 

        if lepton_minus.mass > lepton_plus.mass:
            heavy_lepton = lepton_minus
        else:
            heavy_lepton = lepton_plus

        xb = heavy_lepton.mass/1e3/M
        
        CC_plus = np.sqrt(in_tau_doublet(lepton_plus)*params.Utau4**2
                            + in_mu_doublet(lepton_plus)*params.Umu4**2
                            + in_e_doublet(lepton_plus)*params.Ue4**2)

        rate = CC_plus**2*Gf**2*M**5/192/np.pi**3*(1 - 8*xb**2 + 8*xb**6 - xb**8 - 12*xb**4*np.log(xb**2))

    else:
        rate = 0.0

    return rate + exotic_rate

def nui_nuj_gamma(params, initial_neutrino, final_neutrino):
    dipole = np.sqrt(in_tau_doublet(final_neutrino)*params.dip_tau4**2
            + in_mu_doublet(final_neutrino)*params.dip_mu4**2
            + in_e_doublet(final_neutrino)*params.dip_e4**2)

    rate = params.m4**3*dipole**2/4.0/np.pi

    if params.HNLtype=='majorana':
        rate *= 2 

    return rate


###############################
def nui_nuj_nuk_nuk(params, initial_neutrino, final_neutrinoj):
    mh = params.m4
    
    NC_mixing = np.sqrt(in_tau_doublet(final_neutrinoj)*params.Utau4**2
                    + in_mu_doublet(final_neutrinoj)*params.Umu4**2
                    + in_e_doublet(final_neutrinoj)*params.Ue4**2)

    rate = Gf**2/192.0/np.pi**3*mh**5 * NC_mixing**2

    if params.HNLtype == 'majorana':
        rate *= 2

    return rate



###############################
# Exotic particle rates
def a_to_ell_ell(params, lepton):
    mell = lepton.mass/1e3
    rate = params.m_alp*mell**2*params.inv_f_alp**2*params.c_e**2/8/np.pi*np.sqrt(1 - 4*mell**2/params.m_alp**2)
    return rate

def a_to_gamma_gamma(params):
    rate = params.g_gamma_gamma**2*params.m_alp**3/64/np.pi if not np.isnan(params.g_gamma_gamma) else 0.0
    return rate