#ifndef FLUX_H_
#define FLUX_H_

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>



#define DET_NOCUTS 0
#define DET_SBND 1
#define DET_MUBOONE 2
#define DET_ICARUS 3
#define DET_PS191 4
#define DET_ALL_SBN 5

#define MPION  0.13957
#define MPI0   0.13497
#define MKAON  0.49367
#define MMU    0.10566
#define	ME     0.00051
#define M2GEV 5.06842e15
#define INVS2GEV 6.58e-25
#define FKA 0.1561
#define VUS 0.2252

#define FUDGE_SCALE  false
#define FUDGE_SHIFT false


double shrock_FM(double x, double y);
double shrock_lambda(double x, double y, double z);
double shrock_rho(double x, double y);
double the_shrock_factor(double mM, double ms, double ml);

double other_kaon(double ms,double ml);

double Meson2bodWidth(double Mmeson, double Mn, double Ml, double Fmeson, double VUD);
double MesonBr(double Mmeson, double Mn, double Ml, double Fmeson, double VUD);
double MesonBrPi(double ms, double ml);
double MesonBrKa(double ms, double ml);

double heaviside(double);
double geometry_factor(double);

double fluxPionUu(double E);
double fluxPionUe(double E); //scaled from miniboone
double fluxKaonUu(double E);
double fluxKaonUe(double E);// scaled from miniboone
double fluxKaon2Pion(double E);

double fluxPS191KaonUu(double E);
double fluxPS191PionUu(double E);
double fluxPS191PionUe(double E);
double fluxPS191KaonUe(double E);
double fluxPS191Kaon3Ue(double E);
double SBNDmodificer2Icarus(double x);

double flux(double Ev, double ms, double Ue,  double Uu, double Ut, int DET);

int plot_flux(double, int);
int plot_neutrino_flux( int);

struct Sflux{
	    double MuPi;
	    double EPi;
	    double MuKa;
	    double EKa;
	    double EKa3;
	    double MuKaOther;

	    double total()
	    {
		    return MuPi+EPi+MuKa+EKa+EKa3+MuKaOther;
	    }
	    Sflux();

} ;

struct Sflux flux_sterile(double E, double ms, double Ue, double Um, double Ut, int Det);
struct Sflux flux_neutrino(double E, double ms, int Det);

#endif

