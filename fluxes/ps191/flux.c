#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <nlopt.hpp>
#include <iomanip>

#include "flux.h"



double shrock_FM(double x, double y)
{
	return x+y-pow(x-y,2.0);

}

double shrock_lambda(double x, double y, double z)
{
	return x*x+y*y+z*z-2.0*(x*y+y*z+z*x);
	
}

double shrock_rho(double x, double y)
{
	return shrock_FM(x,y)*sqrt(shrock_lambda(1,x,y));
	
}

double the_shrock_factor(double mM, double ms, double ml)
{
	double dMa = pow(ml/mM,2.0);
	double dMi = pow(ms/mM,2.0);	
	double ans = 0;

	if(mM>=ml+ms)
	{
	 ans=	shrock_rho(dMa,dMi)/(dMa*pow(1-dMa,2.0));
	}

	return ans;
}



double other_kaon(double ms,double ml){

	return	Meson2bodWidth(MKAON,ms,ml,FKA,VUS)/(1.238e-8 *0.2066 )*INVS2GEV;

}





double Meson2bodWidth(double mp,double ms, double ml,double fpi,double vud){
	double gf = 0.00001166;

	return (1/(8*3.14159))*fpi*fpi*vud*vud*gf*gf*mp*((ml*ml+ms*ms)-pow((ml*ml-ms*ms)/mp,2))*sqrt((1-pow((ml+ms)/mp,2))*(1-pow((ml-ms)/mp,2)));

}


double MesonBr(double mp, double ms, double ml,double fpi,double vud){
	double ans;
	if(ms+ml > mp ){
		ans = 0;
	} else {
		ans = Meson2bodWidth(mp,ms,ml,fpi,vud)/Meson2bodWidth(mp,0.0,ml,fpi,vud);
	}

	return ans;
}

double MesonBrPi(double ms, double ml){
	double mp= MPION;
	double fpi=0.1307;
	double vud = 0.97425;
	double br = 1.0;
//		br = (tk*tk*(tm*tm+tl*tl)-pow(tm*tm-tl*tl,2))*sqrt(pow(tk*tk-tm*tm-tl*tl,2)-4*tm*tm*tl*tl)/(tl*tl*pow(tk*tk-tl*tl,2));
		br =  MesonBr(mp,ms,ml,fpi,vud);
	return br;

}

double MesonBrKa(double ms, double ml){
	double mk= MKAON;
	double fka=0.1561;
	double vus = 0.2252;
	double br =1.0;

	//	br = (tk*tk*(tm*tm+tl*tl)-pow(tm*tm-tl*tl,2))*sqrt(pow(tk*tk-tm*tm-tl*tl,2)-4*tm*tm*tl*tl)/(tl*tl*pow(tk*tk-tl*tl,2));

		br = MesonBr(mk,ms,ml,fka,vus);
	return br;

}



double SBNDmodificer2Icarus(double x){
	double ans = 0.0;
	if (x>=3){ ans = 0.0204;
	} else if( 0<=x<3){
	ans = 0.016405470710297916 + 0.05819537343136057*x - 0.24892725116139994*pow(x,2) + 0.34626301711344193*pow(x,3) - 0.16290846186329652*pow(x,4) + 0.00601693077650904*pow(x,5) + 
	   0.013150768041933507*pow(x,6) - 0.0023474486724661876*pow(x,7);
	}
	return ans;
}

double heaviside(double x){
	double ans = 0.0;
	if(x > 0.0){ ans = 1.0;}
	return ans;
}



// NEUTRINO FLUX for any detector

struct Sflux flux_neutrino(double E, double ms, int DET){

	//{ MuPi,EPi,MuKa,EKa, MuKaOther, EKa3}
	struct Sflux ans;
	ans.MuPi=0;ans.MuKa=0;ans.EPi=0;ans.EKa=0;
	ans.MuKaOther=0;
	ans.EKa = 0;

	double fluxDetectorModifierPi = 1.0;
	double fluxDetectorModifierKa = 1.0;
	double micro_calib = 541.0*541.0/(470.0*470.0);

	double E_eff = E; 

	switch (DET) {
  	  case DET_SBND:
	    fluxDetectorModifierPi = micro_calib* 0.61/SBNDmodificer2Icarus(E_eff); 
	    fluxDetectorModifierKa = micro_calib* 0.61/0.0204;
	    ans.MuPi = fluxDetectorModifierPi*fluxPionUu(E_eff);
	    ans.EPi  = fluxDetectorModifierPi*fluxPionUe(E_eff);
	    ans.MuKa = fluxDetectorModifierKa*fluxKaonUu(E_eff);
	    ans.EKa  = fluxDetectorModifierKa*fluxKaonUe(E_eff);
	    ans.MuKaOther = fluxDetectorModifierKa*fluxKaon2Pion(E_eff);
	    break;
	  case DET_MUBOONE:
	    ans.MuPi = micro_calib*fluxPionUu(E_eff);
	    ans.EPi  =  micro_calib*fluxPionUe(E_eff);
	    ans.MuKa =  micro_calib*fluxKaonUu(E_eff);
	    ans.EKa  =  micro_calib*fluxKaonUe(E_eff);
	    ans.MuKaOther =  micro_calib*fluxKaon2Pion(E_eff);
	    break;
	  case DET_ICARUS:
	    fluxDetectorModifierPi =  micro_calib*0.61;
	    fluxDetectorModifierKa =  micro_calib*0.61;
	    ans.MuPi = fluxDetectorModifierPi*fluxPionUu(E_eff);
	    ans.EPi  = fluxDetectorModifierPi*fluxPionUe(E_eff);
	    ans.MuKa = fluxDetectorModifierKa*fluxKaonUu(E_eff);
	    ans.EKa  = fluxDetectorModifierKa*fluxKaonUe(E_eff);
	    ans.MuKaOther = fluxDetectorModifierKa*fluxKaon2Pion(E_eff);
	    break;
	  case DET_PS191:
	    ans.MuPi = fluxPS191PionUu(E_eff);
	    ans.EPi  = fluxPS191PionUe(E_eff);
	    ans.MuKa = fluxPS191KaonUu(E_eff);
	    ans.EKa  = fluxPS191KaonUe(E_eff);
	    ans.EKa3 = fluxPS191Kaon3Ue(E_eff);
	    break;

	}

if(E_eff<=2*ms)
{
	E_eff = E - ms;	
	switch (DET) {
  	  case DET_SBND:
	    fluxDetectorModifierPi = micro_calib* 0.61/SBNDmodificer2Icarus(E_eff); 
	    fluxDetectorModifierKa = micro_calib* 0.61/0.0204;
	    ans.MuPi += fluxDetectorModifierPi*fluxPionUu(E_eff);
	    ans.EPi  += fluxDetectorModifierPi*fluxPionUe(E_eff);
	    ans.MuKa += fluxDetectorModifierKa*fluxKaonUu(E_eff);
	    ans.EKa  += fluxDetectorModifierKa*fluxKaonUe(E_eff);
	    ans.MuKaOther += fluxDetectorModifierKa*fluxKaon2Pion(E_eff);
	    break;
	  case DET_MUBOONE:
	    ans.MuPi += micro_calib*fluxPionUu(E_eff);
	    ans.EPi  +=  micro_calib*fluxPionUe(E_eff);
	    ans.MuKa +=  micro_calib*fluxKaonUu(E_eff);
	    ans.EKa  +=  micro_calib*fluxKaonUe(E_eff);
	    ans.MuKaOther +=  micro_calib*fluxKaon2Pion(E_eff);
	    break;
	  case DET_ICARUS:
	    fluxDetectorModifierPi =  micro_calib*0.61;
	    fluxDetectorModifierKa =  micro_calib*0.61;
	    ans.MuPi += fluxDetectorModifierPi*fluxPionUu(E_eff);
	    ans.EPi  += fluxDetectorModifierPi*fluxPionUe(E_eff);
	    ans.MuKa += fluxDetectorModifierKa*fluxKaonUu(E_eff);
	    ans.EKa  += fluxDetectorModifierKa*fluxKaonUe(E_eff);
	    ans.MuKaOther += fluxDetectorModifierKa*fluxKaon2Pion(E_eff);
	    break;
	  case DET_PS191:
	    ans.MuPi += fluxPS191PionUu(E_eff);
	    ans.EPi  += fluxPS191PionUe(E_eff);
	    ans.MuKa += fluxPS191KaonUu(E_eff);
	    ans.EKa  += fluxPS191KaonUe(E_eff);
	    ans.EKa3 += fluxPS191Kaon3Ue(E_eff);
	    break;

	}
}





return ans;

}

Sflux::Sflux()
{
	
MuPi=0.0;
MuKa=0.0; 
EPi=0.0; 
EKa=0.0;
EKa3=0.0;
MuKaOther = 0.0;

}

// STERILE FLUX from neutrinos flux

struct Sflux flux_sterile(double E, double ms, double Ue, double Um, double Ut, int Det){



	//{ MuPi,EPi,MuKa,EKa}
	struct Sflux nuFlux = flux_neutrino(E,ms,Det);

	struct Sflux sFlux;
	sFlux.MuPi=0.0;sFlux.MuKa=0.0; sFlux.EPi=0.0; sFlux.EKa=0.0;sFlux.EKa3=0.0;sFlux.MuKaOther = 0.0;

	sFlux.MuPi = Um*Um* the_shrock_factor(MPION,ms,MMU) * nuFlux.MuPi;
	sFlux.MuKa = Um*Um* the_shrock_factor(MKAON,ms,MMU) * nuFlux.MuKa;
	
	sFlux.EPi  = Ue*Ue*the_shrock_factor(MPION,ms,ME) * nuFlux.EPi;
	sFlux.EKa  = Ue*Ue*the_shrock_factor(MKAON,ms,ME) * nuFlux.EKa;

	sFlux.MuKaOther = the_shrock_factor(MKAON,ms,MMU)* Um*Um*nuFlux.MuKaOther;
	
	sFlux.EKa3 = Ue*Ue*nuFlux.EKa3;
	
	return sFlux;

}



double flux (double E, double ms, double Ue, double Uu, double Ut, int DET){
//Returns a total of the sterile fluxes
	struct Sflux s_flux;

	s_flux = flux_sterile(E,ms,Ue,Uu,Ut,DET);
	
	if (E<ms)
	{
		return 0.0;
	}
	else 
	{
		return s_flux.total();
	}

}


int plot_flux(double ms,int detector_flag)
{
	double E=0.0;
	double Uu = 1.0;
	struct Sflux s_flux;
	
	for(E=std::max(0.001,ms); E<10.0; E+=0.05)
	{

	s_flux = flux_sterile(E,ms,0.0,1.0,0.0,detector_flag);

		std::cout<<E<<" "<<s_flux.MuPi<<" "<<s_flux.MuKa<<" "<<s_flux.EPi<<" "<<s_flux.EKa<<"  "<<s_flux.MuPi+s_flux.MuKa<<" "<<s_flux.EPi+s_flux.EKa<<std::endl;
	
	}

return 0;
}


int plot_neutrino_flux(int detector_flag)
{
	double E=0.0;
	struct Sflux s_flux;
	
	for(E=0.001; E<30.0; E+=0.05)
	{

	s_flux = flux_neutrino(E,0.0,detector_flag);

		std::cout<<E<<" "<<s_flux.MuPi<<" "<<s_flux.MuKa<<" "<<s_flux.EPi<<" "<<s_flux.EKa<<"  "<<s_flux.MuPi+s_flux.MuKa<<" "<<s_flux.EPi+s_flux.EKa<<std::endl;
	
	}

return 0;
}





/*****************************************************************
 *
 *
 *		Fluxes from here on down
 *
 *
 *****************************************************************
 */




double fluxPionUu(double E){


	std::vector<double > list = {8.80937e-11,2.16774e-10,2.58497e-10,2.93717e-10,3.42692e-10,3.78027e-10,3.99828e-10,4.14408e-10,4.26851e-10,4.40354e-10,4.42409e-10,4.42402e-10,4.43775e-10,4.27484e-10,4.17603e-10,4.05417e-10,3.91751e-10,3.73275e-10,3.60131e-10,3.36791e-10,3.18417e-10,2.9871e-10,2.77183e-10,2.53626e-10,2.29197e-10,2.08416e-10,1.85141e-10,1.63954e-10,1.42724e-10,1.24244e-10,1.06983e-10,9.11215e-11,7.79748e-11,6.52855e-11,5.63911e-11,4.72142e-11,3.97158e-11,3.29946e-11,2.73682e-11,2.32741e-11,1.95474e-11,1.652e-11,1.37243e-11,1.12254e-11,8.9555e-12,7.30213e-12,6.11381e-12,5.0476e-12,4.23935e-12,3.44593e-12,2.77062e-12,2.20349e-12,1.82489e-12,1.50195e-12,1.20573e-12,8.99591e-13,6.97838e-13,4.99985e-13,4.11507e-13,3.22215e-13,2.30859e-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	std::vector<double> list_numubar = {1.44478e-10,9.3512e-11,4.81888e-11,3.61648e-11,3.00803e-11,2.72913e-11,2.63065e-11,2.51335e-11,2.29673e-11,2.07247e-11,1.88863e-11,1.74325e-11,1.61107e-11,1.48965e-11,1.37439e-11,1.26294e-11,1.19351e-11,1.10927e-11,1.03375e-11,9.53898e-12,8.80501e-12,8.10384e-12,7.55345e-12,6.99431e-12,6.34315e-12,5.8051e-12,5.21299e-12,4.65819e-12,4.15301e-12,3.80025e-12,3.37175e-12,2.95246e-12,2.5896e-12,2.28185e-12,2.03054e-12,1.77678e-12,1.54417e-12,1.34511e-12,1.18442e-12,1.0235e-12,8.55044e-13,7.44532e-13,6.37081e-13,5.40691e-13,4.54156e-13,3.72208e-13,3.09729e-13,2.55847e-13,2.14839e-13,1.7891e-13,1.47472e-13,1.20211e-13,9.80139e-14,7.84161e-14,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

	double ans = 0;

	//if(E <= 5.0){
	//	ans = list[floor(E/0.05)];
	//}
	if(E <= 5.0){
		ans = list[floor(E/0.05)] + ((E - 0.05*floor(E/0.05))/0.05)*(list[floor(E/0.05)+1]-list[floor(E/0.05)]);
		ans += list_numubar[floor(E/0.05)] + ((E - 0.05*floor(E/0.05))/0.05)*(list_numubar[floor(E/0.05)+1]-list_numubar[floor(E/0.05)]);
	}


	return ans;

}
double fluxKaonUu(double E){
	std::vector<double > list = {2.38216e-13,5.91502e-13,1.13546e-12,1.09411e-12,1.0296e-11,7.16169e-13,9.71896e-13,1.21063e-12,1.4572e-12,1.60495e-12,1.74035e-12,1.91382e-12,2.07206e-12,2.16441e-12,2.25034e-12,2.38755e-12,2.46691e-12,2.48616e-12,2.57682e-12,2.61316e-12,2.62537e-12,2.62942e-12,2.61711e-12,2.72102e-12,2.67891e-12,2.69561e-12,2.81138e-12,2.78084e-12,2.78079e-12,2.78075e-12,2.84645e-12,2.8464e-12,2.798e-12,2.89101e-12,2.90904e-12,2.92718e-12,2.98702e-12,2.92253e-12,2.9316e-12,2.97758e-12,2.93151e-12,2.93603e-12,2.87265e-12,2.96352e-12,2.9359e-12,2.92672e-12,2.94497e-12,2.89488e-12,2.96329e-12,2.89029e-12,2.93106e-12,2.92646e-12,2.92186e-12,2.89012e-12,2.77102e-12,2.81448e-12,2.85418e-12,2.79692e-12,2.75364e-12,2.71525e-12,2.68158e-12,2.74494e-12,2.75346e-12,2.6277e-12,2.63996e-12,2.6317e-12,2.51934e-12,2.39678e-12,2.34504e-12,2.37811e-12,2.24837e-12,2.19983e-12,2.18613e-12,2.11574e-12,2.10584e-12,2.02537e-12,1.99092e-12,1.89408e-12,1.81038e-12,1.75208e-12,1.69566e-12,1.64106e-12,1.57343e-12,1.49456e-12,1.45321e-12,1.4285e-12,1.37605e-12,1.23194e-12,1.16837e-12,1.12898e-12,1.07574e-12,1.03786e-12,9.63066e-13,9.21942e-13,8.55503e-13,7.87688e-13,7.2187e-13,6.87824e-13,6.43248e-13,5.55612e-13};

	double ans = 0;

//	if(E <= 5.0){
//		ans = list[floor(E/0.05)];
//	}
	if(E <= 5.0){
		ans = list[floor(E/0.05)] + ((E - 0.05*floor(E/0.05))/0.05)*(list[floor(E/0.05)+1]-list[floor(E/0.05)]);
	}


	return ans;

}

double fluxPionUe(double E){
	std::vector<double > list = {4.57867e-13,1.46464e-12,2.00513e-12,2.13872e-12,2.18093e-12,2.11768e-12,2.03295e-12,1.92689e-12,1.80443e-12,1.68523e-12,1.56969e-12,1.46501e-12,1.36732e-12,1.36732e-12,1.26591e-12,1.16887e-12,1.0829e-12,1.00124e-12,9.26359e-13,8.44518e-13,7.79785e-13,7.20981e-13,6.56844e-13,5.9521e-13,5.53286e-13,5.0644e-13,4.59843e-13,4.20344e-13,3.83981e-13,3.47483e-13,3.15512e-13,2.87831e-13,2.61523e-13,2.37461e-13,2.14314e-13,1.93294e-13,1.7551e-13,1.7551e-13,1.56921e-13,1.41435e-13,1.26625e-13,1.12759e-13,1.01223e-13,9.08677e-14,8.07005e-14,7.33737e-14,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

	double ans = 0;

//	if(E <= 5.0){
//		ans = list[floor(E/0.05)];
//	}
	if(E <= 5.0){
		ans = list[floor(E/0.05)] + ((E - 0.05*floor(E/0.05))/0.05)*(list[floor(E/0.05)+1]-list[floor(E/0.05)]);
	}
	//return ans;
	double anti_componant = 1.26/2.1;
	double BrPiMu2PiE = 0.000128;
	return fluxPionUu(E)*BrPiMu2PiE*(1+anti_componant);
}

double fluxKaonUe(double E){
	std::vector<double > list = {9.16296e-14,1.63254e-13,1.80374e-12,1.02946e-12,3.94502e-13,4.79124e-13,5.57993e-13,6.33911e-13,6.98273e-13,7.1397e-13,7.44861e-13,7.45929e-13,7.60142e-13,7.58174e-13,7.46632e-13,7.52735e-13,7.41773e-13,7.30971e-13,7.08345e-13,6.81372e-13,6.82349e-13,6.48052e-13,6.51599e-13,6.16362e-13,6.10655e-13,5.78796e-13,5.59002e-13,5.30549e-13,5.37043e-13,5.37043e-13,5.01567e-13,4.8539e-13,4.66907e-13,4.42547e-13,4.23986e-13,4.04299e-13,3.70068e-13,3.55976e-13,3.46349e-13,3.31822e-13,3.08864e-13,2.94919e-13,2.7433e-13,2.56208e-13,2.48945e-13,2.30635e-13,2.15834e-13,2.04162e-13,1.90164e-13,1.7976e-13,1.69129e-13,1.54601e-13,1.48017e-13,1.48017e-13,1.37039e-13,1.2713e-13,1.17307e-13,1.1171e-13,1.01296e-13,9.84244e-14,8.74116e-14,8.27401e-14,7.59381e-14,7.04947e-14,6.88404e-14,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

	double ans = 0;

//	if(E <= 5.0){
//		ans = list[floor(E/0.05)];
//	}
	if(E <= 5.0){
		ans = list[floor(E/0.05)] + ((E - 0.05*floor(E/0.05))/0.05)*(list[floor(E/0.05)+1]-list[floor(E/0.05)]);
	}
	
	//return ans;
	return fluxKaonUu(E)*1.582e-5+fluxKaon2Pion(E)*1.582e-5;

}

double fluxKaon2Pion(double E){
	std::vector<double > list = {1.16605e-12,1.61427e-12,1.98919e-12,2.22216e-12,2.38442e-12,2.2226e-12,2.01049e-12,1.88619e-12,1.75336e-12,1.54891e-12,1.36936e-12,1.17981e-12,9.845e-13,8.9523e-13,7.88909e-13,6.43428e-13,5.5417e-13,4.68804e-13,4.01626e-13,3.2866e-13,2.74685e-13,2.25461e-13,1.94016e-13,1.65213e-13,1.30808e-13,1.15445e-13,1.02024e-13,7.91002e-14,6.71874e-14,6.13261e-14};
	double ans = 0;

	std::vector<double>  list_other ={5.3891e-12, 3.33041e-12, 3.18926e-12, 2.72131e-12, 	 2.69798e-12, 2.37135e-12, 2.09792e-12, 1.94336e-12, 		  1.76635e-12, 1.55192e-12, 1.31918e-12, 1.05777e-12, 		   8.43636e-13, 7.9545e-13, 6.42827e-13, 5.37716e-13, 		    4.58508e-13, 3.44023e-13, 3.21407e-13, 2.85287e-13, 		     2.43038e-13, 2.11014e-13, 1.85562e-13, 1.50143e-13, 		      1.23105e-13, 1.06821e-13, 9.55103e-14, 8.55764e-14, 		       7.72768e-14, 7.16329e-14};

	std::vector<double> list_other_bar = {7.96627e-12,5.08321e-12,3.20823e-12,2.46661e-12,2.15058e-12,1.78002e-12,1.52325e-12,1.18709e-12,1.00843e-12,8.0951e-13,6.85206e-13,5.8031e-13,4.55557e-13,4.0089e-13,3.34453e-13,2.67411e-13,2.22954e-13,2.04164e-13,1.94142e-13,1.40755e-13,9.77464e-14,7.8789e-14,6.37924e-14,5.28305e-14};

	double other_modifier = 1.0;	

	if(E <= 1.5){
		ans = list[floor(E/0.05)] + ((E - 0.05*floor(E/0.05))/0.05)*(list[floor(E/0.05)+1]-list[floor(E/0.05)]);
	ans =ans+ list_other[floor(E/0.05)] + ((E - 0.05*floor(E/0.05))/0.05)*(list_other[floor(E/0.05)+1]-list_other[floor(E/0.05)]);

	}
	if(E <= 1.2){
	ans =ans+ list_other_bar[floor(E/0.05)] + ((E - 0.05*floor(E/0.05))/0.05)*(list_other_bar[floor(E/0.05)+1]-list_other_bar[floor(E/0.05)]);

	}


	
	return other_modifier* ans;
}



double fluxPS191KaonUu(double E){
	std::vector<double > list ={1.18088e-12,2.71695e-12,4.69988e-12,7.10858e-12,9.92195e-12,1.30636e-11,1.672e-11,2.11675e-11,2.51925e-11,2.8478e-11,3.4222e-11,4.5293e-11,5.62986e-11,6.006e-11,6.80132e-11,8.94188e-11,1.10231e-10,1.26969e-10,1.39033e-10,1.45425e-10,1.52675e-10,1.67845e-10,1.84106e-10,1.96521e-10,2.07493e-10,2.17815e-10,2.27829e-10,2.38448e-10,2.48412e-10,2.55201e-10,2.59999e-10,2.65342e-10,2.68234e-10,2.63739e-10,2.56124e-10,2.50805e-10,2.47534e-10,2.43362e-10,2.34888e-10,2.26008e-10,2.20725e-10,2.15793e-10,2.07557e-10,1.96332e-10,1.88852e-10,1.83845e-10,1.78077e-10,1.70208e-10,1.61873e-10,1.53465e-10,1.45494e-10,1.40445e-10,1.36316e-10,1.31142e-10,1.22085e-10,1.11937e-10,1.0589e-10,1.01482e-10,9.65057e-11,9.14065e-11,8.59886e-11,8.06844e-11,7.6193e-11,7.24273e-11,6.89637e-11,6.52994e-11,6.17853e-11,5.84863e-11,5.55194e-11,5.27591e-11,5.0045e-11,4.74867e-11,4.5322e-11,4.33352e-11,4.11151e-11,3.85084e-11,3.60494e-11,3.39285e-11,3.19357e-11,2.99884e-11,2.81466e-11,2.64292e-11,2.4827e-11,2.33642e-11,2.1997e-11,2.06742e-11,1.93566e-11,1.81498e-11,1.71304e-11,1.62148e-11,1.53614e-11,1.44921e-11,1.36623e-11,1.28722e-11,1.21245e-11,1.14314e-11,1.08047e-11,1.02478e-11,9.7463e-12,0.};


	std::vector<double > list_minus = {0.,0.,0.,7.31511e-12,9.53952e-12,1.13126e-11,1.33836e-11,1.71217e-11,2.10921e-11,2.38738e-11,2.65651e-11,3.01052e-11,3.38353e-11,3.78286e-11,4.00734e-11,3.82019e-11,3.51174e-11,3.33104e-11,3.18021e-11,2.97793e-11,2.76765e-11,2.58836e-11,2.4486e-11,2.36998e-11,2.29948e-11,2.17936e-11,2.06746e-11,2.02171e-11,2.00364e-11,1.96999e-11,1.92281e-11,1.88803e-11,1.87194e-11,1.87658e-11,1.83809e-11,1.74946e-11,1.67965e-11,1.66028e-11,1.64888e-11,1.62384e-11,1.58087e-11,1.50906e-11,1.46014e-11,1.43627e-11,1.40264e-11,1.34668e-11,1.30882e-11,1.31169e-11,1.3001e-11,1.21376e-11,1.12324e-11,1.07122e-11,1.02273e-11,9.71719e-12,9.30201e-12,9.02402e-12,8.77449e-12,8.50012e-12,8.12766e-12,7.54146e-12,7.02089e-12,6.74015e-12,6.46679e-12,6.0365e-12,5.62787e-12,5.34862e-12,5.07595e-12,4.75387e-12,4.45682e-12,4.26676e-12,4.11146e-12,3.95246e-12,3.79475e-12,3.63986e-12,3.4628e-12,3.23578e-12,3.02437e-12,2.8537e-12,2.69316e-12,2.51889e-12,2.35267e-12,2.23367e-12,2.13241e-12,2.01394e-12,1.89175e-12,1.76362e-12,1.63511e-12,1.51341e-12,1.39952e-12,1.29537e-12,1.20878e-12,1.13904e-12,1.07938e-12,1.02172e-12,9.7075e-13,9.2837e-13,8.90666e-13,8.55134e-13,8.1927e-13,0.};	
	double ans = 0;

	if(E <= 9.8){	
		ans += list[floor(E/0.1)] + ((E - 0.1*floor(E/0.1))/0.1)*(list[floor(E/0.1)+1]-list[floor(E/0.1)]);
		ans += list_minus[floor(E/0.1)] + ((E - 0.1*floor(E/0.1))/0.1)*(list_minus[floor(E/0.1)+1]-list_minus[floor(E/0.1)]);
	}	
	if(E>9.8){
		ans = 1.04599e-11*pow(exp(9.801 - E),0.55) + 8.1927e-13*pow(exp(9.801 -E),0.55);
	}
	
	return 2.0*ans;

}

double fluxPS191Kaon3Ue(double E){
	std::vector<double > list= {3.20399e-13,2.39743e-12,4.7909e-12,7.25164e-12,9.47157e-12,1.06564e-11,1.20192e-11,1.49636e-11,1.88061e-11,2.16746e-11,2.35209e-11,2.50398e-11,2.63505e-11,2.73444e-11,2.79984e-11,2.83419e-11,2.84079e-11,2.81858e-11,2.76352e-11,2.65755e-11,2.52991e-11,2.4007e-11,2.27899e-11,2.16352e-11,2.05345e-11,1.94667e-11,1.84156e-11,1.72917e-11,1.59833e-11,1.46906e-11,1.35007e-11,1.23911e-11,1.137e-11,1.04608e-11,9.68724e-12,8.99011e-12,8.32546e-12,7.68859e-12,7.07432e-12,6.494e-12,5.98231e-12,5.56865e-12,5.18307e-12,4.7926e-12,4.32331e-12,3.89519e-12,3.58076e-12,3.32899e-12,3.12576e-12,2.95291e-12,2.75684e-12,2.47956e-12,2.14132e-12,1.92225e-12,1.76547e-12,1.63422e-12,1.5117e-12,1.38714e-12,1.26065e-12,1.13612e-12,1.03098e-12,9.48106e-13,8.6527e-13,7.71141e-13,6.86132e-13,6.15707e-13,5.51229e-13,4.85798e-13,4.28332e-13,3.84712e-13,3.46913e-13,3.0732e-13,2.68774e-13,2.39447e-13,2.17497e-13,1.98877e-13,1.81207e-13,1.60447e-13,1.32788e-13,1.09931e-13,9.19813e-14,7.69918e-14,6.02048e-14,4.75843e-14,4.23016e-14,3.67645e-14,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};	
	double ans = 0;

	if(E <= 10){
	ans = list[floor(E/0.1)] + ((E - 0.1*floor(E/0.1))/0.1)*(list[floor(E/0.1)+1]-list[floor(E/0.1)]);

	}
 
 	return ans;

}

double fluxPS191KaonUe(double E){
	std::vector<double > list= {3.20399e-13,2.39743e-12,4.7909e-12,7.25164e-12,9.47157e-12,1.06564e-11,1.20192e-11,1.49636e-11,1.88061e-11,2.16746e-11,2.35209e-11,2.50398e-11,2.63505e-11,2.73444e-11,2.79984e-11,2.83419e-11,2.84079e-11,2.81858e-11,2.76352e-11,2.65755e-11,2.52991e-11,2.4007e-11,2.27899e-11,2.16352e-11,2.05345e-11,1.94667e-11,1.84156e-11,1.72917e-11,1.59833e-11,1.46906e-11,1.35007e-11,1.23911e-11,1.137e-11,1.04608e-11,9.68724e-12,8.99011e-12,8.32546e-12,7.68859e-12,7.07432e-12,6.494e-12,5.98231e-12,5.56865e-12,5.18307e-12,4.7926e-12,4.32331e-12,3.89519e-12,3.58076e-12,3.32899e-12,3.12576e-12,2.95291e-12,2.75684e-12,2.47956e-12,2.14132e-12,1.92225e-12,1.76547e-12,1.63422e-12,1.5117e-12,1.38714e-12,1.26065e-12,1.13612e-12,1.03098e-12,9.48106e-13,8.6527e-13,7.71141e-13,6.86132e-13,6.15707e-13,5.51229e-13,4.85798e-13,4.28332e-13,3.84712e-13,3.46913e-13,3.0732e-13,2.68774e-13,2.39447e-13,2.17497e-13,1.98877e-13,1.81207e-13,1.60447e-13,1.32788e-13,1.09931e-13,9.19813e-14,7.69918e-14,6.02048e-14,4.75843e-14,4.23016e-14,3.67645e-14,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};	
	double ans = 0;

	if(E <= 10){
	ans = list[floor(E/0.1)] + ((E - 0.1*floor(E/0.1))/0.1)*(list[floor(E/0.1)+1]-list[floor(E/0.1)]);

	}
 
	return fluxPS191KaonUu(E)*1.582e-5;

 	//return ans;

}
double fluxPS191PionUu(double E){
	std::vector<double > list= {0.,3.01813e-9,8.21717e-9,1.44296e-8,1.89444e-8,1.88089e-8,1.68206e-8,1.48104e-8,1.24935e-8,1.02165e-8,8.17802e-9,6.7517e-9,5.62809e-9,4.62799e-9,3.80343e-9,3.15837e-9,2.63568e-9,2.17847e-9,1.8105e-9,1.53456e-9,1.31633e-9,1.1302e-9,9.67116e-10,8.14291e-10,6.8331e-10,5.77819e-10,4.88525e-10,4.08711e-10,3.43279e-10,2.93004e-10,2.53207e-10,2.20651e-10,1.92521e-10,1.6536e-10,1.40927e-10,1.19473e-10,1.01657e-10,8.83538e-11,7.68971e-11,6.42521e-11,5.31071e-11,4.51042e-11,3.84334e-11,3.18176e-11,2.60294e-11,2.13373e-11,1.74564e-11,1.42532e-11,1.18602e-11,1.0198e-11,8.719e-12,6.89608e-12,5.32847e-12,4.39281e-12,3.68198e-12,2.9002e-12,2.21693e-12,1.7349e-12,1.369e-12,1.05556e-12,8.06376e-13,6.2935e-13,5.00522e-13,3.94313e-13,3.18292e-13,2.77855e-13,2.5589e-13,2.37764e-13,2.20572e-13,1.917e-13,1.64633e-13,1.52147e-13,1.42394e-13,1.23519e-13,1.04923e-13,9.78768e-14,9.25749e-14,7.57536e-14,5.37821e-14,4.81051e-14,4.81043e-14,4.7226e-14,3.89169e-14,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
		double ans = 0;
	std::vector<double > list_minus ={1.73154e-10,1.66779e-10,1.8816e-10,2.25594e-10,2.66161e-10,2.96472e-10,3.20262e-10,3.36396e-10,3.39138e-10,3.26686e-10,3.06219e-10,2.84081e-10,2.60646e-10,2.37553e-10,2.14823e-10,1.92318e-10,1.70595e-10,1.50059e-10,1.33306e-10,1.21367e-10,1.10675e-10,9.95714e-11,8.81145e-11,7.62954e-11,6.56608e-11,5.69506e-11,4.99673e-11,4.45088e-11,3.91092e-11,3.33999e-11,2.87765e-11,2.52327e-11,2.14935e-11,1.80462e-11,1.569e-11,1.4271e-11,1.27289e-11,1.10924e-11,9.41762e-12,7.79467e-12,6.66039e-12,5.84288e-12,5.10528e-12,4.35203e-12,3.70708e-12,3.16977e-12,2.65538e-12,2.11244e-12,1.64043e-12,1.28257e-12,1.00942e-12,8.04481e-13,6.4273e-13,4.94098e-13,3.78826e-13,2.9998e-13,2.3771e-13,1.72235e-13,1.14277e-13,6.57257e-14,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
	if(E <= 10){
	ans = list[floor(E/0.1)] + ((E - 0.1*floor(E/0.1))/0.1)*(list[floor(E/0.1)+1]-list[floor(E/0.1)]);
     	ans+=list_minus[floor(E/0.1)] + ((E - 0.1*floor(E/0.1))/0.1)*(list_minus[floor(E/0.1)+1]-list_minus[floor(E/0.1)]);

	}
	return 2*ans;

}
double fluxPS191PionUe(double E){
	double mp= MPION;
	double mk= MKAON;
	double mmu= MMU;
	double me= ME;
	double fpi=0.1307;
	double vud = 0.97425;
	double gf = 0.00001166;
	double fka=0.1561;
	double vus = 0.2252;

	double BRpimu2pie = Meson2bodWidth(mp,0.0, me, fpi, vud)/Meson2bodWidth(mp,0.0,mmu,fpi,vud);

	return fluxPS191PionUu(E)*BRpimu2pie;

}



double geometry_factor(double ms)
{

	double ans = 0;
	std::vector<double > list ={0.987839,0.990401,0.99062,1.00567,1.00349,1.02236,1.00589,1.01145,1.01451,1.01134,1.03338,1.03098,1.06485,1.05574,1.07592,1.09097,1.10035,1.10548,1.12435,1.14322,1.15522,1.17627,1.20147,1.22906,1.25404,1.25949,1.29783,1.32821,1.35204,1.38405,1.41405,1.45277,1.50486,1.53447,1.58469,1.62914,1.6593,1.7103,1.76096,1.85417,1.89802,1.96913,2.02961,2.08519,2.17441,2.29079,2.39796,2.45942,2.59757,2.69895,2.83361,2.9868,3.11764,3.29548,3.47699,3.68085,3.89071,4.16863,4.37282,4.67866,5.01358,5.36606,5.7644,6.23658,6.7338,7.34108,8.00671,8.8077,9.64725,10.704,11.9051,13.3668,15.1648,17.2391,20.0175,23.4142,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387,26.6387};

	ans = list[floor(ms/0.005)] + ((ms - 0.005*floor(ms/0.005))/0.005)*(list[floor(ms/0.005)+1]-list[floor(ms/0.005)]);

	return ans;
}
