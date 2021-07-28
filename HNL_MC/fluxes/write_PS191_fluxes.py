import numpy as np
from scipy import interpolate

from matplotlib import rc, rcParams
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import pathos.multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

from functools import partial

from particle import *
from particle import literals as lp


area = 6e2*3e2 # cm^2
norm = 1/0.89e20/4000
norm2 = 1/0.89e19/area/0.2

print(norm/norm2)

PATH="."
fpi = np.genfromtxt(f"{PATH}/ps191/pi_Uu.dat")
fKp = np.genfromtxt(f"{PATH}/ps191/Kplus_Uu.dat")
fKm = np.genfromtxt(f"{PATH}/ps191/Kminus_Uu.dat")

ep = np.linspace(0,len(fKp)*0.1, len(fKp))

np.savetxt(f"{PATH}/ps191/nu(mu)_pi+.dat",np.array([ep,fpi]).T)
np.savetxt(f"{PATH}/ps191/nu(mu)_K+.dat",np.array([ep,fKp]).T)


my_e, my_numu = np.genfromtxt(f"{PATH}/ps191/numu_flux.dat", unpack=True)

# plt.plot(ep, fpi, label='fpi')
# plt.plot(ep, fKp, label='fKp')
# plt.plot(ep, fKm, label='fKm')

plt.plot(ep, fpi+fKp+fKm, label='tot')
plt.plot(ep, fpi, label='fpi')
plt.plot(ep, fKp, label='fKp')
plt.plot(ep, fKm, label='fKm')


print(np.sum(fpi+fKp+fKm)/norm)
print(np.sum(fpi+fKp+fKm)/norm2)


plt.plot(my_e, my_numu*norm)
# plt.plot(my_e, my_numu*norm2)
# plt.plot(my_e, my_numu*norm)


plt.legend(loc='upper right', frameon=False)
plt.yscale("log")



# dE = 0.1
# index = int(np.floor(E/dE))
# print(index)
# ans = f[index] + ((E - dE*index)/dE)*(f[index+1]-f[index])
# ans = list[floor(E/0.1)] + ((E - 0.1*floor(E/0.1))/0.1)*(list[floor(E/0.1)+1]-list[floor(E/0.1)]);

plt.savefig("../../plots/PS191_fluxes.pdf")

