import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D


#########################################################
# SOME OLD CRAP
#########################################################

def histogram_coh_dif(plotname, DATACOH, DATADIF, TMIN, TMAX,  XLABEL, TITLE, nbins):
	
	fsize = 11
	
	x1 = DATACOH[0]
	w1 = DATACOH[1]
	I1 = DATACOH[2]
	
	x2 = DATADIF[0]
	w2 = DATADIF[1]
	I2 = DATADIF[2]

	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	hist1 = np.histogram(x1, weights=w1, bins=nbins, density = False, range = (TMIN,TMAX) )
	hist2 = np.histogram(x2, weights=w2, bins=nbins, density = False, range = (TMIN,TMAX) )

	ans0 = hist1[1][:nbins]
	ans1 = hist1[0]#/(ans0[1]-ans0[0])
	ans2 = hist2[0]#/(ans0[1]-ans0[0])

	# comb1 = (ans1 + 0*8*case2[2]/case1[2] * ans2)
	# comb2 = (ans3*0 + 8*case4[2]/case3[2] * ans4)

	comb1 = ans1
	comb2 = ans2

	comb1 = comb1/np.sum(comb1) #* I1  #/(ans0[1]-ans0[0])
	comb2 = comb2/np.sum(comb2) #* I2*8/I1# /(ans0[1]-ans0[0])

	ax.bar(ans0,comb1, ans0[1]-ans0[0], label=r"coh",\
			ec=None, fc='indigo', alpha=0.4, align='edge', lw = 0.0)	
	ax.bar(ans0,comb2, ans0[1]-ans0[0], label=r"dif",\
			ec=None, fc='lightblue', alpha=0.4, align='edge', lw = 0.0)	



	ax.step(np.append(ans0,10e10), np.append(comb1, 0.0), where='post',
				c='indigo', lw = 2.0)
	ax.step(np.append(ans0,10e10), np.append(comb2, 0.0), where='post',
				c='lightblue', lw = 2.0)



	ax.set_title(TITLE, fontsize=fsize)

	ax.annotate(xy=(0.3,0.9), xycoords='axes fraction', s=r"$R = N_d/N_c = $"+str(round(8*I2/I1,5)), fontsize=fsize)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(r"PDF",fontsize=fsize)

	ax.set_xlim(TMIN,TMAX)
	ax.set_ylim(0,ax.get_ylim()[1]*1.5)

	# plt.show()
	plt.savefig(plotname)



def histogram1D(plotname, DATA, TMIN, TMAX,  XLABEL, TITLE, nbins):
	
	fsize = 11
	
	x1 = DATA[0]
	w1 = DATA[1]
	I1 = DATA[2]
	
	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	hist1 = np.histogram(x1, weights=w1, bins=nbins, density = False, range = (TMIN,TMAX) )

	ans0 = hist1[1][:nbins]
	ans1 = hist1[0]#/(ans0[1]-ans0[0])

	# comb1 = (ans1 + 0*8*case2[2]/case1[2] * ans2)
	# comb2 = (ans3*0 + 8*case4[2]/case3[2] * ans4)

	comb1 = ans1

	# comb1 = comb1/np.sum(comb1) #* I1  #/(ans0[1]-ans0[0])

	ax.bar(ans0,comb1, ans0[1]-ans0[0], label=r"PDF",\
			ec=None, fc='indigo', alpha=0.4, align='edge', lw = 0.0)	

	ax.step(np.append(ans0,10e10), np.append(comb1, 0.0), where='post',
				c='indigo', lw = 2.0)

	ax.set_title(TITLE, fontsize=fsize)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(r"PDF",fontsize=fsize)

	ax.set_xlim(TMIN,TMAX)
	ax.set_ylim(0,ax.get_ylim()[1]*1.5)

	# plt.show()
	plt.savefig(plotname)
	plt.close()

def histogram2D(plotname, DATACOHX, DATACOHY, XMIN, XMAX, YMIN, YMAX,  XLABEL,  YLABEL, TITLE, NBINS):
	
	fsize = 9
	
	x1 = DATACOHX[0]
	y1 = DATACOHY[0]
	w1 = DATACOHY[1]
	I1 = DATACOHX[2]

	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	bar = ax.hist2d(x1,y1, bins=NBINS, weights=w1, range=[[XMIN,XMAX],[YMIN,YMAX]],cmap="Blues",normed=True)

	ax.set_title(TITLE, fontsize=fsize)
	cbar_R = fig.colorbar(bar[3],ax=ax)
	cbar_R.ax.set_ylabel(r'a.u.', rotation=90)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(YLABEL,fontsize=fsize)

	# ax.set_xlim(XMIN,XMAX)
	# ax.set_ylim(YMIN,YMAX)

	# plt.show()
	plt.savefig(plotname)
	plt.close()

def scatter2D(plotname, DATACOHX, DATACOHY, XMIN, XMAX, YMIN, YMAX,  XLABEL,  YLABEL, TITLE, NBINS, log_plot_x=False,log_plot_y=False):
	
	fsize = 9
	
	x1 = DATACOHX[0]
	y1 = DATACOHY[0]
	w1 = DATACOHY[1]
	I1 = DATACOHX[2]

	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	bar = ax.scatter(x1,y1, color='purple', s=0.2, edgecolor=None,alpha=0.6)

	ax.set_title(TITLE, fontsize=fsize)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(YLABEL,fontsize=fsize)

	if log_plot_x:
		ax.set_xscale("log")
	if log_plot_y:
		ax.set_yscale("log")
	ax.set_xlim(np.min([x1]),np.max([x1]))
	ax.set_ylim(np.min([y1]),np.max([y1]))


	plt.savefig(plotname)
	plt.close()

def histogram2DLOG(plotname, DATACOHX, DATACOHY, XMIN, XMAX, YMIN, YMAX,  XLABEL,  YLABEL, TITLE, NBINS):
	
	fsize = 11
	
	x1 = DATACOHX[0]
	y1 = DATACOHY[0]
	w1 = DATACOHY[1]
	I1 = DATACOHX[2]

	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)
			
	x1 = np.log10(x1)
	y1 = np.log10(y1)


	bar = ax.hist2d(x1,y1, bins=NBINS, weights=w1, range=[[np.log10(XMIN),np.log10(XMAX)],[np.log10(YMIN),np.log10(YMAX)]],cmap="Blues",normed=True)
	# hist[1][:nbins] = 10**hist[1][:nbins]

	ax.set_title(TITLE, fontsize=fsize)
	cbar_R = fig.colorbar(bar[3],ax=ax)
	cbar_R.ax.set_ylabel(r'a.u.', rotation=90)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(YLABEL,fontsize=fsize)

	# ax.set_xlim(XMIN,XMAX)
	# ax.set_ylim(YMIN,YMAX)

	# ax.set_xscale("log")
	# ax.set_yscale("log")
	# plt.show()
	plt.savefig(plotname)
	plt.close()


def data_plot(plotname, X, BINW, MODEL, DATA, ERRORLOW, ERRORUP, XMIN, XMAX, XLABEL,  YLABEL, TITLE):
	
	fsize = 11
	
	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	ax.bar(X, MODEL, BINW, align='center',\
				facecolor='lightblue', lw = 0.0)
	ax.step(np.append(X-BINW/2.0, X[-1]+BINW[-1]/2.0), np.append(MODEL,0.0), where='post',\
				color='dodgerblue', lw = 1.0)

	ax.errorbar(X, DATA, yerr= np.array([ERRORLOW,ERRORUP]), xerr = BINW/2.0, \
													marker="o", markeredgewidth=0.5, capsize=2.0,markerfacecolor="white",\
													markeredgecolor="black",ms=3, color='black', lw = 0.0, elinewidth=1.0, zorder=100)



	ax.set_title(TITLE, fontsize=fsize)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(YLABEL,fontsize=fsize)

	# ax.set_xlim(XMIN,XMAX)
	# ax.set_ylim(YMIN,YMAX)

	# plt.show()
	plt.savefig(plotname)
	plt.close()