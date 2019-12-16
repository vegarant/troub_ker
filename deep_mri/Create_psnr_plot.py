"""
This script reads data from file and create the plot shown in the paper
"""


from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

src = './'
fsize = 20

csdata_path = join(src, 'plots_more_samp', 'psnr_data', 'psnr_values_cs.mat')
dmdata_path = join(src, 'plots_more_samp', 'psnr_data', 'psnr_values_dm.mat')

csdata_dict = loadmat(csdata_path)
dmdata_dict = loadmat(dmdata_path)

csdata = csdata_dict['psnr_arr']
dmdata = dmdata_dict['psnr_arr']
print(csdata)
#print(csdata.shape)
#print(dmdata.shape)

cs_psnr_mean = csdata[:,0]; # np.mean(csdata, axis=1)
dm_psnr_mean = dmdata[:,0]; # np.mean(dmdata, axis=1)

max_elem = max(np.amax(cs_psnr_mean), np.amax(dm_psnr_mean));
min_elem = min(np.amin(cs_psnr_mean), np.amin(dm_psnr_mean));


subsampling_rate = 1/np.arange(14,1,-1)
frac = 0.05;

im_nbr = 13

plt.rc('text', usetex=True);

fig = plt.figure()
plt.plot(subsampling_rate, dm_psnr_mean, label=r'Network $\Psi$')
plt.plot(subsampling_rate, cs_psnr_mean, label=r'l1-min. $\Phi$')
plt.plot(np.array((1/3., 1/3.)), np.array([(1-frac)*min_elem, (1+frac)*max_elem]), 'r--')
plt.legend(fontsize=fsize)
plt.xlim(subsampling_rate[0], subsampling_rate[-1])
plt.ylim((1-frac)*min_elem, (1+frac)*max_elem)
plt.xlabel('Subsampling rate $m/N$', fontsize=fsize);
plt.ylabel('PSNR value', fontsize=fsize);
fig.set_size_inches(6,6)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

#plt.axis('square')
plt.savefig('dm_vs_cs.eps', dpi=150)
#plt.show()




