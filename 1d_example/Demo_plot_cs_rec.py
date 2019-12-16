"""
To run this script make sure you run the matlab/Demo_cs_reconstruction.m script 
first in order to create the right data files. Then this script will read this, 
and produce the images seen in the paper.
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt 
from os.path import join


im_nbr = 3;
dest_plots_full = './plots';
data = loadmat('matlab/plots/cs_rec.mat');
t = data['t'];
x = data['x'];
xz = data['xz'];
gx = data['gx'];
gxz = data['gxz'];


fsize = 14;
plt.rc('text', usetex=True);
im_height = 2.5;

ymax = max(np.amax(x), np.amax(xz), np.amax(gx), np.amax(gxz))
ymin = min(np.amin(x), np.amin(xz), np.amin(gx), np.amin(gxz))

ylim_bottom = ymin - 0.1*np.sign(ymin)*ymin;
print('ymin: ', ymin, 'ylim_bottom: ', ylim_bottom);
ylim_top = 1.1*ymax;

fig = plt.figure();
plt.plot(t, x, label=r'$x$');
plt.plot(t, gx, label=r'$\Phi(y), ~ y = Ax$');
plt.xlim(0,1);
plt.ylim(ylim_bottom, ylim_top);
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.legend(fontsize=fsize, loc='lower right');
fig.set_size_inches(6, im_height)
plt.savefig(join(dest_plots_full, 'image_%03d_cs_rec_x_gx.pdf' % (im_nbr)), dpi=150);
plt.savefig(join(dest_plots_full, 'image_%03d_cs_rec_x_gx.eps' % (im_nbr)), dpi=150);
plt.close(fig)

fig = plt.figure();
plt.plot(t, xz, label=r"$x'$");
plt.plot(t, gxz, label="$\Phi(y'), ~ y' = Ax'$");
plt.xlim(0,1);
plt.ylim(ylim_bottom, ylim_top);
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.legend(fontsize=fsize, loc='lower right');
fig.set_size_inches(6, im_height)
plt.savefig(join(dest_plots_full, 'image_%03d_cs_rec_xz_gxz.pdf' % (im_nbr)), dpi=150);
plt.savefig(join(dest_plots_full, 'image_%03d_cs_rec_xz_gxz.eps' % (im_nbr)), dpi=150);
plt.close(fig)
