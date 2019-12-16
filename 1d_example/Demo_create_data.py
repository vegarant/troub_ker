from Poly_heav import Poly_heav;
from scipy.io import savemat, loadmat;
from instabilities_tools import export_poly_heav, load_poly_heav, src_data;
from os.path import join;
import os;
import numpy as np;

nbr_of_phs = 99; # Number of training samples

N = int(2**9);   # number of sampling points of f(t) in the interval [0,1]
degree = 3;      # Degree of the polynomial
discon = 0       # Number of discontinuities

t = np.linspace(0, 1, N);
for i in range(nbr_of_phs):

    ph = Poly_heav(degree, discon);
    f_values = ph(t);

    fname_array = 'f_N_%d_deg_%d_disc_%d_nr_%04d.mat'  % (N, degree, discon, i);
    fname_ph = 'ph_deg_%d_disc_%d_nr_%04d.mat' % (degree, discon, i);
    export_poly_heav(ph, join(src_data, 'poly_heavs', fname_ph));
    savemat(join(src_data, 'arrays', fname_array), {'f_values': f_values});




