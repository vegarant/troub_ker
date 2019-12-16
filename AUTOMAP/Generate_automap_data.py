
import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg;
import numpy as np;
from automap_config import src_weights, src_mri_data;
from automap_tools import *;
from Runner import Runner;
from Automap_Runner import Automap_Runner;

data = scipy.io.loadmat(join(src_mri_data, 'HCP_mgh_1033_T2_128_w_symbol.mat'));
mri_data = data['mr_images_w_symbol'];
N = 128;

batch_size = mri_data.shape[0];
print('Batch size: ', batch_size);

runner_id = 50;

runner1 = load_runner(runner_id);

x = np.zeros([5,N,N], dtype=mri_data.dtype);
rr = np.zeros([5,N,N], dtype=runner1.r[1].dtype);

x[0, :, :] = mri_data[3, :,:];
x[1, :, :] = mri_data[3, :,:];
x[2, :, :] = mri_data[3, :,:];
x[3, :, :] = mri_data[4, :,:];
x[4, :, :] = mri_data[5, :,:];

rr[1, :, :] = runner1.r[1][3,:,:];
rr[2, :, :] = runner1.r[2][3,:,:];
rr[3, :, :] = runner1.r[3][4,:,:];
rr[4, :, :] = runner1.r[4][5,:,:];

fname = 'Automap_data.mat';
scipy.io.savemat(fname, {'x': x, 'rr': rr});



