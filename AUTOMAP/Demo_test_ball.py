import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg;
import numpy as np;
from automap_config import src_weights, src_mri_data, src_k_mask;
from automap_tools import *;
from PIL import Image

plot_dest = './plots';
fraction_of_l2_norm_rr = 1/2.0;

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();
fname= join(src_k_mask, 'k_mask1.mat');
mask_data_extra = scipy.io.loadmat(join(src_k_mask, 'k_mask1.mat'));
# Sampling mask as a array of shape [N,N] where center element corresponds to 
# zero frequency
mask_as_image = mask_data_extra['k_mask'];

# Load the images and perturbations used in the previous paper
data = scipy.io.loadmat(join(src_mri_data, 'Automap_data.mat'));

mri_data = data['x'];
perturbation = data['rr'];
N = mri_data.shape[1];

nbr_of_images = mri_data.shape[0];

sess = tf.Session();

batch_size = 1
raw_f, raw_df = compile_network(sess, batch_size);

f  = lambda x: hand_f( raw_f, x, k_mask_idx1, k_mask_idx2);

fname_rand_pert = 'random_pert.mat';
rand_pert_array = np.zeros([nbr_of_images, N,N], dtype='complex128');

for i in range(0,nbr_of_images):
    
    # 
    if i == 0:
        rr = perturbation[i+1, :, :]
    else:
        rr = perturbation[i, :,:];
    fourier_data_rr = sampling_op_forward(rr, mask_as_image)
    random_pert = np.multiply(mask_as_image,
                              np.random.randn(N,N) + 1j*np.random.randn(N,N));
    l2_norm_rr = l2_norm_of_tensor(rr) 
    l2_norm_fourier_rr = l2_norm_of_tensor(fourier_data_rr) 
    l2_norm_random_pert = l2_norm_of_tensor(random_pert)
    random_pert *= fraction_of_l2_norm_rr*l2_norm_fourier_rr/l2_norm_random_pert
    print(l2_norm_of_tensor(random_pert)/l2_norm_fourier_rr);
    

    random_pert_image_domain = sampling_op_adjoint(random_pert, mask_as_image)
    rand_pert_array[i, :, :] = random_pert_image_domain

    l2_norm_image_domain = l2_norm_of_tensor(random_pert_image_domain);   
    if i == 0:
        rr_new = random_pert_image_domain
    else:
        rr_new = rr + random_pert_image_domain
    
    l2_norm_new_pert = l2_norm_of_tensor(rr_new);
    print('|r|: %g, |Ar|: %g, |r_new|: %g, |r_im|: %g' % (l2_norm_rr, 
                                              l2_norm_fourier_rr,
                                              l2_norm_new_pert,
                                              l2_norm_image_domain))


    image_out = mri_data[i] + rr_new;
    print('amax(image_out): ', np.amax(np.abs(image_out)));
    image_out_uint8 = np.uint8(255*np.abs(image_out));
    image_out = np.expand_dims(image_out, axis=0);
    
    fx = f(image_out)
    fx = 255*scale_to_01(fx)
    
    image_data = np.uint8(fx[0])
    image_fx = Image.fromarray(image_data)
    image_fx.save(join(plot_dest, 'fx_pert_%d.png' % (i)));
    image_xr = Image.fromarray(image_out_uint8);
    image_xr.save(join(plot_dest, 'xr_pert_%d.png' % (i)));

scipy.io.savemat(join(src_mri_data, fname_rand_pert), {'rand_pert': rand_pert_array})

sess.close();



