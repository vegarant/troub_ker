"""
This script test the Deep MRI net against gaussian noise.
"""

from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format
from utils import compressed_sensing as cs
from main_2d import iterate_minibatch, prep_input
import numpy as np
import sys
import os
from os.path import join
from scipy.io import loadmat, savemat
from deep_mri_config import src_network_weights, cardiac_data_path, src_mri_data 
from deep_mri_tools import load_network, compile_f, l2_norm_of_tensor, compute_psnr
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sys.setrecursionlimit(2000)
    
    dest = 'plots_gaus_stat'
    if not os.path.isdir(dest):
        os.mkdir(dest);

    nbr_itr = 1
    factor = 3;
    nbr_of_saved_worst = 1

    N = 256

    im_nbr = 0;
    
    fname_x0 = 'x0.mat'
    fname_mask = 'mask.mat'
    fname_rr = 'rr.mat'
    
    data_x0 = loadmat(join(src_mri_data, fname_x0))
    data_mask = loadmat(join(src_mri_data, fname_mask))
    data_rr = loadmat(join(src_mri_data, fname_rr))
    
    x0 = data_x0['x0'] # [2, N, N]
    mask = data_mask['mask'] # [2, N, N]
    rr = data_rr['rr'] # [16, 2, N, N]
   
    r1 = np.squeeze(rr[4,:,:,:])
    r2 = np.squeeze(rr[8,:,:,:])
    r3 = np.squeeze(rr[12,:,:,:])

    data = x0;

    norm_r1 = l2_norm_of_tensor(r1); 
    norm_r2 = l2_norm_of_tensor(r2); 
    norm_r3 = l2_norm_of_tensor(r3); 

    batch_size = x0.shape[0] # 2
    input_shape = [batch_size, 2, N, N]
    print('batch_size: ', batch_size)
    # Load and compile network
    net_config, net = load_network(input_shape, src_network_weights);
   
    f = compile_f(net, net_config); # f(input_var, mask, k_space)
    
    largest_nth_psnr_value = 100
    array_of_worst_psnr = largest_nth_psnr_value*np.ones(nbr_of_saved_worst)
    array_of_worst_pred = np.zeros([nbr_of_saved_worst, N,N], dtype='complex64')
    array_of_worst_noisy_images = np.zeros([nbr_of_saved_worst, N,N], dtype='complex64')
    array_of_worst_raw_noise = np.zeros([nbr_of_saved_worst, batch_size, N, N], dtype='complex64')
    array_of_psnr_values = np.zeros(nbr_itr)

    for itr in range(nbr_itr):
        print('%3d/%3d' % (itr, nbr_itr))
        noise_raw = np.random.normal(0,10, size=input_shape);
        noise_raw_complex = from_lasagne_format(noise_raw);
        noise_und, k_und = cs.undersample(noise_raw_complex, mask, centred=False, norm='ortho');

        #noise_raw = np.random.uniform(size=r_list[0].shape).astype('float32')
        #noise_raw_complex = from_lasagne_format(noise_raw)
        #k_und = mask*noise_raw_complex
        noise_und = np.fft.ifft2(k_und, norm='ortho')

        norm_noise_und = l2_norm_of_tensor(noise_und)
        scaled_noise2 = (factor*norm_r1*noise_und)/norm_noise_und

        data = from_lasagne_format(x0)
        im2_noisy = data + scaled_noise2

        im_und2, k_und2 = cs.undersample(im2_noisy, mask, centred=False, norm='ortho');

        mask_l = to_lasagne_format(mask, mask=True);

        im_und_l2 = to_lasagne_format(im_und2);
        k_und_l2  = to_lasagne_format(k_und2);

        pred2 = f(im_und_l2, mask_l, k_und_l2);
        pred2 = from_lasagne_format(pred2);
        pred2_single = pred2[im_nbr,:,:];
        im2_noisy_single = im2_noisy[im_nbr,:,:]

        psnr_value = compute_psnr(pred2_single, im2_noisy_single);
        array_of_psnr_values[itr] = psnr_value
        if psnr_value < largest_nth_psnr_value:
            idx = np.argmax(array_of_worst_psnr)
            array_of_worst_psnr[idx] = psnr_value
            array_of_worst_pred[idx, :, :] = pred2_single
            array_of_worst_noisy_images[idx, :, :] = im2_noisy_single
            array_of_worst_raw_noise[idx, :,:,:] = scaled_noise2 
            largest_nth_psnr_value = np.amax(array_of_worst_psnr)

    print(array_of_psnr_values)
    print(array_of_worst_psnr)


    order_small_to_large = np.argsort(array_of_worst_psnr);
    for i in range(nbr_of_saved_worst):
        idx = order_small_to_large[i];

        fname_im2 = 'im_noise1_fact_%g_worst_%d_place_%d.png' % (factor, nbr_itr, i)
        fname_rec2 = 'im_noise1_fact_%g_rec_worst_%d_place_%d.png' % (factor, nbr_itr, i)

        im2 = abs(np.squeeze(array_of_worst_noisy_images[idx]))
        rec2 = abs(np.squeeze(array_of_worst_pred[idx]))
        mpimg.imsave(join(dest, fname_im2), im2, cmap='gray');
        mpimg.imsave(join(dest, fname_rec2), rec2, cmap='gray');
    savemat(join(dest, 'worst_case_data_fact_%g_itr_%d.mat' % (factor, nbr_itr)), 
                                               {'array_of_psnr_values': array_of_psnr_values,
                                                'array_of_worst_psnr': array_of_worst_psnr,
                                                'array_of_worst_pred': array_of_worst_pred, 
                                                'array_of_worst_noisy_images': array_of_worst_noisy_images,
                                                'array_of_worst_raw_noise': array_of_worst_raw_noise})


