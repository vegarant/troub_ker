from __future__ import print_function, division

import sys;
from os.path import join
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format
from utils import compressed_sensing as cs
from main_2d import iterate_minibatch, prep_input
import numpy as np;
#from deep_mri_tools import scale_to_01, compute_psnr;
import scipy.misc;
from deep_mri_config import src_network_weights, cardiac_data_path, src_mri_data;
import scipy;
from numpy.lib.stride_tricks import as_strided
from deep_mri_tools import *;


# This script test the networks ability to reconstruction images from different 
# subsampling rates. The sampling patterns are drawn at random, so be aware that
# this might affect the reconstruction quality, between conseceutive runs. 


def inc_cartesian_mask(shape, acc, old_indices=None, sample_n=10, centred=False):
    """
    This function is a slight modification of the function 
    utils.compressed_sensing.cartesian_mask. 
    It is modified so that it can remember which lines that where sampled in
    the previuous sampling process and include these lines in the next sampling
    pattern. It assumes that the sampling ratio `acc` is larger than the
    previuous `acc`. This function is only written for this spesific example,
    and it should be fixed if you would like to do something different.   
    """
    
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = cs.normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx
    
    if sample_n and (old_indices is None):
        pdf_x[int(Nx/2-sample_n/2):int(Nx/2+sample_n/2)] = 0
        old_indices = np.arange(int(Nx/2-sample_n/2), int(Nx/2+sample_n/2));
        n_lines -= sample_n
    else: 
        pdf_x[old_indices] = 0;
        n_lines -= len(old_indices);
    pdf_x /= np.sum(pdf_x)
    mask = np.zeros((N, Nx))
   
    idx = np.random.choice(Nx, n_lines, False, pdf_x)
    
    idx = np.concatenate([idx, old_indices]);
    mask[:, idx] = 1

    if sample_n and (old_indices is None):
        mask[:, int(Nx/2-sample_n/2):int(Nx/2+sample_n/2)] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask, idx;

if __name__ == "__main__":
    N = 256

    plot_dest = './plots_more_samp'
    data_dest = join(plot_dest, 'psnr_data')
    mask_dest = join(plot_dest, 'mask_data')

    if not os.path.isdir(plot_dest):
        os.mkdir(plot_dest);
    if not os.path.isdir(data_dest):
        os.mkdir(data_dest);
    if not os.path.isdir(mask_dest):
        os.mkdir(mask_dest);

    sys.setrecursionlimit(2000);

    shuffle_batch = False;

    #batch_size = 30;
    #data = load_data(cardiac_data_path);
    #im = data[0:batch_size];
    #print('data.shape: ', data.shape)
    #im = data[0:batch_size];
    
    fname_data = join(src_mri_data, 'x0.mat');
    data_dict = scipy.io.loadmat(fname_data);
    im = data_dict['x0']; # im is a (2,256,256) complex valued array
    batch_size, Nx, Ny = im.shape; 
    print('batch_size: ', batch_size);
    input_shape = (batch_size, 2, Nx, Ny);

    # Load and compile network
    net_config, net = load_network(input_shape, src_network_weights);

    f = compile_f(net, net_config); # f(input_var, mask, k_space)

    us_rate = range(14,1,-1);
    n = len(us_rate);
    subsamp   = np.zeros(n);
    psnr_arr  = np.zeros([n, batch_size]);

    # Do first iteration outside loop;
    undersampling_rate = us_rate[0];
    mask, idx = inc_cartesian_mask((batch_size, Nx,Ny), undersampling_rate, sample_n=8);
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho');

    im_und_l = to_lasagne_format(im_und);
    k_und_l = to_lasagne_format(k_und);
    mask_l = to_lasagne_format(mask, mask=True);

    pred = f(im_und_l, mask_l, k_und_l);
    pred = from_lasagne_format(pred);


    psnr_values = np.zeros(batch_size);
    for i in range(batch_size):
        psnr_arr[0, i] = compute_psnr(pred[i], im[i]);
   
    print('psnr_arr: ', psnr_arr[0]);

    subsamp[0] = 1./undersampling_rate;
    
    amask = np.squeeze(mask[0,:,:]);
    plt.imsave(os.path.join(mask_dest, "mask_k_%d.png" % 0), 
               amask, cmap='gray');
    set_of_all_masks = np.zeros([len(us_rate), N, N])
    set_of_all_masks[0, :, :] = amask;
    for k in range(1,n):
        print('(k/n): %2d/%2d' % (k, n));
        undersampling_rate = us_rate[k];
        mask,idx = inc_cartesian_mask((batch_size, Nx, Ny), 
                                      undersampling_rate, 
                                      old_indices=idx,
                                      sample_n=8);
        
        amask = np.squeeze(mask[0,:,:]);
        set_of_all_masks[k, :, :] = amask;
        plt.imsave(os.path.join(mask_dest, "mask_k_%d.png" % k), 
                   amask, cmap='gray');
        
        #mask_name = os.path.join(dest, 'mask_data', 'mask_%d.mat' % (k));
        #scipy.io.savemat(mask_name, mdict={'mask': mask});
        im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho');
        #amask = np.squeeze(mask[0,:,:]);
        #plt.imsave(os.path.join(dest, 'mask_data', "mask_k_%d.png" % k), 
        #           amask, cmap='gray');

        im_und_l = to_lasagne_format(im_und);
        k_und_l = to_lasagne_format(k_und);
        mask_l = to_lasagne_format(mask, mask=True);

        pred = f(im_und_l, mask_l, k_und_l);
        pred = from_lasagne_format(pred);
        
        psnr_values = np.zeros(batch_size);
        for i in range(batch_size):
            psnr_arr[k, i] = compute_psnr(pred[i], im[i]);
             
        subsamp[k] = 1./undersampling_rate;

    
    psnr_name = os.path.join(data_dest, 'psnr_values_dm.mat');
    mask_name = os.path.join(mask_dest, 'set_of_all_masks.mat');
    scipy.io.savemat(psnr_name, mdict={'psnr_arr': psnr_arr});
    scipy.io.savemat(mask_name, mdict={'set_of_all_masks': set_of_all_masks});

    fsize=20;
    lwidth=2;
    
    fig = plt.figure();
    psnr_mean = np.mean(psnr_arr, axis=1);
    print('psnr_mean: ', psnr_mean)
    print('arr_mean: ', psnr_arr)
    plt.plot(subsamp, psnr_mean, '*-', linewidth=lwidth, ms=12); 
    p_max = np.amax(psnr_arr);
    p_min = np.amin(psnr_arr);
    x_bar = 1/3;
    plt.plot([x_bar, x_bar], [0.98*p_min, 1.02*p_max], 'r--', linewidth=lwidth);
    plt.axis([0, max(subsamp), 0.98*p_min, 1.02*p_max]);
    plt.xlabel('Subsampling rate', fontsize=fsize); 
    plt.ylabel('Avrage PSNR', fontsize=fsize);
    #plt.title('Deep MRI net', fontsize=fsize);
    fname_plot = os.path.join(plot_dest, 'deep_MRI_add_more_samples.png');
    fig.savefig(fname_plot, bbox_inches='tight');
    #plt.show();
