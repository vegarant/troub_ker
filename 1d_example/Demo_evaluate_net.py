import matplotlib.pyplot as plt;
from scipy.io import savemat, loadmat; 
from scipy.io import savemat, loadmat;
from instabilities_tools import f1_linear, f1_linear_inv, export_poly_heav, load_poly_heav, src_data, fourier_operator, subsampled_ifourier_matrix, sample_data, l2_norm_of_tensor;
from pyfftw.interfaces import scipy_fftpack as fftw;
#from simple_nn import generate_training, u_net_leaky_network, ph_test_time, test_time_loss;
from os.path import join;
import sys;
import os;
from os.path import join;
import numpy as np;
import tensorflow as tf;
from Config_handler import Config_handler;
import configparser;
import functools;
from Data_loader import Data_loader;

dest_plots = './plots'
# Read the validation and training config
val_config_fname = 'config_val.ini';
config_val = configparser.ConfigParser()
config_val.read(val_config_fname)

runner_id         = int(config_val['VAL']['runner_id']);
dest_model        = config_val['VAL']['dest_model'];

dir_name = 'run_%03d' % (runner_id);
run_dir  = join(dest_model, dir_name);
train_config_fname = join(run_dir, 'config.ini')
config_train = configparser.ConfigParser();
config_train.read(train_config_fname);
run_dir_as_mod = run_dir.replace('/', '.');
exec_str = 'from %s.simple_nn import *;' % (run_dir_as_mod);
print(exec_str)
exec(exec_str);

# Create Config Handler

ch = Config_handler(config_train, config_val);

# Experiment
runner_id        = ch.runner_id
read_val_dataset = ch.read_val_dataset
data_set_type    = ch.data_set_type
im_nbr           = ch.im_nbr
epoch_nbr        = ch.epoch_nbr

# DATASET
N             = ch.N;
degree        = ch.degree;
discon        = ch.discon;
data_size     = ch.data_size;
train_size    = ch.train_size;
val_size      = ch.val_size;
test_size     = ch.test_size;
idx_file_name = ch.idx_file_name;
src_data      = ch.src_data;

# SETUP
use_gpu       = ch.use_gpu;
compute_node  = ch.compute_node;
ckpt_dir_name = ch.ckpt_dir;
model_name    = ch.model_name;
print_every   = ch.print_every
save_every    = ch.save_every;
fsize         = ch.fsize;

nbr_epochs    = ch.nbr_epochs;
trainable_ws  = ch.trainable_ws;
network_str   = ch.network_str;
prec          = ch.prec;

tumor_path          = ch.tumor_path;
tumor_amplitude     = ch.amplitude;
add_pert_nbr_train = ch.add_pert_nbr_train;
add_pert_nbr_val   = ch.add_pert_nbr_val;

print(ch);


if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node);
    dev_name = "/device:GPU:0"
else:
    print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"; 
    dev_name = "/device:CPU:%d" % (compute_node);

# Load sampling pattern
idx_data = loadmat(idx_file_name);
idx = np.squeeze(idx_data['idx']);
idx = idx - 1;
nbr_samples = idx.shape[0];



data_it = ph_test_time(N, nbr_samples, prec=prec, dev_name=dev_name);
x_real = data_it['x_real'];
x_imag = data_it['x_imag'];
label  = data_it['label'];

network = eval(network_str);
net     = network(x_real, x_imag, N, idx, in_batch_size=None, dev_name=dev_name)
pred    = net['pred'];


loss = test_time_loss(pred, label, N, dev_name=dev_name);

data_loader  = Data_loader(N, degree, discon, data_path=src_data, 
                           data_size =data_size,
                           train_size=train_size,
                           val_size  =val_size,
                           test_size =test_size, 
                           tumor_path=tumor_path,
                           tumor_amplitude=tumor_amplitude);

if data_set_type.lower() == "test":
    label_data  = data_loader.load_data_test();
    raw_size = test_size;
    tumor_pert_nbr = [];
elif data_set_type.lower() == "val":
    label_data  = data_loader.load_data_val(add_tumor_nbr=add_pert_nbr_val);
    raw_size = val_size;
    tumor_pert_nbr = add_pert_nbr_val;
elif data_set_type.lower() == "train":
    label_data  = data_loader.load_data_train(train_size,\
                                              add_tumor_nbr=add_pert_nbr_train);
    raw_size = train_size;
    tumor_pert_nbr = add_pert_nbr_train;
else:
    print('Error: Unknown data_set_type. test, va;, train');

sample_data = sample_data( label_data, idx );
samp_real = np.real( sample_data );
samp_imag = np.imag( sample_data );

saver = tf.train.Saver()

ckpt_dir = join(run_dir, ckpt_dir_name);

ckpt_nbr = 0;
if epoch_nbr == -1:
    ckpt_nbr = nbr_epochs;
else:
    ckpt_nbr = save_every*epoch_nbr;   
ckpt_name = join( ckpt_dir, "%s-%d" % (model_name, ckpt_nbr) );


with tf.Session() as sess:
    saver.restore(sess, ckpt_name) 
    
    out = sess.run(pred, feed_dict={x_real: samp_real,
                                    x_imag: samp_imag});

    l1 = sess.run(loss, feed_dict={x_real: samp_real,
                                   x_imag: samp_imag,
                                   label: label_data});

dest_plots_run  = join(dest_plots, 'run_%03d' % (runner_id));
dest_plots_full = join(dest_plots_run, data_set_type.lower());
if not os.path.isdir(dest_plots):
    os.mkdir(dest_plots);
if not os.path.isdir(dest_plots_run):
    os.mkdir(dest_plots_run);
if not os.path.isdir(dest_plots_full):
    os.mkdir(dest_plots_full);


print('loss: %g' % l1);
f = label_data[im_nbr, :,0];
approx = out[im_nbr, :, 0];
zero_p = np.zeros([raw_size+len(tumor_pert_nbr), N]);
for i in range(raw_size+len(tumor_pert_nbr)):
    zero_p[i, :] = np.real(fourier_operator(sample_data[i, :], 0, N, idx))
zero_p_im = zero_p[im_nbr, :];

nd1 = np.linalg.norm(f-approx);
nd2 = np.linalg.norm(f-zero_p_im);

print('For image %d:' % (im_nbr));
print('\n\n|f - net |: %g' % nd1);
print('\n\n|f - zero|: %g' % nd2);
print('\nFor all images');
print('\n\n|f - net |: %g' % l2_norm_of_tensor(label_data - out));
print('\n\n|f - zero|: %g' % l2_norm_of_tensor(np.squeeze(label_data) - zero_p));

t = np.linspace(0,1,N);
fsize = 14;
plt.rc('text', usetex=True);
#tumor_pert_nbr = [3];
im_height = 2.5;
for i in range(len(tumor_pert_nbr)):
    im_nbr = tumor_pert_nbr[i];
    print('im_nbr: ', im_nbr)
    x    = label_data[im_nbr];
    fAx  = out[im_nbr,:,0]
    adjx = np.real(zero_p[im_nbr,:]);
    
    xr    = label_data[raw_size+i,:,0];
    fAxr  = out[raw_size+i,:,0];
    adjxr = np.real(zero_p[raw_size+i,:]);

    ymax = max(np.amax(x), np.amax(fAx), np.amax(adjx), np.amax(xr), np.amax(fAxr), np.amax(adjxr));
    ymin = min(np.amin(x), np.amin(fAx), np.amin(adjx), np.amin(xr), np.amin(fAxr), np.amin(adjxr));
    
    ylim_bottom = ymin - 0.1*np.sign(ymin)*ymin;
    print('ymin: ', ymin, 'ylim_bottom: ', ylim_bottom);
    ylim_top = 1.1*ymax;

    fig = plt.figure();
    plt.plot(t, x, label=r'$x$');
    plt.plot(t, fAx, label=r'$\Psi(y), ~ y = Ax$');
    plt.xlim(0,1);
    plt.ylim(ylim_bottom, ylim_top);
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.legend(fontsize=fsize, loc='lower right');
    fig.set_size_inches(6, im_height)
    plt.savefig(join(dest_plots_full, 'image_%03d_x_fx.pdf' % (im_nbr)), dpi=150);
    plt.savefig(join(dest_plots_full, 'image_%03d_x_fx.eps' % (im_nbr)), dpi=150);
    plt.savefig(join(dest_plots_full, 'image_%03d_x_fx.png' % (im_nbr)), dpi=150);
#    plt.plot(t, adjx, label='real(A^* y)');
#    plt.legend(fontsize=fsize);
#    fig.set_size_inches(8,4)
#    plt.savefig(join(dest_plots_full, 'image_%03d_w_adj.eps' % (im_nbr)), dpi=150);
    plt.close(fig)
    
    fig = plt.figure();
    plt.plot(t, xr, label=r"$x'$");
    plt.plot(t, fAxr, label=r"$\Psi(y'), ~ y' = Ax'$");
    plt.xlim(0,1);
    plt.ylim(ylim_bottom, ylim_top);
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.legend(fontsize=fsize, loc='lower right');
    fig.set_size_inches(6, im_height)
    plt.savefig(join(dest_plots_full, 'image_%03d_xz_fxz.pdf' % (im_nbr)), dpi=150);
    plt.savefig(join(dest_plots_full, 'image_%03d_xz_fxz.eps' % (im_nbr)), dpi=150);
    plt.savefig(join(dest_plots_full, 'image_%03d_xz_fxz.png' % (im_nbr)), dpi=150);
#    plt.plot(t, adjxr, label='real(A^* y)');
#    plt.legend(fontsize=fsize);
#    fig.set_size_inches(8,4)
#    plt.savefig(join(dest_plots_full, 'image_%03d_tumor_w_adj.eps' % (im_nbr)), dpi=150);
    plt.close(fig)
    
    fig = plt.figure();
    plt.plot(t, xr, label=r"$x'$");
    plt.plot(t, fAx, label=r"$\Psi(y'+\tilde{e}_2), ~ y' = Ax'$");
    plt.xlim(0,1);
    plt.ylim(ylim_bottom, ylim_top);
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.legend(fontsize=fsize, loc='lower right');
    fig.set_size_inches(6, im_height)
    plt.savefig(join(dest_plots_full, 'image_%03d_xz_fxzr.pdf' % (im_nbr)), dpi=150);
    plt.savefig(join(dest_plots_full, 'image_%03d_xz_fxzr.eps' % (im_nbr)), dpi=150);
    plt.savefig(join(dest_plots_full, 'image_%03d_xz_fxzr.png' % (im_nbr)), dpi=150);
#    plt.plot(t, adjx, label='real(A^* y)');
#    plt.legend(fontsize=fsize);
#    fig.set_size_inches(8,4)
#    plt.savefig(join(dest_plots_full, 'image_%03d_tumor_w_adj_pert.eps' % (im_nbr)), dpi=150);
    plt.close(fig)

    fig = plt.figure();
    plt.plot(t, x, label=r"$x$");
    plt.plot(t, fAxr, label=r"$\Psi(y+\tilde{e}_1),~ y = Ax$");
    plt.xlim(0,1);
    plt.ylim(ylim_bottom, ylim_top);
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.legend(fontsize=fsize, loc='lower right');
    fig.set_size_inches(6, im_height)
    plt.savefig(join(dest_plots_full, 'image_%03d_x_fxr.pdf' % (im_nbr)), dpi=150);
    plt.savefig(join(dest_plots_full, 'image_%03d_x_fxr.eps' % (im_nbr)), dpi=150);
    plt.savefig(join(dest_plots_full, 'image_%03d_x_fxr.png' % (im_nbr)), dpi=150);
    plt.close(fig)


#for i in range(7,14):
#    im_nbr = i;
#    x    = label_data[im_nbr];
#    fAx  = out[im_nbr,:,0]
#    adjx = np.real(zero_p[im_nbr,:]);
#    
#    ymax = max(np.amax(x), np.amax(fAx), np.amax(adjx));
#    ymin = min(np.amin(x), np.amin(fAx), np.amin(adjx));
#    
#    ylim_bottom = ymin - 0.1*np.sign(ymin)*ymin;
#    print('ymin: ', ymin, 'ylim_bottom: ', ylim_bottom);
#    ylim_top = 1.1*ymax;
#
#    fig = plt.figure();
#    plt.plot(t, x, label='x');
#    plt.plot(t, fAx, label='f(y), y = Ax');
#    plt.xlim(0,1);
#    plt.ylim(ylim_bottom, ylim_top);
#    plt.legend();
#    plt.savefig(join(dest_plots_full, 'image_%03d.eps' % (im_nbr)), dpi=150);
#    plt.plot(t, adjx, label='real(A^* y)');
#    plt.legend();
#    plt.savefig(join(dest_plots_full, 'image_%03d_w_adj.eps' % (im_nbr)), dpi=150);
#    plt.close(fig)
    
    
#fig = plt.figure()
#plt.plot(t, f, label='f');
#plt.plot(t, approx, label='rec');
#plt.plot(t, zero_p, label='zero_p');
#plt.legend();
#plt.xlabel('t');
#plt.show();





