import matplotlib.pyplot as plt;
from scipy.io import savemat, loadmat; 
from scipy.io import savemat, loadmat;
from instabilities_tools import f1_linear, f1_linear_inv, export_poly_heav, load_poly_heav, fourier_operator, subsampled_ifourier_matrix, sample_data, read_count;
from pyfftw.interfaces import scipy_fftpack as fftw;
#from simple_nn import generate_training, u_net_leaky_network;
from os.path import join;
import sys;
import os;
from os.path import join;
import numpy as np;
import tensorflow as tf;
from Data_loader import Data_loader, create_data_iterator;
from Config_handler import Config_handler;
import shutil;
import configparser;
import functools;

config_filename = './config.ini';
config = configparser.ConfigParser()
config.read(config_filename)

ch = Config_handler(config);
print(ch)

N             = ch.N;
degree        = ch.degree;
discon        = ch.discon;
data_size     = ch.data_size;
train_size    = ch.train_size;
val_size      = ch.val_size;
test_size     = ch.test_size;
idx_file_name = ch.idx_file_name;
src_data      = ch.src_data;

use_gpu       = ch.use_gpu;
compute_node  = ch.compute_node;
tf_log_level  = ch.tf_log_level;
prec          = ch.prec;
dest_model    = ch.dest_model;
print_every   = ch.print_every
save_every    = ch.save_every;
ckpt_dir_name = ch.ckpt_dir;
model_name    = ch.model_name;
counter_path  = ch.counter_path;
fsize         = ch.fsize;

nbr_epochs    = ch.nbr_epochs;
batch_size    = ch.batch_size;
trainable_ws  = ch.trainable_ws;
shuffle       = ch.shuffle;
optim         = ch.optim;
network_str   = ch.network_str;

lr_dict = ch.lr_dict;

tumor_path          = ch.tumor_path;
tumor_amplitude     = ch.amplitude;
add_pert_nbr_train = ch.add_pert_nbr_train;
add_pert_nbr_val   = ch.add_pert_nbr_val;


if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node);
    dev_name = "/device:GPU:0"
else: 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"; 
    dev_name = "/device:CPU:%d" % (compute_node);
if (tf_log_level+1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '%d' % (tf_log_level);

# Load sampling pattern
idx_data = loadmat(idx_file_name);
idx = np.squeeze(idx_data['idx']);
idx = idx - 1;
nbr_samples = idx.shape[0];

data_loader = Data_loader(N, degree, discon, data_path=src_data, 
                          data_size =data_size,
                          train_size=train_size,
                          val_size  =val_size,
                          test_size =test_size, 
                          tumor_path=tumor_path,
                          tumor_amplitude=tumor_amplitude);

label_data_train  = data_loader.load_data_train(train_size, add_tumor_nbr=add_pert_nbr_train);
label_data_val    = data_loader.load_data_val(add_tumor_nbr=add_pert_nbr_val);

sample_data_train = sample_data(label_data_train, idx);
sample_data_val   = sample_data(label_data_val, idx);

train_samp_real = np.real(sample_data_train);
train_samp_imag = np.imag(sample_data_train);

val_samp_real = np.real(sample_data_val);
val_samp_imag = np.imag(sample_data_val);


data_it = create_data_iterator(train_size, N, nbr_samples, shuffle, prec=prec);
x_real = data_it['x_real'];
x_imag = data_it['x_imag'];
label  = data_it['label'];

# Save everything
count = read_count(count_path=counter_path);

print("""
#########################################################
###              Saving as run: %-5d                 ###
#########################################################
""" % (count));

dir_name = 'run_%03d' % (count);
run_dir = join(dest_model, dir_name);
if not os.path.isdir(run_dir):
    os.mkdir(run_dir);
ckpt_dir = join(run_dir, ckpt_dir_name)
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir);

export_model = join(ckpt_dir, model_name);

# Copy simple_nn.py and config file, and load the newly moved 
# simple_nn file.
shutil.copyfile('./simple_nn.py', join(run_dir, 'simple_nn.py')); 
shutil.copyfile(config_filename, join(run_dir, 'config.ini')); 
init_py_file = join('%s' % (run_dir), '__init__.py');
open(init_py_file, 'a');
run_dir_as_mod = run_dir.replace('/', '.');
exec_str = 'from %s.simple_nn import *;' % (run_dir_as_mod);
exec(exec_str);

# Load the network architechture  
network = eval(network_str);
net = network(x_real, x_imag, N, idx, in_batch_size=None, dev_name=dev_name)
pred   = net['pred'];

train = generate_training(pred, label,  optim, N, lr_dict=lr_dict, batch_size=None, dev_name=dev_name);

label = train['label'];
loss  = train['loss'];
optimizer = train['optimizer'];

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

init = tf.global_variables_initializer();

epoch_arr = np.linspace(0, nbr_epochs, nbr_epochs//print_every);
train_loss_arr = np.zeros(nbr_epochs//print_every);
val_loss_arr   = np.zeros(nbr_epochs//print_every);


train_init_op = data_it['train_init_op'];
val_init_op   = data_it['val_init_op'];
data_x_real = data_it['data_x_real'];
data_x_imag = data_it['data_x_imag'];
data_label  = data_it['data_label'];
ph_batch_size = data_it['ph_batch_size'];
   


with tf.Session() as sess:
    sess.run(init);
    sess.run(train_init_op, feed_dict={data_x_real: train_samp_real, 
                                       data_x_imag: train_samp_imag, 
                                       ph_batch_size: batch_size,
                                       data_label: label_data_train});
    for epoch in range(nbr_epochs):
        # print current epoch
        sys.stdout.write("\rEpoch: %d/%d" % (epoch+1, nbr_epochs))
        sys.stdout.flush()
        out = sess.run(optimizer);
        
        if not (epoch % save_every):
            save_path = saver.save(sess, export_model, global_step=epoch)

        if not (epoch % print_every):

            train_loss = sess.run(loss);
            
            sess.run(val_init_op, feed_dict={data_x_real: val_samp_real, 
                                       data_x_imag: val_samp_imag, 
                                       ph_batch_size: val_size,
                                       data_label: label_data_val});

            val_loss = sess.run(loss);

            sess.run(train_init_op, feed_dict={data_x_real: train_samp_real, 
                                       data_x_imag: train_samp_imag, 
                                       ph_batch_size: batch_size,
                                       data_label: label_data_train});
            
            print('  Train loss: %g, Val loss: %g' % (train_loss, val_loss));
            train_loss_arr[epoch//print_every] = train_loss;
            val_loss_arr[epoch//print_every]   = val_loss;

    train_loss = sess.run(loss);
    sess.run(val_init_op, feed_dict={data_x_real: val_samp_real, 
                               data_x_imag: val_samp_imag, 
                               ph_batch_size: val_size,
                               data_label: label_data_val});

    val_loss = sess.run(loss);

    sess.run(train_init_op, feed_dict={data_x_real: train_samp_real, 
                               data_x_imag: train_samp_imag, 
                               ph_batch_size: batch_size,
                               data_label: label_data_train});
    
    print('  Train loss: %g, Val loss: %g' % (train_loss, val_loss));
    train_loss_arr[-1] = train_loss;
    val_loss_arr[-1]   = val_loss;


    save_path = saver.save(sess, export_model, global_step = nbr_epochs);


    print("\n\nModel saved in path: %s\n\n" % save_path)  
    savemat(join(run_dir, 'train_data.mat'), mdict={ 'val_loss_arr': val_loss_arr,
                                                     'train_loss_arr': train_loss_arr})


    fig = plt.figure();
    plt.semilogy(epoch_arr[2:], train_loss_arr[2:], label='Train'); 
    plt.semilogy(epoch_arr[2:], val_loss_arr[2:],   label='Validation'); 
    plt.xlabel('Number of epochs', fontsize=fsize);
    plt.ylabel('Loss', fontsize=fsize);
    plt.legend(fontsize=fsize);
    plt.savefig(join(run_dir, 'plot_loss.png'));






