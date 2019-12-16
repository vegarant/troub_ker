import numpy as np;
import tensorflow as tf;
from scipy.io import loadmat;
from os.path import join;
import sys;

default_dev = '/device:CPU:0';

class Data_loader:
    
    def __init__(self, N, degree, discon, data_path, data_size, train_size, val_size, test_size, tumor_path, tumor_amplitude):

        self.N = N;
        self.degree = degree;
        self.discon = discon;
        self.data_path = data_path;
        self.data_size = data_size;
        self.train_size = train_size;
        self.val_size = val_size;
        self.test_size = test_size;
        self._data_count = 0;
        self.tumor_path = tumor_path;
        self.tumor_amplitude = tumor_amplitude;
        nbr_of_data = train_size + val_size + test_size
        if nbr_of_data > data_size:
            sys.stderr.write(
            'Warning: Data_loader, train_size+val_size+test_size > data_size\n');

# The data will be partitioned in the following way
# +-------------------------+------------+---------------+
# |    Training set         | Validation |   Test set    |
# +-------------------------+------------+---------------+
# i.e. the validation and test set is always chosen 
# among the last indices. The size of the various sets can be adjusted.

                
    def load_data_train(self, nbr_of_arrays, add_tumor_nbr=None, randomized=False):
        selected_nbrs = self._select_data_numbers_train(nbr_of_arrays, randomized=randomized);
        fname_list    = self._create_filenames(selected_nbrs);
        data = self._load_data(fname_list);
        if add_tumor_nbr is not None:
            tumor_signal = self._load_tumor_signal();
            data_to_be_copied = data[add_tumor_nbr].copy();
            for i in range(data_to_be_copied.shape[0]):
                data_to_be_copied[i,:,0] += tumor_signal;
            data = np.concatenate([data, data_to_be_copied], axis=0);
        return data;    

    def load_data_val(self, add_tumor_nbr=None):
        selected_nbrs = self._select_data_numbers_val();
        fname_list    = self._create_filenames(selected_nbrs);
        data = self._load_data(fname_list);
        if add_tumor_nbr is not None:
            tumor_signal = self._load_tumor_signal();
            data_to_be_copied = data[add_tumor_nbr].copy();
            for i in range(data_to_be_copied.shape[0]):
                data_to_be_copied[i,:,0] += tumor_signal;
            data = np.concatenate([data, data_to_be_copied], axis=0);
        return data;

    def load_data_test(self):
        selected_nbrs = self._select_data_numbers_test();
        fname_list    = self._create_filenames(selected_nbrs);
        return self._load_data(fname_list);

    def _load_tumor_signal(self):
        tumor_path = self.tumor_path;
        tumor_dict = loadmat(tumor_path);
        tumor = np.squeeze(tumor_dict['tumor']);
        tumor *= self.tumor_amplitude;
        return tumor;

    def _load_data(self, fname_list):
        N = self.N;
        nbr_of_arrays = len(fname_list);
        data = np.zeros([nbr_of_arrays, N]);
        for i in range(nbr_of_arrays):
            handle = loadmat(fname_list[i]);
            data[i, :] = np.squeeze(handle['f_values']);
        data = np.expand_dims(data, axis=2);
        return data;

    def _create_filenames(self, selected_nbrs): 
        nbr_elem = len(selected_nbrs);
        fname_list = nbr_elem*[None];
        for i in range(nbr_elem):
            fname = 'f_N_%d_deg_%d_disc_%d_nr_%04d.mat' % (self.N, self.degree,
                                                 self.discon, selected_nbrs[i]);
            fname_full = join(self.data_path, 'arrays', fname);
            fname_list[i] = fname_full;
        return fname_list;

    def _select_data_numbers_train(self, nbr_of_arrays, randomized):
        if not randomized:
            dc = self._data_count;
            over_size = (dc + nbr_of_arrays)// self.train_size;
            if over_size >= 2:
                print('Warning: nbr_of_arrays too large');
            if over_size: # Number of elements exceeds the data_size
                l1 = self.train_size - dc;
                l2 = (dc + nbr_of_arrays) % self.train_size
                selected_data = list(range(dc, self.train_size)) + list(range(l2));
                if len(selected_data) != (l1+l2):
                    print('Something is wrong');
                self._data_count = l2;
            else:
                selected_data = list(range(dc, dc+nbr_of_arrays));
                self._data_count = dc+nbr_of_arrays;
        else:
            selected_data = np.random.randint(0, self.train_size, size=[nbr_of_arrays]);

        return selected_data;

    def _select_data_numbers_val(self):
        ts = self.train_size;
        vs = self.val_size;
        selected_data = list(range(ts, ts+vs));
        return selected_data;

    def _select_data_numbers_test(self):
        ts = self.train_size;
        vs = self.val_size;
        test_s = self.test_size;
        selected_data = list(range(ts+vs, ts+vs+test_s));
        return selected_data;


def create_data_iterator(train_size, N, nbr_samples, shuffle, prec=tf.float32, dev_name=default_dev):

    ph_batch_size = tf.placeholder(tf.int64)
    data_x_real = tf.placeholder(dtype=prec, shape = [None, nbr_samples], name='data_x_real');
    data_x_imag = tf.placeholder(dtype=prec, shape = [None, nbr_samples], name='data_x_imag');
    data_label  = tf.placeholder(dtype=prec, shape = [None, N, 1], name='data_label')    

    if shuffle:
        train_dataset = tf.data.Dataset.from_tensor_slices(( data_x_real, data_x_imag, data_label )).batch(ph_batch_size).repeat().shuffle(train_size);
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(( data_x_real, data_x_imag, data_label )).batch(ph_batch_size).repeat();
    test_dataset  = tf.data.Dataset.from_tensor_slices(( data_x_real, data_x_imag, data_label )).batch(ph_batch_size).repeat();  # always batch even if you want to one shot it
    val_dataset   = tf.data.Dataset.from_tensor_slices(( data_x_real, data_x_imag, data_label )).batch(ph_batch_size).repeat();  # always batch even if you want to one shot it

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

    #with tf.device(dev_name): 
    x_real, x_imag, label  = iter.get_next()

    train_init_op = iter.make_initializer(train_dataset)
    test_init_op  = iter.make_initializer(test_dataset)
    val_init_op   = iter.make_initializer(test_dataset)

    return {'data_x_real': data_x_real, 'data_x_imag': data_x_imag, 
            'data_label': data_label, 'x_real': x_real, 'x_imag': x_imag, 
            'label': label, 'train_init_op': train_init_op, 
            'test_init_op': test_init_op, 'val_init_op': val_init_op,
            'ph_batch_size': ph_batch_size};




























