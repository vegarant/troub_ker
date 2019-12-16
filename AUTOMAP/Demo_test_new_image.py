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
from PIL import Image

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

N = 128;
new_im = mpimg.imread(join(src_mri_data, 'brain1_128_anonymous.png'));
train_data = scipy.io.loadmat(join(src_mri_data, 'HCP_mgh_1003_T2_selected_and_resized.mat'))
train_im = np.float32(train_data['im'])/255 # Data stored as uint8 originaly


mri_data = np.zeros([train_im.shape[0]+1, N,N], dtype='float32')
mri_data[0, :, :] = new_im
for i in range(train_im.shape[0]):
    mri_data[i+1, :, :] = train_im[i, :,:];
batch_size = mri_data.shape[0];
plot_dest = './plots';

if not (os.path.isdir(plot_dest)):
    os.mkdir(plot_dest);

sess = tf.Session();

raw_f, raw_df = compile_network(sess, batch_size);

f  = lambda x: hand_f( raw_f, x, k_mask_idx1, k_mask_idx2);

fx = f(mri_data)
print('amin(fx): ', np.amin(fx))
print('amax(fx): ', np.amax(fx))

for i in range(batch_size):
    image_data = np.uint8(255*scale_to_01(fx[i]));
    image_rec = Image.fromarray(image_data);
    image_rec.save(join(plot_dest, 'rec_image_%g.png' % (i)));
    image_orig = Image.fromarray(np.uint8(255*mri_data[i]));
    image_orig.save(join(plot_dest, 'orig_image_%g.png' % (i)));

sess.close();

#plt.figure();
#plt.subplot(121); plt.matshow(mri_data[0], cmap='gray', fignum=False); plt.colorbar()
#plt.subplot(122); plt.matshow(fx[0], cmap='gray', fignum = False); plt.colorbar();
#plt.show();





