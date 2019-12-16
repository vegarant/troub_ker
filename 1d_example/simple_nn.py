import tensorflow as tf;
from instabilities_tools import subsampled_ifourier_matrix;
import numpy as np;
import keras;

default_dev = '/device:CPU:0';

def generate_training(prediction, label, optim,  N, lr_dict=None,batch_size=None, dev_name=default_dev, prec=tf.float32):

    if lr_dict is not None and optim.lower() == 'gd':
        start_lr    = lr_dict['start_lr'];
        decay_every = lr_dict['decay_every'];
        decay_base  = lr_dict['decay_base'];
        staircase   = lr_dict['staircase'];


    global_step = tf.Variable(initial_value=0, trainable=False);
    with tf.device(dev_name):
        #label = tf.placeholder( dtype=prec,
        #                        shape = [batch_size, N, 1],
        #                        name='label' );
        
        #loss = tf.losses.mean_squared_error(label, prediction);


        loss = tf.reduce_sum(tf.pow(prediction-label, 2))/(2*N);   
        
        if optim.lower() == 'gd':

            learning_rate = tf.train.exponential_decay(
                                                   learning_rate = start_lr,
                                                   global_step   = global_step,
                                                   decay_steps   = decay_every,
                                                   decay_rate    = decay_base,
                                                   staircase     = staircase);

            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                                                         loss, 
                                                         global_step=global_step);

        elif optim.lower() == 'adam': 
            optimizer = tf.train.AdamOptimizer().minimize(loss); 
        
        
    return {'label': label,
            'optimizer': optimizer,
            'loss': loss};




def ph_test_time(N, nbr_samples, prec=tf.float32, dev_name=default_dev):
    with tf.device(dev_name):
        x_real = tf.placeholder(dtype=prec, shape = [None, nbr_samples], name='x_real');
        x_imag = tf.placeholder(dtype=prec, shape = [None, nbr_samples], name='x_imag');
        label  = tf.placeholder(dtype=prec, shape = [None, N, 1], name='label')    
    return {'x_real': x_real, 'x_imag': x_imag, 'label': label};

def test_time_loss(prediction, label, N, dev_name=default_dev):
    with tf.device(dev_name):
        loss = tf.reduce_sum(tf.pow(prediction-label, 2))/(2*N);   
    return loss;

def u_net_large_leaky_network(x_real, x_imag, N, idx, in_batch_size=None,
                    dev_name=default_dev, 
                    prec=tf.float32,
                    trainable_ws = False):
    nbr_samples = len(idx);
    A_transp = subsampled_ifourier_matrix(N, idx);
    act = tf.nn.leaky_relu;
    kernel_size = 3;
    filters = 10;
    pool_size = 2;
    with tf.device(dev_name):
        #x_real = tf.placeholder(dtype=prec, shape = [in_batch_size, nbr_samples], name='x_real');
        #x_imag = tf.placeholder(dtype=prec, shape = [in_batch_size, nbr_samples], name='x_imag');
        
        x_real1 =  tf.transpose(x_real, perm=[1,0]);
        x_imag1 =  tf.transpose(x_imag, perm=[1,0]);
        print('x_real1.shape: ', x_real1.shape)
        init_A_real = tf.Variable(initial_value=np.real(A_transp),  
                                  dtype=prec, 
                                  trainable=trainable_ws);

        init_A_imag = tf.Variable(initial_value=np.imag(A_transp),  
                                  dtype=prec, 
                                  trainable=trainable_ws);
        
        u1 = tf.linalg.matmul(init_A_real, x_real1); # Shape [N, in_batch_size]
        u2 = tf.linalg.matmul(init_A_imag, x_imag1); 
        u3 = tf.linalg.matmul(init_A_real, x_imag1);
        u4 = tf.linalg.matmul(init_A_imag, x_real1);
        
        x1 = u1 - u2;
        x2 = u3 + u4;
        
        # Reorganize the input
        x1 = tf.transpose(x1, perm=[1,0]);
        x2 = tf.transpose(x2, perm=[1,0]);
        
        x1 = tf.expand_dims(x1, 2);
        x2 = tf.expand_dims(x2, 2);
        
        im = tf.concat([x1,x2], axis=2); # [batch_size, height, channel_in]
        
        ksh = kernel_size // 2;
        paddings = tf.constant([ [0,0], [ksh, ksh], [0,0] ]);
        
        im1 = tf.pad(im, paddings, "SYMMETRIC");
        im2 = tf.layers.conv1d(im1, filters=filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True,
                             activation=act);


        im2 = tf.pad(im2, paddings, "SYMMETRIC");
        im3 = tf.layers.conv1d(im2, filters=filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);

        im3 = tf.pad(im3, paddings, "SYMMETRIC");
        im4 = tf.layers.conv1d(im3, filters=filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);

        im1_1 = keras.layers.MaxPooling1D(pool_size, strides=2 ,padding='same')(im4)
        
        im1_1 = tf.pad(im1_1, paddings, "SYMMETRIC");
        im2_1 = tf.layers.conv1d(im1_1, filters=2*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);

        im2_1 = tf.pad(im2_1, paddings, "SYMMETRIC");
        im3_1 = tf.layers.conv1d(im2_1, filters=2*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);
        
        im1_2 = keras.layers.MaxPooling1D(pool_size, strides=2 ,padding='same')(im3_1)

        im1_2 = tf.pad(im1_2, paddings, "SYMMETRIC");
        im2_2 = tf.layers.conv1d(im1_2, filters=4*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);
        
        im2_2 = tf.pad(im2_2, paddings, "SYMMETRIC");
        im3_2 = tf.layers.conv1d(im2_2, filters=4*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);

        im1_3 = keras.layers.MaxPooling1D(pool_size, strides=2 ,padding='same')(im3_2)
        
        im1_3 = tf.pad(im1_3, paddings, "SYMMETRIC");
        im2_3 = tf.layers.conv1d(im1_3, filters=8*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);
        
        im2_3 = tf.pad(im2_3, paddings, "SYMMETRIC");
        im3_3 = tf.layers.conv1d(im2_3, filters=8*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);
        
        im3_3 = tf.expand_dims(im3_3, axis=1);
        
        im4_2 = tf.keras.layers.Conv2DTranspose(filters=2*filters, 
                                           kernel_size=[1, kernel_size],
                                           strides=[1,2],
                                           padding='same',
                                           use_bias=True,
                                           data_format='channels_last',
                                           activation=act)(im3_3);
        im4_2 = tf.squeeze(im4_2, axis=1);
        im5_2 = tf.concat([im4_2, im3_2], axis=2); 

        im5_2 = tf.pad(im5_2, paddings, "SYMMETRIC");
        im6_2 = tf.layers.conv1d(im5_2, filters=8*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);

        im6_2 = tf.pad(im6_2, paddings, "SYMMETRIC");
        im7_2 = tf.layers.conv1d(im6_2, filters=8*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);

        im7_2 = tf.expand_dims(im7_2, axis=1);

        im4_1 = keras.layers.Conv2DTranspose(filters=2*filters, 
                                           kernel_size=[1, kernel_size],
                                           strides=[1,2],
                                           padding='same',
                                           use_bias=True,
                                           data_format='channels_last',
                                           activation=act)(im7_2);
        im4_1 = tf.squeeze(im4_1, axis=1);

        im5_1 = tf.concat([im4_1, im3_1], axis=2); 
        im5_1 = tf.pad(im5_1, paddings, "SYMMETRIC");
        im6_1 = tf.layers.conv1d(im5_1, filters=2*filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);

        im6_1 = tf.expand_dims(im6_1, axis=1);
        
        im5 = keras.layers.Conv2DTranspose(filters=filters, 
                                           kernel_size=[1, kernel_size],
                                           strides=[1,2],
                                           padding='same',
                                           use_bias=True,
                                           data_format='channels_last',
                                           activation=act)(im6_1);
        im5 = tf.squeeze(im5, axis=1);
        im6 = tf.concat([im5, im4], axis=2); 

        im6 = tf.pad(im6, paddings, "SYMMETRIC");
        im7 = tf.layers.conv1d(im6, filters=filters, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);
        
        im7 = tf.pad(im7, paddings, "SYMMETRIC");
        pred = tf.layers.conv1d(im7, filters=1, 
                             kernel_size=kernel_size, 
                             strides=1, 
                             padding='valid',
                             use_bias = True, 
                             activation=act);
        print('pred.shape: ', pred.shape)

    return {'x_real': x_real, 'x_imag': x_imag, 'pred': pred};













