import configparser;
import functools;
import os;
import tensorflow as tf;


class Config_handler:
    """
Class to handle the configuration file. 
"""

    def __init__(self, config_train, config_val=None):
        """

The idea of this initialization is to read and set the right config options.
        
        """
        self.config_train = config_train;
        self.config_val = config_val;

        # Special validatation options
        if config_val is not None:

            self.runner_id         = int(config_val['VAL']['runner_id']);
            self.read_val_dataset  = eval(config_val['VAL']['read_val_dataset']);
            self.data_set_type     = config_val['VAL']['data_set_type'];
            self.im_nbr            = int(config_val['VAL']['im_nbr']);
            self.dest_model        = config_val['VAL']['dest_model'];
            self.epoch_nbr         = int(config_val['VAL']['epoch_nbr'])
            self.use_gpu           = eval(config_val['VAL']['use_gpu']);
            self.compute_node      = int(config_val['VAL']['compute_node']);
            if 'TUMOR' in config_val.keys():
                if 'add_pert_nbr_val' in config_val['TUMOR'].keys():
                    self.add_pert_nbr_val = \
                            eval(config_val['TUMOR']['add_pert_nbr_val']);

        else: # Special for training 

            self.use_gpu      = eval(config_train['SETUP']['use_gpu']);
            self.compute_node = eval(config_train['SETUP']['compute_node']);
            self.dest_model   = config_train['SETUP']['dest_model'];
        

        if config_val is not None and self.read_val_dataset:
            config = config_val;
            print('\nUsing config_val\n')
        else:
            config = config_train;
            print('\nUsing config_train\n')
        
        # Dataset paramters
        self.N             = int(eval(config['DATASET']['N']));
        self.degree        = int(config['DATASET']['degree']);
        self.discon        = int(config['DATASET']['discon']);
        self.data_size     = int(config['DATASET']['data_size']);
        self.train_size    = int(config['DATASET']['train_size']);
        self.val_size      = int(config['DATASET']['val_size']);
        self.test_size     = self.data_size - self.val_size - self.train_size;
        self.idx_file_name = config['DATASET']['idx_file_name'];
        self.src_data      = config['DATASET']['src_data'];


        # Training paramters
        self.nbr_epochs   = int(config_train['TRAIN']['nbr_epochs']);
        self.batch_size   = int(config_train['TRAIN']['batch_size']);
        self.trainable_ws = eval(config_train['TRAIN']['trainable_ws']);
        self.shuffle      = eval(config_train['TRAIN']['shuffle']);
        self.optim        = config_train['TRAIN']['optim'];
        self.network_str  = config_train['TRAIN']['network'];
        
        self.tumor_path = config_train['TUMOR']['tumor_path'];
        self.amplitude = float(config_train['TUMOR']['amplitude']);
        self.add_pert_nbr_train = eval(config_train['TUMOR']['add_pert_nbr_train']);
        if config_val is None:
            self.add_pert_nbr_val = \
                     eval(config_train['TUMOR']['add_pert_nbr_val']);
        else:
            if 'TUMOR' in config_val.keys():
                if 'add_pert_nbr_val' not in config_val['TUMOR'].keys():
                    self.add_pert_nbr_val = \
                         eval(config_train['TUMOR']['add_pert_nbr_val']);

        
        # Gradient decent paramters
        if self.optim.lower() == 'gd':
            start_lr    = float(config_train['TRAIN']['start_lr']); # initial learning rate 
            decay_base  = float(config_train['TRAIN']['decay_base']);
            decay_every = float(config_train['TRAIN']['decay_every']);
            staircase   = eval(config_train['TRAIN']['staircase']);
            self.lr_dict = {'start_lr':    start_lr, 
                       'decay_base':  decay_base, 
                       'decay_every': decay_every,
                       'staircase':   staircase};
        else:
            self.lr_dict = None;

        # Needs to be taken from training setup
        self.prec         = eval(config_train['SETUP']['prec']);
        self.counter_path = config_train['SETUP']['counter_path'];
        if 'TF_CPP_MIN_LOG_LEVEL' in config_train['SETUP'].keys():
            self.tf_log_level   = int(config_train['SETUP']['TF_CPP_MIN_LOG_LEVEL']);
        else:
            self.tf_log_level   = False;
        print_every = int(config_train['TRAIN_SETUP']['print_every']);
        self.print_every = print_every;
        self.save_every = int(eval(config_train['TRAIN_SETUP']['save_every']));
        self.fsize = int(eval(config_train['TRAIN_SETUP']['fsize']));
        self.model_name = config_train['TRAIN_SETUP']['model_name'];
        self.ckpt_dir = config_train['TRAIN_SETUP']['ckpt_dir'];


    def __str__(self):
        if self.config_val is not None:
            val_str = """
#########################################################
###              RUNNER ID: %-5d                     ###
#########################################################

VALIDATION SETUP: 
read_val_dataset  = %s
data_set_type     = %s
im_nbr            = %d
dest_model        = %s
epoch_nbr         = %d
use_gpu           = %s
compute_node      = %d

""" % (self.runner_id, self.read_val_dataset, self.data_set_type, 
self.im_nbr, self.dest_model, self.epoch_nbr, self.use_gpu, 
self.compute_node);

        train_str = """
DATASET:
N             = %d
degree        = %d
discon        = %d
data_size     = %d
train_size    = %d
val_size      = %d
test_size     = %d
idx_file_name = %s
src_data      = %s

SETUP:
use_gpu      = %s
compute_node = %d 
dest_model   = %s
prec         = %s
print_every  = %d
save_every   = %d

TRAIN PARAMTERS:
nbr_epochs   = %d
batch_size   = %d
trainable_ws = %s
shuffle      = %s
optim        = %s
network_str  = %s

TUMOR:
tumor_path          = %s
amplitude          = %g
add_pert_nbr_train = %s
add_pert_nbr_val   = %s

""" % ( self.N, self.degree, self.discon, self.data_size, self.train_size, 
self.val_size, self.test_size, self.idx_file_name, self.src_data, self.use_gpu, 
self.compute_node, self.dest_model, self.prec, self.print_every, 
self.save_every, self.nbr_epochs, self.batch_size, self.trainable_ws, 
self.shuffle, self.optim, self.network_str, self.tumor_path, self.amplitude, 
self.add_pert_nbr_train, self.add_pert_nbr_val); 
        train_header = """
#########################################################
###                TRAINING CONFIG                    ###
#########################################################
"""
            
        if self.config_val is not None:
            return val_str + train_str;
        else:
            return train_header + train_str;











