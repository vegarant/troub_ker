from Poly_heav import Poly_heav
from pyfftw.interfaces import scipy_fftpack as fftw;
from scipy.io import savemat, loadmat;
from os.path import join;
import numpy as np;

src_data = './data';

l2_norm_of_tensor = lambda x: np.sqrt((abs(x)**2).sum());

def export_poly_heav(ph, fname):
    degree = ph.degree;
    discon = ph.discon;
    coeff  = ph.coeff;
    bias   = ph.bias;
    a      = ph.a;
    b      = ph.b;
    
    savemat(fname, {'degree': degree,
                    'discon': discon,
                    'coeff':  coeff,
                    'bias':   bias,
                    'a':      a,
                    'b':      b});

def load_poly_heav(fname):
    f = loadmat(fname);
    
    degree = int(f['degree']);
    discon = int(f['discon']);
    coeff  = f['coeff'].flatten(); 
    bias   = f['bias']; 
    a      = f['a'].flatten(); 
    b      = f['b'].flatten(); 
    ph = Poly_heav(degree, discon);
    ph.coeff = coeff;
    ph.bias  = bias;
    ph.a     = a;
    ph.b     = b;
    ph.function_str = ph._create_poly_heaviside_str();
    
    return ph;

def f1_linear(U):
    N = U.shape[0];
    Y = np.zeros(U.shape, dtype=U.dtype);
    Nh = int(N/2);
    Y[0:N:2] = U[0:Nh];
    Y[1:N:2] = U[N:Nh-1:-1];
    return Y;

def f1_linear_inv(U):
    #print(U.shape)
    N = U.shape[0];
    Y = np.zeros(U.shape, dtype=U.dtype);
    Nh = int(N/2);
    Y[0:Nh]     = U[0:N:2];
    Y[N:Nh-1:-1] = U[1:N:2];
    return Y;


def fourier_operator(f, mode, N, idx=None):

    if (mode == 1): # Forward operator
        x = f1_linear(fftw.fft(f)/np.sqrt(N));
        if idx is not None:
            out = x[idx];
        else:
            out = x;

    else:

        if idx is not None:
            x = np.zeros(N, dtype=np.complex128);
            x[idx] = f; 
        else :
            x = f;

        out = fftw.ifft(f1_linear_inv(x))*np.sqrt(N);

    return out;
    
    
def subsampled_ifourier_matrix(N, idx): 
    X = np.zeros([N,N], dtype=np.complex128);
    ei = np.zeros(N, dtype=np.complex128);
    for i in range(N):
        ei = np.zeros(N, dtype=np.complex128);
        ei[i] = 1;
        X[:, i] = fourier_operator(ei, 0, N);

    Y = X[:, idx];   
    return Y;

def sample_data(data_batch, idx):

    batch_size = data_batch.shape[0];
    N = data_batch.shape[1];
    nbr_of_samples = len(idx);
    out = np.zeros([batch_size, nbr_of_samples], dtype='complex128');
    for i in range(batch_size):
        f = np.squeeze(data_batch[i,:,0]);
        samp = fourier_operator(f, 1, N, idx);
        out[i,:] = samp;
    return out;


def read_count(count_path = './'):
    """ Read and updates the runner count. 
    
    To keep track of all the different runs of the algorithm, one store the 
    run number in the file 'COUNT.txt' at ``count_path``. It is assumed that 
    the file 'COUNT.txt' is a text file containing one line with a single 
    integer, representing number of runs so far. 

    This function reads the current number in this file and increases the 
    number by 1. 
    
    :return: Current run number (int).
    """
    fname = join(count_path, 'COUNT.txt');
    infile = open(fname);
    data = infile.read();
    count = int(eval(data));
    infile.close();

    outfile = open(fname, 'w');
    outfile.write('%d ' % (count+1));
    outfile.close();
    return count;

if __name__ == "__main__":    
    
    
    
    a = 5;    




