"""
This scripts creates the plot of the tumor and the fourier transform of the tumor.
"""

from scipy.io import loadmat;
import numpy as np;
import matplotlib.pyplot as plt;
from os.path import join;
from instabilities_tools import src_data;

def f1_linear(U):
    N = U.shape[0];
    Y = np.zeros(U.shape, dtype=U.dtype);
    Nh = int(N/2);
    Y[0:N:2] = U[0:Nh];
    Y[1:N:2] = U[N:Nh-1:-1];
    return Y;

if __name__ == "__main__":
    dest = 'plots';
    tumor_data = loadmat(join(src_data, 'tumor.mat'));
    idx_data = loadmat(join(src_data, 'sample_idx.mat'));
    tumor = np.squeeze(tumor_data['tumor']);
    idx = np.squeeze(idx_data['idx']);
    idx -= 1; 
    fsize = 14;

    N = len(tumor);
    t = np.linspace(0,1,N);
    
    print('||z||_2: ', np.linalg.norm(tumor))
    print(np.amax(idx));
    
    plt.rc('text', usetex=True);

    fig = plt.figure();
    plt.plot(t, tumor);
    plt.xlim(0,1);
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    fig.set_size_inches(6,2.2)
    fig.savefig(join(dest, 'plot_tumor.eps'));
    fig.savefig(join(dest, 'plot_tumor.png'));
    plt.close();

    fcoeff = abs(np.fft.fft(tumor)/np.sqrt(N));
    fcoeff_lin = f1_linear(fcoeff);
    one_to_N = np.linspace(0,N-1,N);

    fig = plt.figure();
    
    plt.xlim(0,1);
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    #plt.legend(fontsize=fsize);
    #frame1 = plt.gca();
    plt.plot(one_to_N, fcoeff_lin);
    plt.stem(one_to_N[idx], fcoeff_lin[idx], markerfmt='ro', linefmt='r-');
    
    print('||PFz||_2: ', np.linalg.norm(fcoeff_lin[idx]));

    plt.xlim(0,N-1);
    #frame1.axes.get_xaxis().set_ticks([0,])
    fig.set_size_inches(6,2.2)
    fig.savefig(join(dest, 'plot_ftumor.eps'));
    fig.savefig(join(dest, 'plot_ftumor.png'));
    plt.close();


