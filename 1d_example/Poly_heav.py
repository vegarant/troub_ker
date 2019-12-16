import numpy as np;
#import matplotlib.pyplot as plt;
from pyfftw.interfaces import scipy_fftpack as fftw;


class Poly_heav:
    
    def __init__(self, degree, discon):
        self.degree = degree;
        self.discon = discon;
        
        amp = 5;
        self.coeff = np.random.rand(degree +1);
        self.coeff[-1] = 10*self.coeff[-1]; 
        self.bias = amp*np.random.rand(1);
        
        amp_b = 0.5;
        self.a = np.random.rand(discon); 
        self.b = 2*amp_b*np.random.rand(discon) - amp_b;
        self.function_str = self._create_poly_heaviside_str();
        
    def _create_poly_heaviside_str(self):
        degree = self.degree;
        discon = self.discon;
        coeff  = self.coeff;
        bias   = self.bias;
        a      = self.a;
        b      = self.b;
        
        func_str = '';
        func_strings = [];
        for i in range(degree):
            func_strings.append('(x-%18.16f)' % ( coeff[i] ));
        if (degree > 0):
            func_str += '%18.16f*%s' % (coeff[-1], func_strings[0]);
            if (degree > 1):
                for i in range(degree-1):
                    func_str += '*' + func_strings[i+1];
            func_str  += ' + ';
        func_str += '%18.16f '  % (bias);
        
        func_strings = [];
        for i in range(discon):
            func_strings.append('%18.16f*(x > %18.16f)' % (b[i], a[i])); 
        for i in range(discon):
            func_str += ' + ' + func_strings[i];

        return func_str;

    def __call__(self, z):
        eval_str = 'lambda x: %s' % (self.function_str);
        f = eval(eval_str);
        return(f(z));



if __name__ == "__main__":
    
    N = 2**9;
    degree = 3;
    discon = 2;
    
    P = Poly_heav(degree, discon);
    #print(P.function_str);
    x = np.linspace(0,1,N);
    
    f = P(x);

    y = fftw.fft(f);
    
    
    
#    plt.plot(x,f);
#    plt.show();








