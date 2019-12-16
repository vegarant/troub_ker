
function [rec, z] = sample_fourier_wavelet_1d(func_val, noise, idx, vm)

    if size(func_val, 1) == 1
        func_val = func_val.';
    end
    N = size(func_val, 1);
    r = round(log2(N));
    wname = sprintf('db%d', vm);
    nres = wmaxlev(N, wname)-1; % Obs: increasing this might result in failure
    j0 = r - nres;
    j0
    b = f1_linear(fft(func_val))/sqrt(N);    
    b = b(idx);

    opA = @(x, mode) fourier2IDB1d(x, mode, N, idx, nres, vm);
    
    spgl1_verbose = 1;
    
    %  minimize ||x||_1  s.t.  ||Ax - b||_2 <= sigma
    opts_spgl1 = spgSetParms('verbosity', spgl1_verbose, 'iterations', 300);
    z    = spg_bpdn(opA, b, noise, opts_spgl1); 
    
%    S = get_wavedec_s(r, nres);
    %rec = waverec(real(z), S, wname);
    rec = IWT_CDJV(real(z), j0, vm);
    %rec = IWT_CDJV_noP(real(z), j0, vm);
    
end 




