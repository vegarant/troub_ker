
function y = fourier2IDB2d(x, mode, N, idx, nres, vm);

    if (~isvector(x))
        error('Input is not a vector');
    end

    R = round(log2(N));

    if (abs(2^R - N) > 0.5) 
        error('Input length is not equal 2^R for some R ∈ ℕ');
    end

    j0 = R-nres;

    if size(x,1) == 1
        x = x.';
    end

    if (mode == 1)

        %z = IWT_CDJV_noP(real(x), j0, vm) + 1j*IWT_CDJV_noP(imag(x), j0, vm);
        z = IWT_CDJV(x, j0, vm);
        z = f1_linear(fft(z))/sqrt(N);
        y = z(idx);
        y = y(:);
    else % Transpose

        z = zeros([N, 1]);
        z(idx) = x;
        z = ifft(f1_linear_inv(z))*sqrt(N);
        y = FWT_CDJV(real(z), j0, vm);

    end
end

