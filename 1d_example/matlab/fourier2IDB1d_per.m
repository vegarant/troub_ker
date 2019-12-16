function y = fourier2IDB2d(x, mode, N, idx, nres, vm);

    if (~isvector(x))
        error('Input is not a vector');
    end

    R = round(log2(N));

    if (abs(2^R - N) > 0.5) 
        error('Input length is not equal 2^R for some R ∈ ℕ');
    end

    j0 = R-nres;

    S = get_wavedec_s(R, nres);
    wname = sprintf('db%d', vm);

    if (mode == 1)

        %z = IWT_CDJV_noP(real(x), j0, vm) + 1j*IWT_CDJV_noP(imag(x), j0, vm);
        z = IWT_CDJV(real(x), j0, vm) + 1j*IWT_CDJV(imag(x), j0, vm);
        %z = waverec(real(x), S, wname) + 1j*waverec(imag(x), S, wname);
        z = fftshift(fft(z))/sqrt(N);
        y = z(idx);
    else % Transpose

        z = zeros([N, 1]);
        z(idx) = x;
        z = ifft(ifftshift(z))*sqrt(N);
        %y = wavedec(real(z), nres, wname) + 1j*wavedec(imag(z), nres, wname);
        %y = FWT_CDJV_noP(real(z), j0, vm) + 1j*FWT_CDJV_noP(imag(z), j0, vm);
        y = FWT_CDJV(real(z), j0, vm) + 1j*FWT_CDJV(imag(z), j0, vm);

    end
end


