% We use this function rather than matlabs `psnr` function to ensure that 
% we compute the psnr values in the same way
function psnr_val = compute_psrn(rec, ref)

    mse = sum( abs(rec(:)-ref(:)).^2 )/prod(size(ref));
    I_max = max(abs(ref(:)));

    psnr_val = 10*log10( (I_max*I_max)/mse );

end

