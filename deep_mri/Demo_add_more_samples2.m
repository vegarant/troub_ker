clear all; close all;
deep_mri_config; % src_mri_data

dwtmode('per', 'nodisp');

src_images = fullfile(src_mri_data, 'x0.mat');
dest = fullfile('plots_more_samp', 'psnr_data');


load(src_images); % x0

%images = double(permute(seq, [3,1,2]));
images = double(x0);

masks = load(fullfile('plots_more_samp', 'mask_data', 'set_of_all_masks.mat' ));
mask = masks.set_of_all_masks;

nbr_of_masks = size(mask, 1);
batch_size = 2;
im_nbr = 1
N = 256;


wave_name = 'db4';
nres = wmaxlev(N, wave_name);
noise = 0.001;

psnr_arr = zeros(nbr_of_masks, batch_size);

im = squeeze(images(im_nbr, :,:));
for mask_nbr = 1:nbr_of_masks
    amask = double(squeeze(mask(mask_nbr, :, :)));
    %amask = logical(fftshift(amask));
    amask = fftshift(amask);
    idx = find(amask);

    b = fftshift(fft2(im))./N;             % 1/N is a normalizing factor        
    b = b(idx);                            % Extract the sampled indices


    % Initialize the `A` matrix operator                                        
    opA = @(x, mode) csl_op_fourier_wavelet_2d(x, mode, N, idx, nres, wave_name);
        %A = getFourierOperator([N,N], amask);
    %mask_nbr
    %for im_nbr = 1:1

    %    im = squeeze(images(im_nbr, :,:));
    opts = spgSetParms('verbosity', 1);          
    z    = spg_bpdn(opA, b, noise, opts);                                       

    % Reconstruct image                                                         
    s = csl_get_wavedec2_s(round(log2(N)), nres);                   
    im_rec  = waverec2(z, s, wave_name);                                        
    reshape(im_rec, [N,N]);
    psnr_val = compute_psnr(im_rec, im)
    psnr_arr(mask_nbr, im_nbr) = psnr_val;
    imwrite(im2uint8(abs(im_rec)), fullfile('plots_more_samp', sprintf('rec_splg1_mask_%d.png', mask_nbr)));
    %end
end

save(fullfile(dest, 'psnr_values_cs.mat'), 'psnr_arr');


