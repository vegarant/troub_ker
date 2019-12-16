dwtmode('per', 'nodisp');

automap_config; % sets src_mri_data and src_k_mask  

load(fullfile(src_mri_data, 'k_mask.mat')); % k_mask
load(fullfile(src_mri_data, 'Automap_data.mat')); % x, rr
load(fullfile(src_mri_data, 'random_pert.mat'));
x = double(x);
rr = double(rr);
rand_pert = double(rand_pert);

plot_dest = 'plots';

idx = find(k_mask);

vm = 2; % number of vanishing moments
wname = sprintf('db%d', vm);
noise = 0.1
N = 128;

batch_size = size(x, 1);
for i = 1:batch_size
    im = squeeze(x(i, :, :) + rr(i, :, :) + rand_pert(i, :, :));
    b = fftshift(fft2(im))./N;
    b = b(idx);

    nres = wmaxlev(N, wname);

    opA = @(x, mode) csl_op_fourier_wavelet_2d(x, mode, N, idx, nres, wname);

    opts = spgSetParms('verbosity', 1); 
    wave_coeff = spg_bpdn(opA, b, noise, opts);

    s = csl_get_wavedec2_s(round(log2(N)), nres);

    im_rec  = waverec2(wave_coeff, s, wname);
    im_rec = (im_rec - min(im_rec(:)))/(max(im_rec(:)) - min(im_rec(:)));
    im_rec = abs(im_rec);
    fname = sprintf('cs_rec_pert_%d.png', i-1);
    imwrite(im2uint8(im_rec), fullfile(plot_dest, fname));
end
