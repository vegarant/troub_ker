dwtmode('per', 'nodisp');
N = 128;

automap_config; % sets src_mri_data and src_k_mask  

load(fullfile(src_mri_data, 'HCP_mgh_1003_T2_selected_and_resized.mat'))
new_im = double(imread(fullfile(src_mri_data, 'brain1_128_anonymous.png')));

%im = double(im);
batch_size = size(im, 1);
im_array = zeros([size(im,1)+1, N, N]);

im_array(1,:,:) = new_im;

for i = 1:batch_size
    im_array(i+1,:,:) = im(i,:,:);
end
batch_size = size(im_array, 1);

load(fullfile(src_mri_data, 'k_mask.mat')); % k_mask
plot_dest = 'plots';

idx = find(k_mask);

vm = 2; % number of vanishing moments
wname = sprintf('db%d', vm);
noise = 0.1


for i = 1:batch_size
    single_im = squeeze(im_array(i, :, :));
    b = fftshift(fft2(single_im))./N;
    b = b(idx);

    nres = wmaxlev(N, wname);

    opA = @(x, mode) csl_op_fourier_wavelet_2d(x, mode, N, idx, nres, wname);

    opts = spgSetParms('verbosity', 1); 
    wave_coeff = spg_bpdn(opA, b, noise, opts);

    s = csl_get_wavedec2_s(round(log2(N)), nres);

    im_rec = waverec2(wave_coeff, s, wname);
    im_rec = (im_rec - min(im_rec(:)))/(max(im_rec(:)) - min(im_rec(:)));
    im_rec = abs(im_rec);

    fname = sprintf('new_im_cs_rec_%d.png', i-1);
    imwrite(im2uint8(im_rec), fullfile(plot_dest, fname));

end



