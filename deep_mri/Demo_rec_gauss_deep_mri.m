deep_mri_config;
plot_fldr = 'plots_gaus_stat';

N = 256;
place = 1;
factor = 3;
trails = 20;

load(fullfile(src_mri_data, 'x0.mat'))
load(fullfile(src_mri_data, 'mask.mat'))

load(fullfile(plot_fldr, sprintf('worst_case_data_fact_%g_itr_%d.mat', factor, trails))); % array_of_worst_raw_noise size([5,2,N,N])

clear array_of_psnr_values
clear array_of_worst_noisy_images
clear array_of_worst_psnr
clear array_of_worst_pred

r = array_of_worst_raw_noise;
im_nbr = 1;
x0 = double(squeeze(x0(im_nbr, :,:)));
r  = double(squeeze(r(place+1, im_nbr, :,:)));

mask = double(mask);
mask = logical(fftshift(squeeze(mask(im_nbr,:,:))));


m = [N,N];

ma = max(x0(:));
mi = min(x0(:));

x0 = (x0 - mi)/(ma-mi);
r  = (r - mi)/(ma-mi);

% fetch Operators
A = getFourierOperator([N,N], mask);
% D = getWaveletOperator(m,2,3);
pm.sparse_param = [1, 1, 2];
pm.sparse_trans = 'shearlets';
pm.solver    = 'TGVsolver';
D = getShearletOperator([N,N], pm.sparse_param);

% fetch measurement operator
y       = A.times(x0(:));
yr       = A.times(x0(:)+r(:));

    % set parameters
pm.beta        = 1e5;
pm.alpha       = [1 1];
pm.mu          = [5e3, 1e1, 2e1];
pm.epsilon     = 1e-5;

pm.maxIter     = 500;
pm.adaptive    = 'NewIRL1';
%correct     = @(x) real(x);
doTrack     = true;
doPlot      = false;



%%% solve
%outy = TGVsolver(y, [N,N], A, D, pm.alpha, pm.beta, pm.mu, ...
%                'maxIter',  pm.maxIter, ...
%                'adaptive', pm.adaptive, ...
%                'f',        abs(x0), ...
%                'epsilon',  pm.epsilon, ...
%                'doTrack',  doTrack, ...
%                'doPlot',   doPlot);
%
%im1 = abs(outy.rec);
%
%fname1 = fullfile(location_of_folder, plot_fldr, ...
%                 sprintf('rec_cs_x_run_%d.png', runner_id));
%
%imwrite(im2uint8(im1), fname1);

outyr = TGVsolver(yr, [N,N], A, D, pm.alpha, pm.beta, pm.mu, ...
                'maxIter',  pm.maxIter, ...
                'adaptive', pm.adaptive, ...
                'f',        abs(x0+r), ...
                'epsilon',  pm.epsilon, ...
                'doTrack',  doTrack, ...
                'doPlot',   doPlot);

fname2 = fullfile(plot_fldr, ...
                  sprintf('cs_rec_fact_%g_trails_%d_itr_%d_place_%d.png', factor, trails, pm.maxIter, place));

im2 = abs(outyr.rec);

imwrite(im2uint8(im2), fname2);








