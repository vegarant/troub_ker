
ell_50_weights_path = '/local/scratch/public/va304/storage3/FBPConvNet';

load(ell_50_weights_path);

net = vl_simplenn_move(net, 'cpu');
gpuDevice(1); % Clear GPU
net = vl_simplenn_move(net, 'gpu');

N = 512;
fsize = 11;

src = '/local/scratch/public/va304/storage3/FBPConvNet';
dest  = 'plots_bird';
if (exist(dest) ~= 7) 
    mkdir(dest);
end
res = 75;
nbr = 4;
im1 = imread(fullfile(src, sprintf('ellipses4_bird_%d_siam%d.png', res, nbr)));
im = double(im1)/255;

ma =  150;
mi = -227;

Y = (ma-mi)*im + mi;

Y = 1.1*Y;
irad = @(I, ang) iradon(I, ang, 'linear', 'Ram-Lak', 1, N);

f  = @(net, x0)  hand_f_FBP (net, x0);

nbr_lines = 1000;
theta = linspace(0, 180*(1-1/nbr_lines), nbr_lines);

subs = 1:20:nbr_lines; 

theta1 = theta(subs);

rec = irad(radon(Y, theta1), theta1);

fx = f(net, rec);

fname1 = fullfile(dest, ...
        sprintf('FBPConvNet_bird_%d_siam_%d_org.png', res, nbr));
fname2 = fullfile(dest, ...
        sprintf('FBPConvNet_bird_%d_siam_%d_rec.png', res, nbr));
imwrite(im1, parula(256), fname1);
imwrite(im2uint8(scale_to_01(fx)), parula(256), fname2);



