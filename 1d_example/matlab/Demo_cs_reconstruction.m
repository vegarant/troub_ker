% This scritps reads a spesific poly heav and preform a compressive sensing 
% reconstruction using wavelets, of the poly heave both with and without 
% a tumor. The final reconstructions are saved at fullfile(dest, cs_rec.mat)
% so that the result can be read by a python script to produce the final 
% picture

clear('all'); close('all');
dwtmode('per', 'nodisp');

dest = 'plots'
% Create destination for the plots
if (exist(dest) ~= 7) 
    mkdir(dest);
end

r = 9;
N = 2^r;
vm = 4;
degree = 3;
discon = 3;
disp_plots = 'off';
load('../data/sample_idx.mat');
load('../data/tumor.mat');
data_src = '../data/poly_heavs';

wname = sprintf('db%d', vm);
nres = wmaxlev(N, wname)-1;
S = get_wavedec_s(r, nres);

% Poly heav number
ph_nbr = 3;

ph_fname = sprintf('ph_deg_%d_disc_%d_nr_%04d.mat', degree, discon, ph_nbr);
fname = fullfile(data_src, ph_fname);
ph = load_poly_heav(fname);


%poly_struct

t = linspace(0,1,N)';
x = ph.function_handle(t);
xz = x + tumor;

noise=0.01;
[gx, z] = sample_fourier_wavelet_1d(x, noise, idx, vm);
[gxz, z] = sample_fourier_wavelet_1d(xz, noise, idx, vm);


fig = figure('visible', disp_plots);

ymax = max(max(gx), max(x));                       
ymin = min(min(gx), min(x));                       
                                                                           
ylim_bottom = ymin - 0.1*sign(ymin)*ymin;                               
ylim_top = 1.1*ymax;                                                       

plot(t,x, 'linewidth', 2);
hold('on');
plot(t,gx, 'linewidth', 2);

ylim([ylim_bottom, ylim_top]);
legend({'x', 'reconstruction'}, 'fontsize', 14)
                                                    
saveas(fig, fullfile(dest, 'rec_cs_no_pert.png'));

fig = figure('visible', disp_plots);

ymax = max(max(gxz), max(xz));                       
ymin = min(min(gxz), min(xz));                       
                                                                           
ylim_bottom = ymin - 0.1*sign(ymin)*ymin;                               
ylim_top = 1.1*ymax;                                                       

plot(t, xz, 'linewidth', 2);
hold('on');
plot(t,gxz, 'linewidth', 2);

ylim([ylim_bottom, ylim_top]);
legend({'x', 'reconstruction'}, 'fontsize', 14)
saveas(fig, 'rec_cs_pert.png');

save(fullfile(dest, 'cs_rec.mat'), 't', 'x', 'xz', 'gx', 'gxz');

