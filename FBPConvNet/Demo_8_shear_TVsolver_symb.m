
dest = 'plots_bird';
if (exist(dest) ~= 7)                                                           
    mkdir(dest);                                                                
end 

nbr_lines = 50;
theta = linspace(0,180*(1-1/nbr_lines), nbr_lines);

res = 75;
src = 'data';
nbr = 2;
im = double(imread(fullfile(src, sprintf('ellipses4_bird_%d_siam%d.png', res, nbr))));

load('~/storage/radon_matrices/radonMatrix2N512_ang50.mat');
theta = linspace(0,180*(1-1/nbr_lines), nbr_lines);                            

N = 512;                                                                        
m = [N,N];

z = scale_to_01(im);
f = 0.05*(im/255);

y = A*f(:);    

%% set up operator
B.times     = @(x) A*x;
B.adj       = @(x) A'*x;
B.mask      = NaN;

pm.sparse_param = [0, 0, 1, 1];
pm.sparse_trans = 'shearlets';
D = getShearletOperator([N,N], pm.sparse_param);
% set parameters
pm.alpha     = 0;
pm.beta      = 5e1;
pm.mu        = [0, 5e2];
pm.lambda    = 3e-4;
pm.maxIter   = 50;
pm.epsilon   = 1e-8;
pm.adaptive  = 'NewIRL1';
pm.normalize = false;
pm.solver    = 'TVsolver';

doPlot = false;
doReport = true;

out  = TVsolver(y, [N,N], B, D, pm.alpha, pm.beta, pm.mu(1), pm.mu(2), ...
                'lambda', pm.lambda, ...
                'adaptive', pm.adaptive, ...
                'f', f, ...
                'doPlot', doPlot, ...
                'doReport', doReport, ...
                'maxIter', pm.maxIter, ...
                'epsilon', pm.epsilon);

rec = out.rec;

fname = fullfile(dest, ...                                                      
        sprintf('CS_rec_bird_%d_siam%d.png', res, nbr));

imwrite(im2uint8(scale_to_01(rec)), parula(256), fname);



