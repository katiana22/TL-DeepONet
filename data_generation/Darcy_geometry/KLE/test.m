clear,clc, close all
rng('default');
H = 1; % vertical scale
L = 1; % horizontal scale
m = 100; % rows
n = 100; % columns
x = linspace(0,L,n);
y = linspace(0,H,m);
[X,Y] = meshgrid(x,y);
mesh = [X(:) Y(:)]; % 2-D mesh
N = 2100;   %number of samples
corr.name = 'spher'; % available patterns: gauss, exp, s_exp, spher and turbulent
% 'spher' : for V-DeepONet
corr.c0 = [0.25 0.25]; % [corLx/L,corLy/H]
% corr.c1 = [0.02 0.02]; % anisotropic correlation, only used for turbulent
corr.sigma = 1;
kl_term = 100; 

[Fi, F, KL] = randomfield(corr,mesh, 'trunc', kl_term);

meanty = 0;
ty = meanty + Fi*randn(kl_term, N);
K = exp(ty);
K_field = K;
save('K_field.mat', 'K_field');
% save Gaussian_lnK_64x64_10K_Channel.txt -ascii ty

test_id = randi(N, 1);
test_id = 10;
F = log(K(:,test_id));
% plot the realization
figure;
surf(X,Y,reshape(F,m,n)); view(2); colorbar;
contourf(reshape(F,m,n),20,'linestyle','none')
colormap(jet)
colorbar

coeff = zeros(N,m,n);
for i = 1:N
    
    coeff(i,:,:) = reshape(K(:,i),m,n);
end

save('Coeff.mat', 'coeff');
