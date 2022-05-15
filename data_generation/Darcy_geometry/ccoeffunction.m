%K_field used in the equation---------------------------------------------
%@Author: Xuhui Meng---------------------------------------------------
function cmatrix = ccoeffunction(location, state)
N = 100;

x = linspace(0, 1, N);

y = linspace(0, 1, N);

[X, Y] = meshgrid(x, y);

% load K;

% K = ones(N, N);

global per;

k = interp2(X, Y, per, location.x, location.y);

n1 = 2;

nr = numel(location.x);

cmatrix = zeros(n1,nr);

cmatrix(1,:) = k.*ones(1,nr);

cmatrix(2,:) = cmatrix(1,:);
end