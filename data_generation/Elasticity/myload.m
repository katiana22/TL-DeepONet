function bcMatrix = myload(location, state)

bcMatrix = zeros(2,length(location.x));
global ubc
N = 101;
X = linspace(-1, 1, N);

load_y = interp1(X, ubc, location.y);

bcMatrix(1,:) = 1e3*load_y;
end