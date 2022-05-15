rng('default');

N = 101;

xf = linspace(0, 1, N);

l = 0.12;

len = length(xf);

cov = zeros(len, len);

for i = 1:len
    for j  = 1:len
        cov(i, j) = exp(-0.5*(xf(i) - xf(j)).*(xf(i) - xf(j))/l/l);
    end
end
%
mu = zeros(1, len);
f_bc = mvnrnd(mu, cov, 2000).^2;

for i = 1:len
    plot(xf, f_bc(i, :), 'linewidth', 2.0);
    hold on
end

save('bc_source', 'f_bc');