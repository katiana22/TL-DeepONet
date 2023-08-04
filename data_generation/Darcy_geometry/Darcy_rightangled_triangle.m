%2D Darcy flow------------------------------------------------------------
%-\nabla \cdot (K \nabla h) = 0-------------------------------------------
%pde toolbox: https://www.mathworks.com/help/pde/ug/equations-you-can-solve.html

clear all
close all

global per;
load K_field;

K_field = K_field';
N = 100;
num = 2100; % number of samples

x = linspace(0, 1, N);
y = linspace(0, 1, N);

[xx, yy] = meshgrid(x, y);

u = zeros(num,1200);
for i = 1:num
    i
    per = K_field(i, :);
    per= reshape(per, [N, N]);
   
    %------PDE solver------------------------------------------------------
    model = createpde(1);
    R1 = [3, 4, 0, 0.5, 0.5+1e-12, 0.5-1e-12, 0, 0, sqrt(3)/2, sqrt(3)/2]';
    gm = [R1];
    
    sf = 'R1';
    ns = char('R1');
    ns = ns';
    g = decsg(gm,sf,ns);
    
    geometryFromEdges(model,g);
    
   pdegplot(model,'EdgeLabels','on')
    % Boundary conditions
    applyBoundaryCondition(model,'dirichlet','Edge',1, 'u', 0);
    applyBoundaryCondition(model,'dirichlet','Edge',2, 'u', 0);
    applyBoundaryCondition(model,'dirichlet','Edge',3, 'u', 0);
    applyBoundaryCondition(model,'dirichlet','Edge',4, 'u', 0);
    
    % See more details from pde toolbox tutorial
    specifyCoefficients(model,'m',0,...
        'd',0,...
        'c',@ccoeffunction,...
        'a',0,...
        'f',1);
    
    hmax = 0.03;
    generateMesh(model,'Hmax',hmax);
    pdeplot(model)
    axis equal
    axis off
    results = solvepde(model);
    ut = results.NodalSolution;
    %sol = interpolateSolution(results, xx, yy);
    %ut = reshape(sol, size(xx));
    %     figure;
    %     scatter(xx,yy,[],ut)
    %     imagesc(ut); colormap(jet); axis equal;
    
    u(i,:) = ut;

    %     pdeplot(model,'XYData',results.NodalSolution); colormap(jet); axis equal;
    
end

X = results.Mesh.Nodes;
X = X';
xx = X(:, 1);
yy = X(:, 2);
    
u_train = u(1:2000,:);
u_test = u(2001:2100,:);

load Coeff;
k_train = coeff(1:2000,:,:);
k_test = coeff(2001:2100,:,:);

save("Dataset.mat","xx","yy","k_test","k_train","u_test","u_train")
