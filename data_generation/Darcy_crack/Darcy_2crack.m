%2D Darcy flow------------------------------------------------------------
%-\nabla \cdot (K \nabla h) = 0-------------------------------------------
%pde toolbox: https://www.mathworks.com/help/pde/ug/equations-you-can-solve.html

clear all
close all

global per;
K_filename = "KLE/K_field.mat";
load(K_filename)

K_field = K_field';
N = 100;
num = 2100; % Number of samples

notch_bottom_1 = 0.7950;
notch_top_1 = 0.8050;  % thickness of the crack 5e-3

notch_bottom_2 = 0.3950;
notch_top_2 = 0.4050;  % thickness of the crack 5e-3

for i = 1:num
    
    per = K_field(i, :);
    per= reshape(per, [N, N]);
    
    %     f = @(location, state) 5*location.x;
    %------PDE solver------------------------------------------------------
    model = createpde(1);
    R1 = [3, 4, 0, 1, 1, 0, 0, 0, 1, 1]'; % Square Domain
    R2 = [3, 4, 0, 0.5, 0.5, 0, notch_bottom_1, notch_bottom_1, notch_top_1, notch_top_1]';
    R2 = [R2;zeros(length(R1)-length(R2),1)];
    R3 = [3, 4, 0.5, 1, 1, 0.5, notch_bottom_2, notch_bottom_2, notch_top_2, notch_top_2]';
    R3 = [R3;zeros(length(R1)-length(R3),1)];
    gm = [R1,R2,R3];
    
    sf = 'R1-R2-R3';
    ns = char('R1','R2','R3');
    ns = ns';
    g = decsg(gm,sf,ns);
   
    geometryFromEdges(model,g);
    pdegplot(model,'EdgeLabels','on')
    
    % Boundary conditions
    applyBoundaryCondition(model,'dirichlet','Edge',1:12, 'u', 0);
    
    % See more details from pde toolbox tutorial
    specifyCoefficients(model,'m',0,...
        'd',0,...
        'c',@ccoeffunction,...
        'a',0,...
        'f',10);
    
    generateMesh(model);
    % pdeplot(model)
    results = solvepde(model);
    X = results.Mesh.Nodes;
    X = X';
    xx = X(:, 1);
    yy = X(:, 2);
    
    sol = interpolateSolution(results, xx, yy);
    ut = reshape(sol, size(xx));
    ut = results.NodalSolution;
    u(i,:) = ut;
    
    pdeplot(model,'XYData',results.NodalSolution); colormap(jet); axis equal;
end


u_train = u(1:2000,:,:);
u_test = u(2001:2100,:,:);

Coeff_filename = "KLE/Coeff.mat";
load(Coeff_filename)
k_train = coeff(1:2000,:,:);
k_test = coeff(2001:2100,:,:);
save("Dataset_2crack.mat","xx","yy","k_test","k_train","u_test","u_train")
