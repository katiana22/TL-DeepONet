clear;
clc;
close all

curDir = pwd;
folder_name = [curDir '/Predictions_right_Target'];
mkdir(folder_name)

flag_plot = true;

% -------------------------------------------
% This is the unstructured mesh for the original data
% -------------------------------------------
model = createpde(1);
R1 = [3, 4, 0, 0.5, 0.5+1e-12, 0.5-1e-12, 0, 0, sqrt(3)/2, sqrt(3)/2]'; % Triangular Domain
gm = [R1];
sf = 'R1';
ns = char('R1');
ns = ns';
g = decsg(gm,sf,ns);
geometryFromEdges(model,g);
hmax = 0.03;
generateMesh(model,'Hmax',hmax);

% -------------------------------------------
% 100 testing samples
N_data = 50;
resultFileName = ['Darcy_right_target.mat'];
Result = load(resultFileName);
% to store the l2 errors
L2_mesh = zeros(N_data,1);

if flag_plot
    figure('color',[1,1,1],'Units',...
        'centimeters','Position',[10,5,30,15]);
end

load Dataset_right_triangle.mat
x_grid = linspace(0,1,100);
y_grid = linspace(0,1,100);
[x_grid,y_grid]  = meshgrid(x_grid,y_grid);

% -------------------------------------------
% loop for all the testing examples
for idx = 1:N_data
    
    % -------------------------------------------
    % load the original data - unstructured mesh
    u_truth = reshape(Result.y_test(idx,:,:),[],1);    % Nx1
    
    u_pred = double(squeeze(Result.y_pred(idx,:,:) ) )';
    L2_mesh(idx) = norm(u_truth-u_pred) / norm(u_truth);
    
    k_grid = reshape(Result.x_test(idx,:,:),[],1);    % Nx1
    
    func_data = scatteredInterpolant(x_grid(:),y_grid(:),k_grid,'linear','nearest');
    vq = func_data(xx,yy);
    
    % -------------------------------------------
    % display (on the unstructured mesh)
    if flag_plot
        
        subplot(141);
        pdeplot(model,'XYData',vq);
        colormap(jet); %caxis(clim);
        axis equal; xlim([0,0.5]); ylim([0,1])
        title('Conductivity','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        box on;
        
        clim_min = min(min(u_pred),min(u_truth));
        clim_max = max(max(u_pred),max(u_truth));
        clim = [clim_min, clim_max];
        subplot(142);
        pdeplot(model,'XYData',u_truth);
        colormap(jet); caxis(clim);
        axis equal;
        xlim([0,0.5]); ylim([0,1])
        title('Pressure: Truth','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        box on;
        
        subplot(143);
        pdeplot(model,'XYData',u_pred);
        colormap(jet); caxis(clim);
        axis equal;
        xlim([0,0.5]); ylim([0,1])
        title('Pressure: DeepONet','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        box on;
        
        subplot(144);
        %         clim = [0, 0.1];
        pdeplot(model,'XYData',abs(u_truth-u_pred) );
        colormap(jet); %caxis(clim);
        axis equal;
        xlim([0,0.5]); ylim([0,1])
        title('Error','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        box on;
        
        saveas(gcf, [folder_name '/TestCase', num2str(idx),'.png'])
    end
end

% -------------------------------------------
% mean l2 error of unstructured mesh
fprintf('L2 norm error: %.3e\n', mean(L2_mesh));