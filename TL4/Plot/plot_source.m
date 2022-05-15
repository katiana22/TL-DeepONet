clear;
clc;
close all

curDir = pwd;
folder_name = [curDir '/Predictions_Source'];
if exist(folder_name,'dir')
    rmdir(folder_name,'s');
end
mkdir(folder_name);

flag_plot = true;

% -------------------------------------------
% This is the unstructured mesh for the original data
% -------------------------------------------
model = createpde('structural','static-planestrain');
radius = 0.25;

R1 = [3, 4,0,1,1,0,0,0,1,1]';
C1 = [1,0.5,0.5,0.25]';
C1 = [C1;zeros(length(R1) - length(C1),1)];
gm = [R1,C1];
sf = 'R1-C1';
ns = char('R1','C1');
ns = ns';
g = decsg(gm,sf,ns);
geometryFromEdges(model,g);
generateMesh(model,'Hmax',radius/4);

% -------------------------------------------
% 100 testing samples
N_data = 100;
resultFileName = ['Elastic_source.mat'];
Result = load(resultFileName);
% to store the l2 errors
L2_mesh = zeros(N_data,2);

% -------------------------------------------
% loop for all the testing examples
x_bc = linspace(1,100,101);

scrsz = get(groot, 'ScreenSize');
hFig  = figure('Position',[1 scrsz(4)/6 3*scrsz(3)/5 1*scrsz(4)/4]);

for idx = 1:N_data
    
    % -------------------------------------------
    % load the original data - unstructured mesh
    ux_truth = reshape(Result.ux_test(idx,:),[],1);    % Nx1
    ux_pred = reshape(Result.ux_pred(idx,:),[],1);    % Nx1

    uy_truth = reshape(Result.uy_test(idx,:),[],1);    % Nx1
    uy_pred = reshape(Result.uy_pred(idx,:),[],1);    % Nx1
    
    L2_mesh(idx,1) = norm(ux_truth-ux_pred) / norm(ux_truth);
    L2_mesh(idx,2) = norm(uy_truth-uy_pred) / norm(uy_truth);
    
    % -------------------------------------------
    % display (on the unstructured mesh)
    if flag_plot
        
        sph = subplot(2,4,[1,5]);
        plot(x_bc, Result.x_test(idx,:)*1e3, 'k-', 'linewidth', 2.0);
        xlim([0,101]); ylim([-2.5,1.5]); axis tight
        xlabel('$x$', 'interpreter', 'latex', 'fontweight','bold','fontsize', 14);
        ylabel('$h$', 'interpreter', 'latex', 'fontweight','bold','fontsize', 14);
        dx0 = -0.02; dy0 = 0.25; dwithx = 0.0; dwithy = -0.5;
        set(sph,'position',get(sph,'position')+[dx0,dy0,dwithx,dwithy])
        box on;
        
        clim_min = min(min(ux_pred),min(ux_truth));
        clim_max = max(max(ux_pred),max(ux_truth));
        clim = [clim_min, clim_max];
        
        subplot(2,4,2);
        pdeplot(model,'XYData',ux_truth);
        colormap(jet); %caxis(clim);
        axis equal;
        xlim([0,1]); ylim([0,1])
        title('X-Disp:Truth','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        
        subplot(2,4,3);
        pdeplot(model,'XYData',ux_pred);
        colormap(jet); %caxis(clim);
        axis equal;
        xlim([0,1]); ylim([0,1])
        title('X-Disp:Pred','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        
        subplot(2,4,4);
        pdeplot(model,'XYData',abs(ux_truth-ux_pred) );
        colormap(jet); axis equal;
        xlim([0,1]); ylim([0,1])
        title('Error','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        box on;
        
        clim_min = min(min(uy_pred),min(uy_truth));
        clim_max = max(max(uy_pred),max(uy_truth));
        clim = [clim_min, clim_max];
        
        subplot(2,4,6);
        pdeplot(model,'XYData',uy_truth);
        colormap(jet); %caxis(clim);
        axis equal;
        xlim([0,1]); ylim([0,1])
        title('Y-Disp:Truth','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        
        subplot(2,4,7);
        pdeplot(model,'XYData',uy_pred);
        colormap(jet); %caxis(clim);
        axis equal;
        xlim([0,1]); ylim([0,1])
        title('Y-Disp:Pred','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        
        subplot(2,4,8);
        pdeplot(model,'XYData',abs(uy_truth-uy_pred) );
        colormap(jet);
        axis equal;
        xlim([0,1]); ylim([0,1])
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