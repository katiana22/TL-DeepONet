close all
clear
load bc_source
global ubc

num = 2000; % Number of samples

for i = 1:num
    
    ubc = f_bc(i, :);
    
    model = createpde('structural','static-planestrain');
    radius = 0.25;
    
    R1 = [3, 4,0,1,1,0,0,0,1,1]';
    C1 = [1,0.75,0.75,0.1]';
    C1 = [C1;zeros(length(R1) - length(C1),1)];
    C2 = [1,0.25,0.25,0.1]';
    C2 = [C2;zeros(length(R1) - length(C2),1)];
    gm = [R1,C1,C2];
    sf = 'R1-C1-C2';
    ns = char('R1','C1','C2');
    ns = ns';
    g = decsg(gm,sf,ns);
    
    geometryFromEdges(model,g);
    %     figure
    pdegplot(model,'EdgeLabel','on');
    %     title 'Geometry with Edge Labels';
    %
        %figure
        %pdegplot(model,'VertexLabels','on');
        %title 'Geometry with Vertex Labels';
    structuralProperties(model,'YoungsModulus',410.0E3,'PoissonsRatio',0.35);
    
    structuralBC(model,'Edge',3,'XDisplacement',0, 'YDisplacement',0);
    
    structuralBoundaryLoad(model,'Edge',1,'SurfaceTraction',@myload, 'Vectorized', 'on');
    
    generateMesh(model,'Hmax',radius/4);
    R = solve(model);
    
    X = R.Mesh.Nodes;
    X = X';
    xx = X(:, 1);
    yy = X(:, 2);
    intrpDisp = interpolateDisplacement(R,xx,yy);

    ux(i,:) = reshape(intrpDisp.ux,size(xx));
    uy(i,:) = reshape(intrpDisp.uy,size(yy));
    %quiver(xx,yy,ux,uy)
    %ux(i,:) = ut(:,1);
    %uy(i,:) = ut(:,2);
    
    %figure
    pdeplot(model,'XYData',ux(i,:),'ColorMap','jet')
    %axis equal
    %title 'Displacement Along x-Direction';
    
    stress_x(i,:) = R.Stress.sxx;
    stress_y(i,:) = R.Stress.syy;
        %figure
        %pdeplot(model,'XYData',R.Stress.sxx,'ColorMap','jet')
        %axis equal
        %title 'Normal Stress Along x-Direction';
    %
    %     figure
    %     pdeplot(model,'XYData',R.Stress.syy,'ColorMap','jet')
    %     axis equal
    %     title 'Normal Stress Along y-Direction';
    %
    %     thetaHole = linspace(0,2*pi,200);
    %     xr = radius*cos(thetaHole);
    %     yr = radius*sin(thetaHole);
    %     CircleCoordinates = [xr;yr];
    %
    %     stressHole = interpolateStress(R,CircleCoordinates);
    %
    %     figure
    %     plot(thetaHole,stressHole.sxx)
    %     xlabel('\theta')
    %     ylabel('\sigma_{xx}')
    %     title 'Normal Stress Around Circular Boundary';
end

ux_train = ux(1:1900,:);
ux_test = ux(1901:2000,:);
uy_train = uy(1:1900,:);
uy_test = uy(1901:2000,:);
stressX_train = stress_x(1:1900,:);
stressX_test = stress_x(1901:2000,:);
stressY_train = stress_y(1:1900,:);
stressY_test = stress_y(1901:2000,:);
f_bc_train = f_bc(1:1900,:);
f_bc_test = f_bc(1901:2000,:);

save('Dataset_2Circle', 'f_bc_train', 'f_bc_test', 'xx', 'yy', 'ux_train', ...
    'ux_test', 'uy_train', 'uy_test', 'stressX_train', 'stressX_test', 'stressY_train', 'stressY_test');
