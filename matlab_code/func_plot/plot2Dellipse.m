function h = plot2Dellipse(h, joint_KF)



    % Define the elements of the covariance matrix.

    sigmaA =  3;
    sigmaB =  1;
    sigmaC =  0.01; % this defines the direction of the eigenvector

    figurew('ellipse1'); 
    loop(sigmaA, sigmaB, sigmaC);
    title('Covariance 0, eigenvector aligned with XY axes');


    sigmaC = 1.2;
    figurew('ellipse2'); 
    loop(sigmaA, sigmaB, sigmaC);
    title('Covariance nonzero, eigenvector not-aligned with XY axes');


end


function [] = loop(sigmaA, sigmaB, sigmaC)


    A = (sigmaA^2);  % variance 1st variable
    B = (sigmaB^2);  % variance 2nd variable
    C = (sigmaC^2);  % covariance


    % check if ellipse is possible (or they will become a parabola or hyperbola)

    % maximum value for C_sigma
    Cmax       = sqrt(A*B);
    Cmax_sigma = sqrt(Cmax);
    disp(['Max value for C_sigma: ', num2str(Cmax_sigma)]);

    if (A*B-C^2) <= 0 % check the determinant
        error('Ellipse not possible');
    end


    % Covariance matrix and ellipse plot

    E = [A  C; C B];
    plot_ellipse(E);

    
    % Eigenvector and eigenvalue

    % compute eigen value and vector
    [eigenVector, lambda] = eig(E);

    % get ellipse radius
    radius = sqrt(diag(lambda));

    % eigenVector are given as column vectors
    vector1 = eigenVector(:,1)*radius(1);
    vector2 = eigenVector(:,2)*radius(2);

    plot([0   vector1(1)], [0   vector1(2)],   SRED(3));
    plot([0   vector2(1)], [0   vector2(2)],   SGREEN(3));
    axis 'equal'
    
   

end



