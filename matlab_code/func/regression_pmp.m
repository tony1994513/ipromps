function pmp = regression_pmp(data, nBasis, basisType)
% Create the ProMP.
% Currently the only basisType is gaussian.
%
    nJoints = size(data{1},2);

    switch basisType

        case 'UnitFeatures'
    
            % Do primitives without gaussian bases, such that each weight is
            % the value of the data itself. Not efficient, but easy to
            % understand.
            error('to do');

        case 'GaussianFeatures'
            % Generate the gaussian bases

            nDemo = size(data,1);
            nTraj = size(data{1},1); % length of each demonstration

            % locate the center of each base. Currently they are equaly spaced
            % between the normalized time [0 1], but we will have to change this in
            % the future.
            mu_location = linspace(0,1,nBasis); 

            dt     = 1/nTraj;    % such that z = [dt 2dt 3dt. . . 1], where [1 x nTraj]
            phase  = Phase(dt);  % replaces the phase

            % create empty object
            weight = Weight(nBasis, nJoints, nTraj, nDemo);


            % this is noise to be added in the matrix diagonal such that
            % inversion is stable
            weight.my_linRegRidgeFactor = 1e-8*eye(size(mu_location,2),...
                                                   size(mu_location,2) );   

            sigma = 0.05*ones(1,nBasis); % kernel width

            [basis.Gn, basis.Gndot, basis.Gnddot] = generateGaussianBasis( phase, mu_location, sigma);

        otherwise
            error('UnitFeature or GaussianFeatures');

    end
    
    % Compute the vector of weights of each trajectory w1, w2, ... wn.
    % Compute the mean and covariance of all weights
    weight.least_square_on_weights(basis.Gn, data);
    
    
    pmp.phase   = phase;
    pmp.w       = weight;
    pmp.basis   = basis;
    pmp.nBasis  = nBasis;
    pmp.nJoints = nJoints;
    pmp.nDemo   = nDemo;
    pmp.nTraj   = nTraj;
    
   
    
end



