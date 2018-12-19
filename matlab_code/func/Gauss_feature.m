classdef Gauss_feature < handle

    % Separate two components
    % phaseFunction
    % baseFunction
    

      
   properties
      basis;
      phase;
      nDemo =[];
      nBasis;
      weight;
      control;
      nTrajSteps;
   end
   
        
    
methods

    function obj = Gauss_feature(basesLocation, basesWidth, phase_dt, nDemo, nTrajSteps)
        obj.basis   = Bases(basesLocation, basesWidth);
        obj.phase   = Phase(phase_dt);
        obj.weight  = Weight();
        obj.control = Control();
        
        
        obj.weight.my_linRegRidgeFactor = 1e-8*eye(size(basesLocation,2),...
                                                 size(basesLocation,2) );    
                                
                                
        obj.nBasis = size(basesLocation, 2);
        obj.nDemo = nDemo;
        obj.nTrajSteps = nTrajSteps;
    end

   function obj = make_nOrderGaussian(obj)
    % Original implementation from class system in vectorized form

    
        
        basisCenter = obj.basis.location;
        sigma = obj.basis.width;
        
        z = obj.phase.z;
        zd = obj.phase.zd;
        zdd = obj.phase.zdd;
        

        %> (z - basisCenter)
        z_minus_center = bsxfun(@minus, z, basisCenter );


        at = bsxfun(@times, z_minus_center, 1./sigma);

        % computing and normalizing basis (order 0)
        basis     = bsxfun(@times, exp( -0.5 * at.^2 ), 1./sigma/sqrt(2*pi) );
        basis_sum = sum(basis,2);
        basis_n   = bsxfun(@times, basis, 1 ./ basis_sum);

        z_minus_center_sigma = bsxfun(@times, -z_minus_center, 1./(sigma.^2) );

        %> derivative of the basis
        basisD     =  z_minus_center_sigma .* basis; % closed form

        % normalizing basisD
        basisD_sum = sum(basisD,2);
        basisD_n_a = bsxfun(@times, basisD, basis_sum);
        basisD_n_b = bsxfun(@times, basis, basisD_sum);
        basisD_n   = bsxfun(@times, basisD_n_a - basisD_n_b, 1 ./(basis_sum.^2) );


        %> second derivative of the basis
        tmp =  bsxfun(@times,basis, -1./(sigma.^2) );
        basisDD = tmp + z_minus_center_sigma .* basisD;
        basisDD_sum = sum(basisDD,2);

        % normalizing basisDD
        basisDD_n_a  = bsxfun(@times, basisDD, basis_sum.^2);
        basisDD_n_b1 = bsxfun(@times, basisD, basis_sum);
        basisDD_n_b  = bsxfun(@times, basisDD_n_b1, basisD_sum);
        basisDD_n_c1 =  2 * basisD_sum.^2 - basis_sum .* basisDD_sum;
        basisDD_n_c  = bsxfun(@times, basis,  basisDD_n_c1);
        basisDD_n_d  = basisDD_n_a - 2 .* basisDD_n_b + basisDD_n_c;                                              
        basisDD_n    = bsxfun(@times, basisDD_n_d, 1 ./ basis_sum.^3);


        % This is extra code that can be found in 
        % +TrajectoryGenerators/NormalizedGaussiansBasisGenerator
        % I do not know why it is used, and it seems Gdd_n is not
        % correct
        basisDD_n = bsxfun(@times, basisDD_n, zd.^2) + bsxfun(@times, basisD_n, zdd );
        basisD_n =  bsxfun(@times, basisD_n,  zd);

        
        obj.basis.Gn    = basis_n;
        obj.basis.Gn_d  = basisD_n;
        obj.basis.Gn_dd = basisDD_n;

        % This is useful to understand and compare
        if 0 % Get only 0 order Gaussian without vectorizing

            % This is mainly to understand simple Gaussian implementation


            % normalized Gaussian computation
            G = [];
            nBasis = length(basisCenter);
            for k = 1:nBasis
                G1 = 1/(2*pi*sigma(k)^2 )^0.5;
                G2 = exp( -1/(2* sigma(k)^2 ) *( z-basisCenter(k) ).^2 );

                singleG = G1.*G2;
                G = [G  singleG];
            end

            % normalize 2: normalize across all basis on the same time step
            % This is the method used by the ClassSystem.
            sum2 = sum(G, 2);
            Gn2 = [];
            for k = 1:nBasis
                Gn2 = [Gn2    G(:,k)./sum2];
            end

            % This is the correct method: Gn2 where you normalize accross all
            % bases at the same time step (normalize the supperposition of them)
            G_n = Gn2; 

        end


        if 0 % check derivatives here

            keyboard 

            db.basisDD =[];
            % computing derivative numerically
            for k = 1:size(basisD_n,2)
                dd_ = diff(basisD_n(:,k))./diff(z);
                dd_ = [dd_; dd_(end)];
                db.basisDD= [db.basisDD dd_];
            end

            % Weird result: why Gdd is not symmetric??
            figurew;
            plot(basisDD_n, SGRAY(.7))
            plot(db.basisDD, 'b')
            title('2nd derivative of Gaussian');

        end    


    end % n_ordergaussian
    
    
    function obj = learn_weights(obj, demoq)
        
        obj.weight.demoq = demoq;
        dbg = 0;
        
        Gn = obj.basis.Gn;
        my_linRegRidgeFactor = obj.weight.my_linRegRidgeFactor ;
        
        % First get the weights of each trajectory
        w = [];
        for k = 1:obj.nDemo

            % find weights by normal solution with Moore-Penrose Pseudo-Inverse
            MPPI = (Gn'*Gn  + my_linRegRidgeFactor) \ Gn';
            w_   = MPPI*demoq{k}(:,1);
            w = [w ; w_'];

            if dbg % reproduce each trajectory
                figurew(['learned weights ', num2str(k) ]);
                title 'Comparing trajectory reproduction by weights'
                plot(demoq{k}(:,1), SBLUEBALLW(10));
                plot(w_'*Gn', SREDBALL(5));
            end
        end
        obj.weight.w = w;
        

        obj.weight.mean = mean(w)';
        obj.weight.cov  = cov(w);
        

        % better covariance
        [obj.weight.cov, obj.weight.covchol] = ...
             post_processing_covariance(obj.weight.cov, obj.nBasis, obj.nDemo);

  
         
         
         
         
    end % learn weights
    

    
        function [phase, phaseD, phaseDD] = generatePhase(obj)
            phase = obj.doGeneratePhase();
            
            dt = obj.phaseTimeStep;
            phaseD = diff(phase)/dt;
            phaseD = [phaseD; phaseD(end)];
            phaseDD = diff(phaseD)/dt;
            phaseDD = [phaseDD; phaseDD(end)];

        end
        
        
    
end %methods


end
