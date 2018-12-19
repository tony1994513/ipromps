function [basis_n, basisD_n, basisDD_n] = generateGaussianBasis( phase, mu, sigma)

    basisCenter = mu;

    z   = phase.z;
    zd  = phase.zd;
    zdd = phase.zdd;


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


end