function [basis_n, basisD_n, basisDD_n] = generateBasis( phase, mu, sigma)


    %> phase - mu, element by element of mu
    time_mu = bsxfun(@minus, phase, mu );
    for kk = 1:length(mu) %centering distrib at mu
        my_time_mu(:, kk) = phase - mu(kk); 
    end
    if sum( time_mu - my_time_mu ) ~= 0
        error ('sanity check!!!');
    end

    at = bsxfun(@times, time_mu, 1./sigma);

    basis = bsxfun(@times, exp( -0.5 * at.^2 ), 1./sigma/sqrt(2*pi) );

    basis_sum = sum(basis,2);

    basis_n = bsxfun(@times, basis, 1 ./ basis_sum);


    time_mu_sigma = bsxfun(@times, -time_mu, 1./(sigma.^2) );


    %> derivative of the basis
    basisD =  time_mu_sigma .* basis;

    basisD_sum = sum(basisD,2);

    basisD_n_a = bsxfun(@times, basisD, basis_sum);
    basisD_n_b = bsxfun(@times, basis, basisD_sum);
    basisD_n = bsxfun(@times, basisD_n_a - basisD_n_b, 1 ./(basis_sum.^2) );

    tmp =  bsxfun(@times,basis, -1./(sigma.^2) );
    basisDD = tmp + time_mu_sigma .* basisD;
    basisDD_sum = sum(basisDD,2);


    basisDD_n_a = bsxfun(@times, basisDD, basis_sum.^2);
    basisDD_n_b1 = bsxfun(@times, basisD, basis_sum);
    basisDD_n_b = bsxfun(@times, basisDD_n_b1, basisD_sum);

    basisDD_n_c1 =  2 * basisD_sum.^2 - basis_sum .* basisDD_sum;
    basisDD_n_c = bsxfun(@times, basis,  basisDD_n_c1);

    basisDD_n_d = basisDD_n_a - 2 .* basisDD_n_b + basisDD_n_c;                                              

    basisDD_n = bsxfun(@times, basisDD_n_d, 1 ./ basis_sum.^3);
end











