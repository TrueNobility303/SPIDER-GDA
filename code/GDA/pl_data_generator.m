function [data] = pl_data_generator(n, d, cond, r)
% Data generator for PL game
%
% Inputs:
%       n               number of samples.
%       d               number of dimensions.
%       cond            condition number;
% Output:
%       data            data set

        UA = orth(randn(d,d));
        UB = orth(randn(d,d));
        UC = randn(d,d);
        
        D = diag([linspace(1/cond, 1,r) zeros(1,d-r)]);
        SigmaA = UA * D * UA';
        SigmaB = UB * D * UB';
        SigmaC = 0.1 *  UC * UC';
        A =  mvnrnd(zeros(d,1), SigmaA,n)';
        B =  mvnrnd(zeros(d,1), SigmaB,n)';
        C =  mvnrnd(zeros(d,1), SigmaC,n)';
        
        data.A = A;
        data.B = B;
        data.C = C;
        
        % if we set e = f =0, then the saddle point of this problem is (0,0)
        data.e = zeros(d,1);
        data.f =  zeros(d,1);
        
        %initialization
        data.w_init = randn(d,1);
        data.z_init = randn(d,1);
end

