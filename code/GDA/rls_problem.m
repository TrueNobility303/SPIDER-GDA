classdef rls_problem
% This file defines RLS(robust linear regression) problem class
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       lambda      l2-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min max f(w,z) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w,z) = 1/2 * (w' * x_i - y_i)^2 - lambda/2 * (z_i -y_i)^2.
%
% "w" is the model parameter of size dx1 vector
% "z" is the model parameter of size 1xn vector

    properties
        name;    
        dim;
        samples;
        lambda;
        n_train;
        n_test;
        x_train;
        y_train;
        x_test;
        y_test;
        x;         
    end
    
    methods
        function obj = rls_problem(x_train, y_train, x_test, y_test) 
            
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;            
        
            obj.dim = size(x_train, 1);
            obj.n_train = length(y_train);
            obj.n_test = length(y_test);      

            obj.name = 'RLS';    
            obj.samples = obj.n_train;
            obj.x = x_train;    
            obj.lambda = 2;
        end

    
        function f = cost(obj, w, z )

            f = sum((w'*obj.x_train-z).^2 - obj.lambda * (z - obj.y_train).^2 )/ (2 * obj.n_train) ;

        end

        function [gw,gz] = grad(obj, w, z, indices_w, indices_z)
            
            residual_w = w'*obj.x_train(:,indices_w)-z(indices_w);
            gw = obj.x_train(:,indices_w) * residual_w'/length(indices_w);
            
            residual_z = w'*obj.x_train(:,indices_z)-z(indices_z);
            n = length(z);
            gz = zeros(1,n);
            gz(indices_z) = - (residual_z + obj.lambda * ( z(indices_z) - obj.y_train(indices_z) )) / length(indices_z); 
        
        end

        function [gw,gz] = full_grad(obj, w,z)
            
            [gw,gz] = obj.grad(w,z, 1:obj.n_train, 1:obj.n_train);
        end
        
        function [L,mu] = cond(obj)
           mu_y  = 1;
           [L, mu_x, ~] = condition_number(obj.x_train);
           mu = min(mu_x,mu_y);
        end
        
        function val = grad_norm(obj,w,z)
           [gw,gz] = obj.full_grad(w,z);
           val = norm([gw',gz],'fro');
        end
        
        function val = stoc_grad_norm(obj,w,z,indices_w, indices_z)
            [gw,gz] = obj.grad(w,z,indices_w, indices_z);
            val = norm([gw',gz],'fro');
        end
    end
end