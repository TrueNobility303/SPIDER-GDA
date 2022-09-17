classdef pl_game
% This file defines PL game problem class
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min max f(w,z) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w,z) = 1/2 * (w' * a_i * a_i' *w) - 1/2 * (z' * b_i * b_i' * z) + w'* c_i * c_i' *z 
%
% "w" is the model parameter of size dx1 vector
% "z" is the model parameter of size dx1 vector

    properties
        name;    
        dim;
        samples;
        A;
        B;
        C;         
        e;
        f;
    end
    
    methods
        function obj = pl_game(A,B,C,e,f) 
            
            [obj.dim,obj.samples] = size(A);

            obj.name = 'PL game';    
            
            obj.A = A;    
            obj.B = B;
            obj.C = C;
            obj.e = e;
            obj.f = f;
        end

    
        function f = cost(obj, w, z )

            f = w'*obj.A * obj.A' * w / 2 - z' * obj.B * obj.B' *z /2  + w' * obj.C * obj.C'  *z + obj.e'* w + obj.f'*z; 

        end
        
        function [gw,gz] = grad(obj, w, z, indices_w, indices_z)
            
            a_i = obj.A(:,indices_w);
            c_i = obj.C(:,indices_w);
            gw = (a_i * (a_i' * w) + c_i * (c_i' * z)) / length(indices_w) + obj.e;
            
            b_i = obj.B(:,indices_z);
            c_i = obj.C(:,indices_z);
            gz = (-b_i * (b_i' * z) + c_i * (c_i' * w)) / length(indices_z) + obj.f;
        end

        function [gw,gz] = full_grad(obj, w,z)
            
            [gw,gz] = obj.grad(w,z, 1:obj.samples, 1:obj.samples);
          
        end
        
        function [L,mu] = cond(obj)
           [L_x, mu_x, ~] = condition_number(obj.A  / sqrt(obj.samples) );
           [L_y, mu_y, ~] = condition_number(obj.B  / sqrt(obj.samples) );
           mu = min(mu_x,mu_y);
           L  = max(L_x,L_y);
        end
        
        function[L,mu]  = cond_one_side(obj)
            [L, mu, ~] = condition_number(obj.A  / sqrt(obj.samples) );
        end
        
        function val = grad_norm(obj,w,z)
           [gw,gz] = obj.full_grad(w,z);
           val = norm([gw; gz],'fro');
        end
        
        function val = stoc_grad_norm(obj,w,z,indices_w, indices_z)
            [gw,gz] = obj.grad(w,z,indices_w, indices_z);
            val = norm([gw; gz],'fro');
        end
    end
end