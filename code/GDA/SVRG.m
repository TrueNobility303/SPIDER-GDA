function [w,z, infos] = SVRG(problem, options)
% SVRG-GDA
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options  options
% Output:
%       w,z         solution of w,z
%       infos       information
%
% References:
%       Rie Johnson and Tong Zhang, 
%       "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction,"
%       NIPS, 2013.
%    
% This file modified from part of algorithm SVRG from SGDLibrary
%  "https://github.com/hiroyuki-kasai/SGDLibrary"

    % set dimensions and samples
    n = problem.samples();
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;   
    w = options.w_init;
    z = options.z_init;
    
    num_of_batchces = floor(n / options.batch_size);  
   
    % store first infos
    clear infos;    
    [infos, f_val, gnorm] = record_infos(problem, w,z, [], epoch, grad_calc_count, options); 
    
    % display infos
    if options.verbose > 0
        fprintf('SVRG: Epoch = %03d, cost = %.4e, gnorm = %.4e\n', epoch, f_val, gnorm);
    end      
    
    % main loop
    while (gnorm > options.tol_gnorm) && (epoch < options.max_epoch)

        % permute samples
        if options.permute_on
            perm_idx_w = randperm(n);
            perm_idx_z = randperm(n);
        else
            perm_idx_w = 1:n;
            perm_idx_z = 1:n;
        end

        % compute full gradient
        [full_grad_w,full_grad_z] = problem.full_grad(w,z);
        
        % store w, z
        w0 = w;
        z0 = z;
        grad_calc_count = grad_calc_count + n;
        
        for j = 1 : num_of_batchces
            
            % calculate variance reduced gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_w = perm_idx_w(start_index:start_index+options.batch_size-1);
            indice_z = perm_idx_z(start_index:start_index+options.batch_size-1);
            
            [grad_w,grad_z]   = problem.grad(w,z, indice_w, indice_z);
            [grad_w0,grad_z0] = problem.grad(w0, z0, indice_w, indice_z);
            
            % update w,z
            v_w = full_grad_w + grad_w - grad_w0;
            w = w - options.step_w * v_w;
           
            v_z = full_grad_z + grad_z - grad_z0;
            z = z + options.step_z * v_z;
            
            total_iter = total_iter + 1;
                 
        end
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * options.batch_size;        
        epoch = epoch + 1;
         
        % store infos
        [infos, f_val, gnorm] = record_infos(problem, w, z, infos, epoch, grad_calc_count, options);           

        % display infos
        if options.verbose > 0
            fprintf('SVRG: Epoch = %03d, cost = %.4e, gnorm = %.4e\n', epoch, f_val, gnorm);
        end
    end
    
    if gnorm < options.tol_gnorm
        fprintf('Optimality gnorm tolerance reached: gnorm = %g\n', options.tol_gnorm);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', options.max_epoch);
    end
      
end

