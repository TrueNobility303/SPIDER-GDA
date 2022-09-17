function [w,z, infos] = SPIDER(problem, options)
% SPIDER-GDA
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options  options
% Output:
%       w,z           solution of w
%       infos       information
%
% References:
%       L. M. Nguyen, J. Liu, K. Scheinberg, and M. Takac, 
%       "SARAH: A novel method for machine learning problems using stochastic recursive gradient,"
%       ICML, 2017.
%    
% This file is modified from algorithm sarah from
% "SGDLibrary: https://github.com/hiroyuki-kasai/SGDLibrary"

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
        fprintf('SPIDER: Epoch = %03d, cost = %.4e, gnorm = %.4e\n', epoch, f_val, gnorm);
    end     

    % main loop
    while (gnorm > options.tol_gnorm) && (epoch < options.max_epoch)

        % permute samples, SPIDER can use the same indexes on w,z
        if options.permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end

        % compute full gradient
        [v0_w,v0_z] = problem.full_grad(w,z);
        grad_calc_count = grad_calc_count + n;        
        
        % update w with full gradient
        w = w - options.step_w * v0_w;
        z = z + options.step_z * v0_z;
        
        v_w = v0_w;
        v_z = v0_z;
        
        w_prev = w;
        z_prev = z;
        
        for j = 1 : num_of_batchces
                       
            % calculate variance reduced gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            [grad_w,grad_z] = problem.grad(w, z, indice_j, indice_j);
            
            %grad_0 = problem.grad(w0, indice_j);
            [grad_prev_w,grad_prev_z] = problem.grad(w_prev, z_prev, indice_j, indice_j);
            
            % store variable
            w_prev = w;
            z_prev = z;
            
            % update v
            v_w = grad_w - grad_prev_w + v_w;
            v_z = grad_z - grad_prev_z + v_z;
            
            % update w,z
            w = w - options.step_w * v_w;
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
            fprintf('SPIDER: Epoch = %03d, cost = %.4e, gnorm = %.4e\n', epoch, f_val, gnorm);
        end
    end
    
    if gnorm < options.tol_gnorm
        fprintf('Optimality gnorm tolerance reached: gnorm = %g\n', options.tol_gnorm);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', options.max_epoch);
    end
      
end

