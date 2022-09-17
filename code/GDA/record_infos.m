function [infos, f_val, gnorm] = record_infos(problem, w, z, infos, epoch, grad_calc_count, options)
% Function to store statistic information
%
% Inputs:
%       problem         function (cost/grad/dist)
%       w,z             solution 
%       infos           struct to store statistic information
%       epoch           number of outer iteration
%       grad_calc_count number of calclations of gradients
%       options         options
% Output:
%       infos           updated struct to store statistic information
%       f_val           cost function value
%       gnorm           norm of gradient
% This file is modifed from SGDLibrary.

    if ~epoch
        
        infos.step_w = options.step_w;
        infos.step_z = options.step_z;
        infos.iter = epoch; 
        infos.grad_calc_count = grad_calc_count;
        f_val = problem.cost(w,z);
        
        % calculate norm of full gradient
        gnorm = problem.grad_norm(w,z);
        infos.cost = [f_val];
        infos.best_cost = f_val;
        infos.gnorm = [gnorm];
        infos.best_gnorm = gnorm;
        
        dist = norm([w; z]);
        infos.dist = [dist];
        infos.best_dist = dist;
    else
        
        infos.iter = [infos.iter epoch];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        
        f_val = problem.cost(w,z);
        gnorm = problem.grad_norm(w,z);
       
        infos.cost = [infos.cost f_val];
        if f_val < infos.best_cost
            infos.best_cost = f_val;
        end
        
        infos.gnorm = [infos.gnorm gnorm]; 
        if gnorm < infos.best_gnorm
            infos.best_gnorm = gnorm;
        end
        
        % distance to (0,0)
        dist = norm([w;z]);
        infos.dist = [infos.dist dist]; 
        if dist < infos.best_dist
            infos.best_dist = dist;
        end
    end

end

