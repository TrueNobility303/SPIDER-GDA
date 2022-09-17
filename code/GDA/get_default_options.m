function options = get_default_options()
    
    options.tol_gnorm       = 1.0e-2; 
    options.batch_size      = 1;
    options.max_epoch       = 100;
    options.permute_on      = 1;
    options.verbose         = 2;
    
    options.step_w          = 0.01; 
    options.step_z          = 0.1; 
    
    % paramters for Catalyst
    options.regular            = 0; 
    options.sub_iterations     = 1;
    options.momemtum_parameter = 0.999;
end

