function [data] = rls_data_generator(n, d, std)
% Data generator for linear regression problem.
%
% Inputs:
%       n               number of samples.
%       d               number of dimensions.
%       std             standard deviation.
% Output:
%       data            data set
%       data.x_train    train data of x of size d x n.
%       data.y_train    train data of y of size 1 x n.
%       data.x_test     test data of x of size d x n.
%       data.y_test     test data of y of size 1 x n.
%       data.w_opt      solution.
%
% This file is modified from SGDLibrary.

    % true
    %w_opt = randn(d, 1);  
    w_opt = 0.5 * ones(d+1, 1);
    data.w_opt = w_opt;    

    % train data
    x_tmp = randn(1,2*n);
    x = repmat(x_tmp, [d 1]);     
    % add intercept term to x
    x = [x; ones(1, 2*n)];      
    
    %normalize the data
    %x = normalize(x);
    
    y = w_opt' * x;
    % add noise
    noise = randn(1, 2*n);
    y = y + std * noise;
    
    % shuffle data
    perm_idx = randperm(2*n); 
    
    % split data into train and test data
    % train data
    train_indices = perm_idx(1:n); 
    data.x_train = x(:,train_indices); 
    data.y_train = y(train_indices); 

    % test data    
    test_indices = perm_idx(n+1:end);
    data.x_test = x(:,test_indices); 
    data.y_test = y(test_indices);    
    
    % add noise
    noise = randn(1, n);
    data.z_init = data.y_train +  0.0001 * noise;
    
    noise = randn(d+1,1);
    data.w_init = data.w_opt +  noise;
end

