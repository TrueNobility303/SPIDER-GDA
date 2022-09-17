% calculate the condition number related to PL condition of matrix A * A'
function [lambda_max,lambda_min,k] = condition_number(A)
    tol = 1.0e-10;
    v = svd(A);
    lambda_max = v(1)^2;
    lambda_min = 0;
    for  i =length(v):-1:1
        % ignore the zero eigen values for PL functions
        if(v(i)^2 > tol)
           lambda_min = v(i)^2;
           break;
        end
    end
    k  = lambda_max / lambda_min;
end