clc;
clear;
close all;
rng(42);

%% a script for tuning the parameters 

%% generate synthetic data        
n = 6000;
d = 10;
r = 5;
cond = 1e9;
% generate data
data = pl_data_generator(n, d, cond,r);
        
%% define problem definitions
problem = pl_game(data.A, data.B, data.C,data.e, data.f); 
[L,mu] = problem.cond_one_side();
options = get_default_options();
fprintf("PL game with n = %d, d= %d, L = %.e, mu = %.e\n",n,d, L,mu);

options.w_init = data.w_init;
options.z_init = data.z_init;   
options.batch_size = 1;
options.regular = 0;
options.tol_gnorm = 1e-6;
options.max_epoch = 100;
options.batch_size = 1;
step_lst = [1e-2, 1e-3, 1e-4, 1e-5];

%% perform SVRG 
info_svrg.iter = [options.max_epoch];
info_svrg.step_w = 1;
info_svrg.best_gnorm = inf;

for step_w = step_lst
    for step_z = step_lst
        fprintf('testing SVRG with step w: %f, step z: %f \n',step_w,step_z);
        options.step_w = step_w;
        options.step_z = step_z;

        [~,~, info] = SVRG(problem, options);
        if info.gnorm(end) < options.tol_gnorm && info.iter(end) < info_svrg.iter(end)
           info_svrg = info; 
        end
        if info.iter(end) == info_svrg.iter(end) && info.best_gnorm < info_svrg.best_gnorm
            info_svrg = info;
        end
    end
end

%% perform SPIDER 

info_spider.iter = [options.max_epoch];
info_spider.step_w = 1;
info_spider.best_gnorm = inf;

for step_w = step_lst
    for step_z = step_lst
        fprintf('testing SPIDER with step w : %f \n',step_w);
        options.step_w = step_w;
        options.step_z = step_z;
        [~,~, info] = SPIDER(problem, options);

        if info.gnorm(end) < options.tol_gnorm && info.iter(end) < info_spider.iter(end)
           info_spider = info; 
        end
        if info.iter(end) == info_spider.iter(end) && info.best_gnorm < info_spider.best_gnorm
            info_spider = info;
        end
    end
end


%% plot
clf;
semilogy(info_svrg.grad_calc_count, info_svrg.gnorm,'LineWidth',3,'LineStyle','--');
hold on;
semilogy(info_spider.grad_calc_count, info_spider.gnorm,'LineWidth',3,'LineStyle',':');
hold on;
legend('SVRG','SPIDER','FontSize',20);
xlabel('#grad/n');
ylabel('gradient norm');
set(gca,'FontSize',20);

%% results
fprintf('============================\n');
fprintf('============================\n');
fprintf('====SVRG===== step_w : %.e, step_z: %.e \n',info_svrg.step_w, info_svrg.step_z);
fprintf('====SPIDER=== step_w : %.e, step_z: %.e \n',info_spider.step_w, info_spider.step_z);
fprintf("PL game with n = %d, d= %d, r=%d, L = %.e, mu = %.e\n",n,d, r, L,mu);

%% the results of tuned parameters for SVRG and SPIDER
% ====SVRG===== step_w : 1e-03, step_z: 1e-02 
% ====SPIDER=== step_w : 1e-03, step_z: 1e-02 
% PL game with n = 6000, d= 10, r=5, L = 1e+00, mu = 1e-09
