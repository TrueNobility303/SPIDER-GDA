clc;
clear;
close all;
rng(1340);

%% generate synthetic data        
n = 6000;
d = 10;
r = 5;
cond = 1e9;
% generate data
data = pl_data_generator(n, d, cond,r);
        
%% define problem definitions
problem = pl_game(data.A, data.B, data.C, data.e, data.f); 
[L,mu] = problem.cond();
options = get_default_options();

fprintf("PL game with n = %d, d= %d, r = %d, L = %.e, mu = %.e\n",n,d,r, L,mu);

options.w_init = data.w_init;
options.z_init = data.z_init;   
options.batch_size = 1;
options.max_epoch = 100;
options.tol_gnorm = 2e-7;

%% perform SVRG 
options.step_w = 1e-3;
options.step_z = 1e-2;
[~,~, info_svrg] = SVRG(problem, options);


%% perform SPIDER
[~,~, info_spider] = SPIDER(problem, options);

%% peerform AccSPIDER
options.regular = L/(10*n);
[~,~, info_acc_spider]  = AccSPIDER(problem, options);

%% plot
clf;
semilogy(info_svrg.grad_calc_count, info_svrg.gnorm,'LineWidth',3,'LineStyle','--');
hold on;
semilogy(info_spider.grad_calc_count, info_spider.gnorm,'LineWidth',3,'LineStyle',':');
hold on;
semilogy(info_acc_spider.grad_calc_count, info_acc_spider.gnorm,'LineWidth',3);
legend('SVRG','SPIDER','AccSPIDER','FontSize',20);
xlabel('#SFO');
str = '$$\Vert \nabla f(x,y) \Vert^2  $$';
ylabel(str,'Interpreter','latex');
ylim([8e-6,1]);
set(gca,'FontSize',20);

