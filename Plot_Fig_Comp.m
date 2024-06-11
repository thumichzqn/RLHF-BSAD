clear all
close all
clc
%%
len = 200;
samples = 50;

loader = load('Proposed_16.mat', 'value_record');
traj_proposed_16 = loader.value_record(1:len, 1:samples);
mean_proposed_16 = mean(traj_proposed_16');
ci_proposed_16 = bootci(1000, @mean, traj_proposed_16');

loader = load('PEPSQ_1.mat', 'value_record');
traj_PEPSQ_1 = loader.value_record(1:len, 1:samples);
mean_PEPSQ_1 = mean(traj_PEPSQ_1');
ci_PEPSQ_1 = bootci(1000, @mean, traj_PEPSQ_1');

% loader = load('PEPSQ_16.mat', 'value_record');
% traj_PEPSQ_16 = loader.value_record(1:len,:);
% mean_PEPSQ_16 = mean(traj_PEPSQ_16');
% ci_PEPSQ_16 = bootci(1000, @mean, traj_PEPSQ_16');

% loader = load('PEPSQ_16_3.mat', 'value_record');
% traj_PEPSQ_16_3 = loader.value_record(1:len, 1:samples);
% mean_PEPSQ_16_3 = mean(traj_PEPSQ_16_3');
% ci_PEPSQ_16_3 = bootci(1000, @mean, traj_PEPSQ_16_3');

loader = load('Pure_Q.mat', 'value_record');
traj_Q = loader.value_record(1:len, 1: samples);
mean_Q = mean(traj_Q');
ci_Q = bootci(1000, @mean, traj_Q');

loader = load('P2R.mat', 'value_record');
traj_P2R = loader.value_record(1:len, 1:samples);
mean_P2R = mean(traj_P2R');
ci_P2R = bootci(1000, @mean, traj_P2R');

loader = load('Reward_free_MLE.mat', 'value_record');
traj_MLE = loader.value_record(1:len, 1:samples);
mean_MLE = mean(traj_MLE');
ci_MLE = bootci(1000, @mean, traj_MLE');

%%
x_tick = 1:50:50*len;
figure;
set(gca, 'Ylim',[0.4,1.6])
set(gca, 'Xlim',[0,8000])
xlabel('Number of Episodes Sampled','FontName', 'Arial', 'FontSize', 16)
ylabel('Reward','FontName', 'Arial', 'FontSize', 16)
hold on;

% means

% proposed batch
plot(x_tick, mean_proposed_16,'g', 'LineWidth', 1.8);
plot(x_tick, mean_PEPSQ_1,'r', 'LineWidth', 1.8);
% plot(x_tick, mean_PEPSQ_16,'b', 'LineWidth', 1.6);
% plot(x_tick, mean_PEPSQ_16_3,'m', 'LineWidth', 1.6);
plot(x_tick, mean_Q,'b', 'LineWidth', 1.8);
plot(x_tick, mean_P2R,'c', 'LineWidth', 1.8);
plot(x_tick, mean_MLE,'k', 'LineWidth', 1.8);

% CIs
fill([x_tick, x_tick(end:-1:1)], ...
    [ci_proposed_16(1,:), ci_proposed_16(2,end:-1:1)], ...
    'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 

fill([x_tick, x_tick(end:-1:1)], ...
    [ci_PEPSQ_1(1,:), ci_PEPSQ_1(2,end:-1:1)], ...
    'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 

% fill([x_tick, x_tick(end:-1:1)], ...
%     [ci_PEPSQ_16(1,:), ci_PEPSQ_16(2,end:-1:1)], ...
%     'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');  
% 
% fill([x_tick, x_tick(end:-1:1)], ...
%     [ci_PEPSQ_16_3(1,:), ci_PEPSQ_16_3(2,end:-1:1)], ...
%     'm', 'FaceAlpha', 0.1, 'EdgeColor', 'none');  

fill([x_tick, x_tick(end:-1:1)], ...
    [ci_Q(1,:), ci_Q(2,end:-1:1)], ...
    'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');  

fill([x_tick, x_tick(end:-1:1)], ...
    [ci_P2R(1,:), ci_P2R(2,end:-1:1)], ...
    'c', 'FaceAlpha', 0.1, 'EdgeColor', 'none');  

fill([x_tick, x_tick(end:-1:1)], ...
    [ci_MLE(1,:), ci_MLE(2,end:-1:1)], ...
    'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');  




legend('BSAD','PEPS','Q-learning','P2R','REGIME' ,'location','best', 'FontSize', 14)



