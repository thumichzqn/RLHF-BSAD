clear all
close all
clc
%%
len = 400;
samples = 100;

loader = load('Proposed_16.mat', 'value_record');
traj_proposed_16 = loader.value_record(1:len, 1:samples);
mean_proposed_16 = mean(traj_proposed_16');
ci_proposed_16 = bootci(1000, @mean, traj_proposed_16');

loader = load('PEPSQ_16_1.mat', 'value_record');
traj_PEPSQ_16_1 = loader.value_record(1:len, :);
mean_PEPSQ_16_1 = mean(traj_PEPSQ_16_1');
ci_PEPSQ_16_1 = bootci(1000, @mean, traj_PEPSQ_16_1');

loader = load('PEPSQ_16_2.mat', 'value_record');
traj_PEPSQ_16_2 = loader.value_record(1:len, :);
mean_PEPSQ_16_2 = mean(traj_PEPSQ_16_2');
ci_PEPSQ_16_2 = bootci(1000, @mean, traj_PEPSQ_16_2');

loader = load('PEPSQ_16_3.mat', 'value_record');
traj_PEPSQ_16_3 = loader.value_record(1:len, :);
mean_PEPSQ_16_3 = mean(traj_PEPSQ_16_3');
ci_PEPSQ_16_3 = bootci(1000, @mean, traj_PEPSQ_16_3');

%%
x_tick = 1:50:50*len;
figure;
hold on;

set(gca, 'Ylim',[0.4,1.6])
set(gca, 'Xlim',[0,8000])
xlabel('Number of Episodes Sampled','FontName', 'Arial', 'FontSize', 16)
ylabel('Reward','FontName', 'Arial', 'FontSize', 16)
% means

% proposed batch
plot(x_tick, mean_proposed_16,'r', 'LineWidth', 1.8);
plot(x_tick, mean_PEPSQ_16_1,'--g', 'LineWidth', 1.8);
plot(x_tick, mean_PEPSQ_16_2,'--b', 'LineWidth', 1.8);
plot(x_tick, mean_PEPSQ_16_3,'--m', 'LineWidth', 1.8);

% CIs


fill([x_tick, x_tick(end:-1:1)], ...
    [ci_proposed_16(1,:), ci_proposed_16(2,end:-1:1)], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');   


% fill([x_tick, x_tick(end:-1:1)], ...
%     [ci_PEPSQ_1(1,:), ci_PEPSQ_1(2,end:-1:1)], ...
%     'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 

% fill([x_tick, x_tick(end:-1:1)], ...
%     [ci_PEPSQ_16(1,:), ci_PEPSQ_16(2,end:-1:1)], ...
%     'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');  
% 
% fill([x_tick, x_tick(end:-1:1)], ...
%     [ci_PEPSQ_16_3(1,:), ci_PEPSQ_16_3(2,end:-1:1)], ...
%     'm', 'FaceAlpha', 0.1, 'EdgeColor', 'none');  



legend('BSAD-16','PEPS-16-1K', 'PEPS-16-2K', 'PEPS-16-3K', 'location','southeast', 'FontSize', 14)
%%
