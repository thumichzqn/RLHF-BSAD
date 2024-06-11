clear all
close all
clc
%%
len = 400;
samples = 100;

loader = load('Proposed_2.mat', 'value_record');
traj_proposed_2 = loader.value_record(1:len, 1:samples);
mean_proposed_2 = mean(traj_proposed_2');
ci_proposed_2 = bootci(1000, @mean, traj_proposed_2');

loader = load('Proposed_4.mat', 'value_record');
traj_proposed_4 = loader.value_record(1:len, 1:samples);
mean_proposed_4 = mean(traj_proposed_4');
ci_proposed_4 = bootci(1000, @mean, traj_proposed_4');

loader = load('Proposed_8.mat', 'value_record');
traj_proposed_8 = loader.value_record(1:len, 1:samples);
mean_proposed_8 = mean(traj_proposed_8');
ci_proposed_8 = bootci(1000, @mean, traj_proposed_8');

loader = load('Proposed_16.mat', 'value_record');
traj_proposed_16 = loader.value_record(1:len, 1:samples);
mean_proposed_16 = mean(traj_proposed_16');
ci_proposed_16 = bootci(1000, @mean, traj_proposed_16');

loader = load('Proposed_32.mat', 'value_record');
traj_proposed_32 = loader.value_record(1:len, 1:samples);
mean_proposed_32 = mean(traj_proposed_32');
ci_proposed_32 = bootci(1000, @mean, traj_proposed_32');

loader = load('PEPSQ_1.mat', 'value_record');
traj_PEPSQ_1 = loader.value_record(1:len, :);
mean_PEPSQ_1 = mean(traj_PEPSQ_1');
ci_PEPSQ_1 = bootci(1000, @mean, traj_PEPSQ_1');

loader = load('PEPSQ_16.mat', 'value_record');
traj_PEPSQ_16 = loader.value_record(1:len, :);
mean_PEPSQ_16 = mean(traj_PEPSQ_16');
ci_PEPSQ_16 = bootci(1000, @mean, traj_PEPSQ_16');

loader = load('PEPSQ_16_3.mat', 'value_record');
traj_PEPSQ_16_3 = loader.value_record(1:len, :);
mean_PEPSQ_16_3 = mean(traj_PEPSQ_16_3');
ci_PEPSQ_16_3 = bootci(1000, @mean, traj_PEPSQ_16_3');

%%
x_tick = 1:50:50*len;
figure;
hold on;

set(gca, 'Ylim',[0.4,1.6])
set(gca, 'Xlim',[0,6000])
xlabel('Number of Episodes Sampled','FontName', 'Arial', 'FontSize', 16)
ylabel('Reward','FontName', 'Arial', 'FontSize', 16)
% means

% proposed batch
plot(x_tick, mean_proposed_2,'k', 'LineWidth', 1.8);
plot(x_tick, mean_proposed_4,'b', 'LineWidth', 1.8);
plot(x_tick, mean_proposed_8,'m', 'LineWidth', 1.8);
plot(x_tick, mean_proposed_16,'g', 'LineWidth', 1.8);
plot(x_tick, mean_proposed_32,'r', 'LineWidth', 1.8);
% plot(x_tick, mean_PEPSQ_1,'--b', 'LineWidth', 1.6);
% plot(x_tick, mean_PEPSQ_16,'--m', 'LineWidth', 1.6);
% plot(x_tick, mean_PEPSQ_16_3,'--g', 'LineWidth', 1.6);

% CIs
fill([x_tick, x_tick(end:-1:1)], ...
    [ci_proposed_2(1,:), ci_proposed_2(2,end:-1:1)], ...
    'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); 

fill([x_tick, x_tick(end:-1:1)], ...
    [ci_proposed_4(1,:), ci_proposed_4(2,end:-1:1)], ...
    'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); 

fill([x_tick, x_tick(end:-1:1)], ...
    [ci_proposed_8(1,:), ci_proposed_8(2,end:-1:1)], ...
    'm', 'FaceAlpha', 0.2, 'EdgeColor', 'none');   

fill([x_tick, x_tick(end:-1:1)], ...
    [ci_proposed_16(1,:), ci_proposed_16(2,end:-1:1)], ...
    'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');   

fill([x_tick, x_tick(end:-1:1)], ...
    [ci_proposed_32(1,:), ci_proposed_32(2,end:-1:1)], ...
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



legend('BSAD-2','BSAD-4','BSAD-8','BSAD-16','BSAD-32','location','southeast','FontSize', 14)
%%



