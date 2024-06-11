clear all
close all
clc

%% MDP setup
S = 3;          % State Space = {1,2,3}
A = 2;          % Action Space = {1,2}
H = 2;          % Stage = {1,2,3,4}

rng(1)
P = cell(H-1,A);  % Transition Kernel
for h = 1:H-1
    P{h,2} = [0.1, 0.9, 0; 0.1, 0.9, 0; 0.1, 0.9, 0];
    P{h,1} = [0,0,1 ; 0,0,1; 0,0,1];
end
R = cell(H,1);  % Instantaneous Reward
for h = 1:H
    R{h} = zeros(S,A);
end
R{H}(1,2) = 10;
R{H}(2,2) = 0.5;
R{H}(3,2) = 1;

Init = [0.3, 0.4, 0.3];   % Initial Distribution

MDP_Setup = cell(6,1);
MDP_Setup{1} = S;
MDP_Setup{2} = A;
MDP_Setup{3} = H;
MDP_Setup{4} = P;
MDP_Setup{5} = R;
MDP_Setup{6} = Init;

[OptPolicyVI, J, ValueVI, QvalVI] = ValueIteration(MDP_Setup);

%% Our proposed
if 0

% initialization of parameters

logdelta = 0.5;          % log terms
batch_size = 8;        % batch size

% parameters for reward-free exploraion
Q = cell(H,1);          % Q function for reward-free exploration
L = cell(H,1);          % visitation for reward-free exploration
M = cell(H,1);          % target state visitation
for idx = 1:H
    Q{idx} = ones(S,A);
    L{idx} = zeros(S,A);
    M{idx} = zeros(S,1);
end

% parameters for dueling bandits
w = cell(H,S);          % number of wins
N = cell(H,S);          % number of comparisons
sigma = cell(H,S);
for idx = 1:H
    for jdx = 1:S
        w{idx, jdx} = zeros(A,A);
        N{idx, jdx} = zeros(A,A);
        sigma{idx,jdx} = zeros(A,A);
    end
end

policy_candidate = zeros(H,S);      % candidate policy
for h=1:H
    for s=1:S
        policy_candidate(h,s) = randsample(A,1);
    end
end
Dataset0 = cell(H,S);               % datasets for comparison
Dataset1 = cell(H,S);
k=0;
l = H;                              % current step to find optimal policy

% Begin Simulation
a_hat = zeros(S,1);
a_tilde = zeros(S,1);
lcb = cell(H,S);
while l>= 1
    [~, state] = MDP(MDP_Setup, 0, 1, 1);
    k = k+1;
    % reward-free exploration
    for h = 1:l-1
        [~, action] = max(Q{h}(state, :));                              % take action based on Q function
        [~, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
        L{h}(state, action) = L{h}(state, action) + 1;
        t = L{h}(state, action);
        
        if h == l-1
            V_next = min(sqrt(H * logdelta / max(1, M{h+1}(state_next))), 1);                                 % next state value function
        else
            V_next = min(max(Q{h+1}(state_next,:)), 1);                                 % next state value function
        end
        learning_rate = (H + 1) / (H + t);                              % learning rate
        exploration_bonus = sqrt(H * logdelta / max(1,t));                     % exploration bonus
        Q{h}(state, action) = (1 - learning_rate) * Q{h}(state, action) ... 
            + learning_rate * (V_next + 2 * exploration_bonus);     %Q leanring update
        
        state = state_next;                         % transition to next state
    end
    M{l}(state) = M{l}(state) + 1;
    state_find = state;                             % record the state which we want to find best action
    
    % RUCB sub-routine
    if mod(M{l}(state_find) - 1, 2 * batch_size) + 1 <= batch_size
        if mod(M{l}(state_find)-1, batch_size) +1 == 1 || batch_size ==1
            % the first in batch, decide the relaive arm.
            sigma{l,state_find} = w{l,state_find} ./ max(N{l,state_find}, 1);              % updare empirical mean
            ucb = sigma{l,state_find} + sqrt(logdelta ./ max(1,N{l,state_find}));     % update ucb
            action_set = [1:A];
            candidate_action_set = action_set(sum(ucb >= 0.5, 2) == A);         % candidate best action
            
            a_hat(state_find) = randsample(cat(2,candidate_action_set, candidate_action_set),1);
            
            Dataset0{l,state_find} = 0;                                          % clear datasets
            Dataset1{l,state_find} = 0;
        end
        % take candidate best action
        action = a_hat(state);
        [reward, state_next] = MDP(MDP_Setup, l, state, action);        % observe reward and next state
        Dataset0{l,state_find} = Dataset0{l,state_find} + reward;                 % datasets for comparison
        state = state_next;
        
        % simulate the rest trace using candidate optimal policy
        for h =l+1:H
            action = policy_candidate(h, state);                        % use candidate optimal policy
            [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
            Dataset0{l,state_find} = Dataset0{l,state_find} + reward;                 % datasets for comparison
            state = state_next;
        end
    else
        if mod(M{l}(state_find)-1, batch_size)+1 == 1 || batch_size ==1
            % the first in batch, decide the relaive arm.
            sigma{l,state_find} = w{l,state_find} ./ max(N{l,state_find}, 1);              % updare empirical mean
            ucb = sigma{l,state_find} + sqrt(logdelta ./ max(1,N{l,state_find}));     % update ucb
            ucb_for_choose = ucb(:, a_hat(state_find));
            ucb_for_choose(a_hat(state_find)) = -100;                       % avoid compare a_hat to himself
            [~, a_tilde(state_find)] = max(ucb_for_choose);      %ucb arm
            
        end
        % take candidate best action
        action = a_tilde(state);
        [reward, state_next] = MDP(MDP_Setup, l, state, action);        % observe reward and next state
        Dataset1{l,state_find} = Dataset1{l,state_find} + reward;                 % datasets for comparison
        state = state_next;
        
        % simulate the rest trace using candidate optimal policy
        for h =l+1:H
            action = policy_candidate(h, state);                        % use candidate optimal policy
            [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
            Dataset1{l,state_find} = Dataset1{l,state_find} + reward;                 % datasets for comparison
            state = state_next;
        end
    end
    
    % Human comparison
    if mod(M{l}(state_find), 2 * batch_size) == 0
        human_feedback = Dataset0{l,state_find} < Dataset1{l,state_find};
        if Dataset0{l,state_find} == Dataset1{l,state_find}
            human_feedback = randsample([0,1],1);
        end
        
        % update statistics
        w{l, state_find}(a_tilde(state_find), a_hat(state_find)) = ...
            w{l, state_find}(a_tilde(state_find), a_hat(state_find)) + human_feedback; 
        w{l, state_find}(a_hat(state_find), a_tilde(state_find)) = ...
            w{l, state_find}(a_hat(state_find), a_tilde(state_find)) + 1 - human_feedback;
        N{l, state_find}(a_hat(state_find), a_tilde(state_find)) = ... 
            N{l,state_find}(a_hat(state_find), a_tilde(state_find)) + 1;
        N{l, state_find}(a_tilde(state_find), a_hat(state_find)) = N{l, state_find}(a_hat(state_find), a_tilde(state_find));
    end
    
    % stopping:
    flag = zeros(1,S);           % stopping flag
    for s_idx = 1:S
        sigma{l,s_idx} = w{l,s_idx} ./ max(N{l,s_idx}, 1);              % updare empirical mean
        lcb{l, s_idx} = sigma{l,s_idx} - sqrt(logdelta ./ max(1,N{l,s_idx}));     % update lcb
        flag(s_idx) = any((sum(lcb{l, s_idx} >= 0.5, 2) >= A-1) ~= 0);
        %disp([l,flag]);
    end
    if sum(flag) >= S
        for s_idx = 1:S
            action_set = [1:A];
            confidence_action_set = action_set(sum(lcb{l,s_idx} >= 0.5, 2) >= A-1);         % candidate best action
            policy_candidate(l,s_idx) = randsample(cat(2,confidence_action_set, confidence_action_set), 1);
        end
        for idx = 1:H
            Q{idx} = ones(S,A);
            L{idx} = zeros(S,A);
        end
        l = l-1;
    end
end


end

%% Without stopping
if 0

% initialization of parameters

logdelta = 0.5;          % log terms
batch_size = 8;        % batch size
K = 1000;

% parameters for reward-free exploraion
Q = cell(H,1);          % Q function for reward-free exploration
L = cell(H,1);          % visitation for reward-free exploration
M = cell(H,1);          % target state visitation
for idx = 1:H
    Q{idx} = ones(S,A);
    L{idx} = zeros(S,A);
    M{idx} = zeros(S,1);
end

% parameters for dueling bandits
w = cell(H,S);          % number of wins
N = cell(H,S);          % number of comparisons
sigma = cell(H,S);
for idx = 1:H
    for jdx = 1:S
        w{idx, jdx} = zeros(A,A);
        N{idx, jdx} = zeros(A,A);
        sigma{idx,jdx} = zeros(A,A);
    end
end

policy_candidate = zeros(H,S);      % candidate policy
Dataset0 = cell(H,S);               % datasets for comparison
Dataset1 = cell(H,S);
k=0;
l = H;                              % current step to find optimal policy

% Begin Simulation
a_hat = zeros(S,1);
a_tilde = zeros(S,1);
ucb = cell(H,S);

for l = H:-1:1
    for kl = 1:K
        [~, state] = MDP(MDP_Setup, 0, 1, 1);
        k = k+1;
        % reward-free exploration
        for h = 1:l-1
            [~, action] = max(Q{h}(state, :));                              % take action based on Q function
            [~, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
            L{h}(state, action) = L{h}(state, action) + 1;
            t = L{h}(state, action);

            if h == l-1
                V_next = min(sqrt(H * logdelta / max(1, M{h+1}(state_next))), 1);                                 % next state value function
            else
                V_next = min(max(Q{h+1}(state_next,:)), 1);                                 % next state value function
            end
            learning_rate = (H + 1) / (H + t);                              % learning rate
            exploration_bonus = sqrt(H * logdelta / max(1,t));                     % exploration bonus
            Q{h}(state, action) = (1 - learning_rate) * Q{h}(state, action) ... 
                + learning_rate * (V_next + 2 * exploration_bonus);     %Q leanring update

            state = state_next;                         % transition to next state
        end
        M{l}(state) = M{l}(state) + 1;
        state_find = state;                             % record the state which we want to find best action

        % RUCB sub-routine
        if mod(M{l}(state_find) - 1, 2 * batch_size) + 1 <= batch_size
            if mod(M{l}(state_find)-1, batch_size) +1 == 1 || batch_size ==1
                % the first in batch, decide the relaive arm.
                sigma{l,state_find} = w{l,state_find} ./ max(N{l,state_find}, 1);              % updare empirical mean
                ucb{l,state_find} = sigma{l,state_find} + sqrt(logdelta ./ max(1,N{l,state_find}));     % update ucb
                action_set = [1:A];
                candidate_action_set = action_set(sum(ucb{l,state_find} >= 0.5, 2) == A);         % candidate best action

                a_hat(state_find) = randsample(cat(2,candidate_action_set, candidate_action_set),1);

                Dataset0{l,state_find} = 0;                                          % clear datasets
                Dataset1{l,state_find} = 0;
            end
            % take candidate best action
            action = a_hat(state);
            [reward, state_next] = MDP(MDP_Setup, l, state, action);        % observe reward and next state
            Dataset0{l,state_find} = Dataset0{l,state_find} + reward;                 % datasets for comparison
            state = state_next;

            % simulate the rest trace using candidate optimal policy
            for h =l+1:H
                action = policy_candidate(h, state);                        % use candidate optimal policy
                [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
                Dataset0{l,state_find} = Dataset0{l,state_find} + reward;                 % datasets for comparison
                state = state_next;
            end
        else
            if mod(M{l}(state_find)-1, batch_size)+1 == 1 || batch_size ==1
                % the first in batch, decide the relaive arm.
                sigma{l,state_find} = w{l,state_find} ./ max(N{l,state_find}, 1);              % updare empirical mean
                ucb{l,state_find} = sigma{l,state_find} + sqrt(logdelta ./ max(1,N{l,state_find}));     % update ucb
                ucb_for_choose = ucb{l,state_find}(:, a_hat(state_find));
                ucb_for_choose(a_hat(state_find)) = -100;                       % avoid compare a_hat to himself
                [~, a_tilde(state_find)] = max(ucb_for_choose);      %ucb arm

            end
            % take candidate best action
            action = a_tilde(state);
            [reward, state_next] = MDP(MDP_Setup, l, state, action);        % observe reward and next state
            Dataset1{l,state_find} = Dataset1{l,state_find} + reward;                 % datasets for comparison
            state = state_next;

            % simulate the rest trace using candidate optimal policy
            for h =l+1:H
                action = policy_candidate(h, state);                        % use candidate optimal policy
                [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
                Dataset1{l,state_find} = Dataset1{l,state_find} + reward;                 % datasets for comparison
                state = state_next;
            end
        end

        % Human comparison
        if mod(M{l}(state_find), 2 * batch_size) == 0
            human_feedback = Dataset0{l,state_find} < Dataset1{l,state_find};
            if Dataset0{l,state_find} == Dataset1{l,state_find}
                human_feedback = randsample([0,1],1);
            end

            % update statistics
            w{l, state_find}(a_tilde(state_find), a_hat(state_find)) = ...
                w{l, state_find}(a_tilde(state_find), a_hat(state_find)) + human_feedback; 
            w{l, state_find}(a_hat(state_find), a_tilde(state_find)) = ...
                w{l, state_find}(a_hat(state_find), a_tilde(state_find)) + 1 - human_feedback;
            N{l, state_find}(a_hat(state_find), a_tilde(state_find)) = ... 
                N{l,state_find}(a_hat(state_find), a_tilde(state_find)) + 1;
            N{l, state_find}(a_tilde(state_find), a_hat(state_find)) = N{l, state_find}(a_hat(state_find), a_tilde(state_find));
        end

    end

    for s_idx = 1:S
        action_set = [1:A];
        confidence_action_set = action_set(sum(ucb{l,s_idx} >= 0.5, 2) >= A);         % candidate best action
        policy_candidate(l,s_idx) = randsample(cat(2,confidence_action_set, confidence_action_set), 1);
    end
    for idx = 1:H
        Q{idx} = ones(S,A);
        L{idx} = zeros(S,A);
    end

end



end




%% Pure Random + Reward Infer + Q learning

if 0

logdelta = 0.5;          % log terms
K = 20000;
R_hat = cell(H,1);
for h = 1:H
    R_hat{h} = 10*ones(S,A);
end

Q = cell(H,1);          % Q function for reward-free exploration
L = cell(H,1);          % visitation for reward-free exploration
for idx = 1:H
    Q{idx} = 10*ones(S,A);
    L{idx} = zeros(S,A);
end

% Collect traces for reward-inference
human_set = zeros(K/2,1);

traj_set0 = zeros(K/2,H,S,A);
traj_set1 = zeros(K/2,H,S,A);

policy_candidate = zeros(H,S);
rng(2*2017)
for k = 1:K
    [~, state] = MDP(MDP_Setup, 0, 1, 1);
    reward_traj0 = 0;                                                % trajectory reward, not visible by agent
    reward_traj1 = 0;
    for h = 1:H
        action = randsample(A,1);                                       % take random action
        [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
        
        % accumulate trajectory and reward
        if mod(k,2) ==1
            traj_set0((k+1)/2, h, state, action) = 1;
            reward_traj0 = reward_traj0 + reward;
        else
            traj_set0(k/2,h, state, action) = 1;
            reward_traj1 = reward_traj1 + reward;
        end
        
        state = state_next;                         % transition to next state
    end
    
    % human feedback
    if mod(k,2) == 0
        human_feedback = (reward_traj0 > reward_traj1);
        if reward_traj0 == reward_traj1
            human_feedback = randsample([0,1],1);
        end
        
        % record human feedback
        human_set(k/2) = human_feedback;
    end
end

% Reward-model inference
%
theta = ones(H,S,A);
nlog_likelihood = @(theta) -log_likelihood(traj_set0, traj_set1, ...
        theta, human_set(1:k/2));
[theta_opt, fval] = fminunc(nlog_likelihood, theta);
for idx = 1:H
    for s = 1:S
        for a = 1:A
            R_hat{idx}(s,a) = theta_opt(idx,s,a);
        end
    end
end
%
% VI
MDP_Setup = cell(6,1);
MDP_Setup{1} = S;
MDP_Setup{2} = A;
MDP_Setup{3} = H;
MDP_Setup{4} = P;
MDP_Setup{5} = R_hat;
MDP_Setup{6} = Init;

[policy_candidate_VI, ~, ~, Q_hatVI] = ValueIteration(MDP_Setup);
%
% Q learning

Q = cell(H,1);          % Q function for reward-free exploration
L = cell(H,1);          % visitation for reward-free exploration
for idx = 1:H
    Q{idx} = 10*ones(S,A);
    L{idx} = zeros(S,A);
end
for k = 1:5000
    [~, state] = MDP(MDP_Setup, 0, 1, 1);
    reward_traj = 0;                                % trajectory reward, not visible by agent
    for h = 1:H
        [~, action] = max(Q{h}(state, :));                              % take action based on Q function
        [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
        
        L{h}(state, action) = L{h}(state, action) + 1;
        t = L{h}(state, action);
        
        if h == H
            V_next = 0;
        else
            V_next = min(max(Q{h+1}(state_next,:)),10);                                 % next state value function
        end
        
        learning_rate = (H + 1) / (H + t);                              % learning rate
        exploration_bonus = sqrt(H^3 * logdelta / max(1,t));                     % exploration bonus
        
        Q{h}(state, action) = (1 - learning_rate) * Q{h}(state, action) ... 
            + learning_rate * (R_hat{h}(state, action) + V_next + 2 * exploration_bonus);     %Q leanring update
        
        state = state_next;                         % transition to next state
    end
    
    for idx = 1:H
        for s = 1:S
            [~,policy_candidate(idx,s)] = max(Q{idx}(s,:));
        end
    end
end






end


%% P2R + Q learning + MLE

if 0 
% Execute random policy to get baseline trajectory
[~, state] = MDP(MDP_Setup, 0, 1, 1);
reward_tau0 = 0;
traj_tau0 = zeros(H,S,A);
for h = 1:H
    action = randsample(A,1);
    traj_tau0(h, state, action) = 1;
    [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
    reward_tau0 = reward_tau0 + reward;
    state = state_next;
end

logdelta = 1;          % log terms
K = 20000;
R_hat = cell(H,1);
for h = 1:H
    R_hat{h} = 10*ones(S,A);
end
Q = cell(H,1);          % Q function for reward-free exploration
L = cell(H,1);          % visitation for reward-free exploration
for idx = 1:H
    Q{idx} = 10*ones(S,A);
    L{idx} = zeros(S,A);
end

human_set = zeros(K,1);
traj_set = zeros(K,H,S,A);
policy_candidate = zeros(H,S);
for k = 1:K
    [~, state] = MDP(MDP_Setup, 0, 1, 1);
    reward_traj = 0;                                % trajectory reward, not visible by agent
    for h = 1:H
        [~, action] = max(Q{h}(state, :));                              % take action based on Q function
        traj_set(k, h, state, action) = 1;
        [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
        reward_traj = reward_traj + reward;
        
        L{h}(state, action) = L{h}(state, action) + 1;
        t = L{h}(state, action);
        
        if h == H
            V_next = 0;
        else
            V_next = min(max(Q{h+1}(state_next,:)), 10);                                 % next state value function
        end
        
        learning_rate = (H + 1) / (H + t);                              % learning rate
        exploration_bonus = sqrt(H^3 * logdelta / max(1,t));                     % exploration bonus
        
        Q{h}(state, action) = (1 - learning_rate) * Q{h}(state, action) ... 
            + learning_rate * (R_hat{h}(state, action) + V_next + 2 * exploration_bonus);     %Q leanring update
        
        state = state_next;                         % transition to next state
    end
    
    human_feedback = (reward_traj > reward_tau0);
    if reward_traj == reward_tau0
        human_feedback = randsample([0,1],1);
    end
    
    human_set(k) = human_feedback;
    
    if mod(k,2000) ==0
        % Call the human expert
        theta = ones(H,S,A);
        
        % maximize log likelihood to obtain reward
        nlog_likelihood = @(theta) -log_likelihood_base(traj_set(1:k, :, :, :), ...
            theta, human_set(1:k), traj_tau0);
        [theta_opt, fval] = fminunc(nlog_likelihood, theta);
        for idx = 1:H
            for s = 1:S
                for a = 1:A
                    R_hat{idx}(s,a) = theta_opt(idx,s,a);
                end
            end
        end
    end
    for idx = 1:H
        for s = 1:S
            [~,policy_candidate(idx,s)] = max(Q{h}(s,:));
        end
    end
end






end


%% Q learning + MLE + reward-free

if 1 
% Execute random policy to get baseline trajectory
[~, state] = MDP(MDP_Setup, 0, 1, 1);
reward_tau0 = 0;
traj_tau0 = zeros(H,S,A);
for h = 1:H
    action = randsample(A,1);
    traj_tau0(h, state, action) = 1;
    [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
    reward_tau0 = reward_tau0 + reward;
    state = state_next;
end

logdelta = 1;          % log terms
K = 20000;
R_hat = cell(H,1);
for h = 1:H
    R_hat{h} = 10*ones(S,A);
end
Q = cell(H,1);          % Q function for reward-free exploration
L = cell(H,1);          % visitation for reward-free exploration
for idx = 1:H
    Q{idx} = 10*ones(S,A);
    L{idx} = zeros(S,A);
end

human_set = zeros(K,1);
traj_set = zeros(K,H,S,A);
policy_candidate = zeros(H,S);
for k = 1:K
    [~, state] = MDP(MDP_Setup, 0, 1, 1);
    reward_traj = 0;                                % trajectory reward, not visible by agent
    for h = 1:H
        [~, action] = max(Q{h}(state, :));                              % take action based on Q function
        traj_set(k, h, state, action) = 1;
        [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
        reward_traj = reward_traj + reward;
        
        L{h}(state, action) = L{h}(state, action) + 1;
        t = L{h}(state, action);
        
        if h == H
            V_next = 0;
        else
            V_next = min(max(Q{h+1}(state_next,:)), 10);                                 % next state value function
        end
        
        learning_rate = (H + 1) / (H + t);                              % learning rate
        exploration_bonus = sqrt(H^3 * logdelta / max(1,t));                     % exploration bonus
        
        Q{h}(state, action) = (1 - learning_rate) * Q{h}(state, action) ... 
            + learning_rate * (V_next + 2 * exploration_bonus);     %Q leanring update
        
        state = state_next;                         % transition to next state
    end
    
    human_feedback = (reward_traj > reward_tau0);
    if reward_traj == reward_tau0
        human_feedback = randsample([0,1],1);
    end
    
    human_set(k) = human_feedback;
    
    if mod(k,2000) ==0
        % Call the human expert
        theta = ones(H,S,A);
        
        % maximize log likelihood to obtain reward
        nlog_likelihood = @(theta) -log_likelihood_base(traj_set(1:k, :, :, :), ...
            theta, human_set(1:k), traj_tau0);
        [theta_opt, fval] = fminunc(nlog_likelihood, theta);
        for idx = 1:H
            for s = 1:S
                for a = 1:A
                    R_hat{idx}(s,a) = theta_opt(idx,s,a);
                end
            end
        end
    end
    % VI
    MDP_Setup = cell(6,1);
    MDP_Setup{1} = S;
    MDP_Setup{2} = A;
    MDP_Setup{3} = H;
    MDP_Setup{4} = P;
    MDP_Setup{5} = R_hat;
    MDP_Setup{6} = Init;

    [policy_candidate, ~, ~, Q_hatVI] = ValueIteration(MDP_Setup);
end






end

%% Pure Q learning

if 0
logdelta = 0.5;          % log terms
Q = cell(H,1);          % Q function for reward-free exploration
L = cell(H,1);          % visitation for reward-free exploration
for idx = 1:H
    Q{idx} = 10*ones(S,A);
    L{idx} = zeros(S,A);
end

MDP_Setup = cell(6,1);
MDP_Setup{1} = S;
MDP_Setup{2} = A;
MDP_Setup{3} = H;
MDP_Setup{4} = P;
MDP_Setup{5} = R;
MDP_Setup{6} = Init;

for k = 1:5000
    [~, state] = MDP(MDP_Setup, 0, 1, 1);
    reward_traj = 0;                                % trajectory reward, not visible by agent
    for h = 1:H
        [~, action] = max(Q{h}(state, :));                              % take action based on Q function
        [reward, state_next] = MDP(MDP_Setup, h, state, action);        % observe reward and next state
        
        L{h}(state, action) = L{h}(state, action) + 1;
        t = L{h}(state, action);
        
        if h == H
            V_next = 0;
        else
            V_next = min(max(Q{h+1}(state_next,:)),10);                                 % next state value function
        end
        
        learning_rate = (H + 1) / (H + t);                              % learning rate
        exploration_bonus = sqrt(H^3 * logdelta / max(1,t));                     % exploration bonus
        
        Q{h}(state, action) = (1 - learning_rate) * Q{h}(state, action) ... 
            + learning_rate * (reward + V_next + 2 * exploration_bonus);     %Q leanring update
        
        state = state_next;                         % transition to next state
    end
    
    for idx = 1:H
        for s = 1:S
            [~,policy_candidate(idx,s)] = max(Q{idx}(s,:));
        end
    end
end



end


function y = log_likelihood_base(traj_set, theta, human_set, traj_tau0)
    [K, H, S, A] = size(traj_set);
    % human_set controls the length
    K = length(human_set);
    traj_set1 = repmat(reshape(traj_tau0, [1, H, S, A]), [K, 1, 1, 1]);
    traj_set0 = traj_set;
    reward_set0 = repmat(reshape(theta, [1, H, S, A]), [K, 1, 1, 1]) .* (traj_set0 - traj_set1);
    reward_sum0 = sum(sum(sum(reward_set0, 4), 3), 2);
    reward_set1 = repmat(reshape(theta, [1, H, S, A]), [K, 1, 1, 1]) .* (traj_set1 - traj_set0);
    reward_sum1 = sum(sum(sum(reward_set1, 4), 3), 2);
    y = sum(log(human_set .* sigmoid(reward_sum0) + (1-human_set) .* sigmoid(reward_sum1)));
end

function y = log_likelihood(traj_set0, traj_set1, theta, human_set)
    % human_set controls the length
    [K, H, S, A] = size(traj_set0);
    K = length(human_set);
    reward_set0 = repmat(reshape(theta, [1, H, S, A]), [K, 1, 1, 1]) .* (traj_set0 - traj_set1);
    reward_sum0 = sum(sum(sum(reward_set0, 4), 3), 2);
    reward_set1 = repmat(reshape(theta, [1, H, S, A]), [K, 1, 1, 1]) .* (traj_set1 - traj_set0);
    reward_sum1 = sum(sum(sum(reward_set1, 4), 3), 2);
    y = sum(log(human_set .* sigmoid(reward_sum0) + (1-human_set) .* sigmoid(reward_sum1)));
end

function y = sigmoid(x)
    y = 1./(1+exp(-x));
end












