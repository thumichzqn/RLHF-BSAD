clear all
close all
clc

%% MDP setup
S = 3;          % State Space = {1,2,3}
A = 2;          % Action Space = {1,2}
H = 2;          % Stage = {1,2,3,4}

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

% initialization of parameters

iternum = 10;
logdelta = 0.5;          % log terms
batch_size = 16;        % batch size
K = 20000;

% record the policy every 50 episodes
policy_record = cell(K/50, iternum);
value_record = zeros(K/50, iternum);

tic
for iter = 1:iternum
    
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
    
    % candidate policy
    policy_candidate = zeros(H,S);      
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

    rng(iter * 2019);
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

        % Output a policy 
        if mod(k,50) == 0
            policy_record{k/50, iter} = policy_candidate;
            value_record(k/50, iter) = Policy_Eval(MDP_Setup, policy_candidate); 
        end
    end

    finish_time(iter) = k;
    
    while k <= K
        if mod(k,50) == 0
            policy_record{k/50, iter} = policy_candidate;
            value_record(k/50, iter) = Policy_Eval(MDP_Setup, policy_candidate); 
        end
        k = k+1;
    end

end
toc

%% record the data

%save('Proposed_2.mat','policy_record','value_record','finish_time')







