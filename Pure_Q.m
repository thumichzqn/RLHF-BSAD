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

%% Pure Q learning
% initialization of parameters

iternum = 10;
logdelta = 0.5;          % log terms
batch_size = 32;        % batch size
K = 20000;

% record the policy every 50 episodes
policy_record = cell(K/50, iternum);
value_record = zeros(K/50, iternum);

tic
for iter = 1:iternum
    Q = cell(H,1);          % Q function for reward-free exploration
    L = cell(H,1);          % visitation for reward-free exploration
    for idx = 1:H
        Q{idx} = 10*ones(S,A);
        L{idx} = zeros(S,A);
    end
    
    % candidate policy
    policy_candidate = zeros(H,S);      
    for h=1:H
        for s=1:S
            policy_candidate(h,s) = randsample(A,1);
        end
    end

    MDP_Setup = cell(6,1);
    MDP_Setup{1} = S;
    MDP_Setup{2} = A;
    MDP_Setup{3} = H;
    MDP_Setup{4} = P;
    MDP_Setup{5} = R;
    MDP_Setup{6} = Init;

    for k = 1:K
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
        
        if mod(k,50) == 0
            policy_record{k/50, iter} = policy_candidate;
            value_record(k/50, iter) = Policy_Eval(MDP_Setup, policy_candidate); 
        end
    end
end
toc

%% record the data

%save('Pure_Q.mat','policy_record','value_record')