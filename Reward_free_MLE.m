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

%% PureRandom + Q learning + MLE

% initialization of parameters

iternum = 10;
logdelta = 1;          % log terms
K = 10000;

% record the policy every 50 episodes
policy_record = cell(K/50, iternum);
value_record = zeros(K/50, iternum);

tic

for iter = 1:iternum
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
    
    rng(iter * 2017)
    
    R_hat = cell(H,1);
    for h = 1:H
        R_hat{h} = 10*ones(S,A);
    end
    Q_free = cell(H,1);          % Q function for reward-free exploration
    Q = cell(H,1);          % Q function for pessimissm
    L = cell(H,1);          % visitation for reward-free exploration
    for idx = 1:H
        Q_free{idx} = 10*ones(S,A);
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
            [~, action] = max(Q_free{h}(state, :));                              % take action based on Q function
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
            
            % Reward-free Q leanring update
            Q_free{h}(state, action) = (1 - learning_rate) * Q{h}(state, action) ... 
                + learning_rate * (V_next + 2 * exploration_bonus);         
            
            %Pessimistic Q
            Q{h}(state, action) = (1 - learning_rate) * Q{h}(state, action) ... 
                + learning_rate * (R_hat{h}(state, action) + V_next - 2 * exploration_bonus);         %Q leanring update
            
            
            state = state_next;                         % transition to next state
        end

        human_feedback = (reward_traj > reward_tau0);
        if reward_traj == reward_tau0
            human_feedback = randsample([0,1],1);
        end

        human_set(k) = human_feedback;

        if mod(k,400) ==0
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
                [~,policy_candidate(idx,s)] = max(Q{idx}(s,:));
            end
        end
        if mod(k,50) == 0
            policy_record{k/50, iter} = policy_candidate;
            value_record(k/50, iter) = Policy_Eval(MDP_Setup, policy_candidate); 
            %disp([iter, k])
        end
    end
    
end
toc
%% record the data

%save('Reward_free_MLE.mat','policy_record','value_record')


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