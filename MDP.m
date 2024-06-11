%% Building MDP environment

function [reward, state_next] = MDP(MDP_Setup, step_cur, state_cur, action_cur)
    % This function simulates the MDP,
    % use step_cur = 0 gives the initial state
    % use step_cur = 1 gives the 2nd state
    % use step_cur = H gives the terminal reward
    
    % MDP parameters
    S = MDP_Setup{1};
    A = MDP_Setup{2};
    H = MDP_Setup{3};
    P = MDP_Setup{4};
    R = MDP_Setup{5};
    Init = MDP_Setup{6};
    
    if step_cur < 1
        % simulate initial state
        state_next = randsample(S, 1, true, Init);
        reward = 0;
        
    elseif step_cur >= H
        % If last step, only feedback the reward.
        reward = R{step_cur}(state_cur, action_cur);
        state_next = 0;
        
    else
        % Feedback a reward
        reward = R{step_cur}(state_cur, action_cur);

        % Feedback a next state
        state_next = randsample(S, 1, true, P{step_cur, action_cur}(state_cur, :) );
    end
end