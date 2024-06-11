function [J, V, Q] = Policy_Eval(MDP_Setup, policy)
% policy is in H*S
% Value iteration
    S = MDP_Setup{1};
    A = MDP_Setup{2};
    H = MDP_Setup{3};
    P = MDP_Setup{4};
    R = MDP_Setup{5};
    Init = MDP_Setup{6};

    V = zeros(H , S);
    Q = cell(H,1);
    for h = 1:H
        Q{h} = zeros(S,A);
    end

    for s = 1:S
        for a = 1:A
            Q{H}(s,a) = R{H}(s,a);
        end
        V(H,s) = Q{H}(s, policy(H,s)); 
    end
    for h = H-1:-1:1
        for s = 1:S
            for a = 1:A
                Q{h}(s,a) = R{h}(s,a) + P{h,a}(s,:) * V(h+1, :)';
            end
            V(h,s) = Q{h}(s,policy(h,s)); 
        end
    end
    J = Init * V(1,:)';
end