
# Off-policy / On-policy
1. Find the action the agent *really* takes next. That comes from the behavior policy, usually ${}\varepsilon{}$-greedy.
2. Find the action, or action distribution, that *appears in the update target*. That is the target policy. 
3. If they are the same, the algorithm is on-policy. If they differ, it is off-policy. 