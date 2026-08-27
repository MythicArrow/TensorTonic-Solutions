def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    new_values = []
    num_states = len(values)
    
    for s in range(num_states):
        # Get the list of actions for this state
        # The number of actions is the length of rewards[s] or transitions[s]
        num_actions = len(rewards[s])
        
        best_q_value = -float('inf')
        
        for a in range(num_actions):
            # 1. Get immediate reward
            immediate_reward = rewards[s][a]
            
            # 2. Calculate expected next-state value
            # transitions[s][a] is a list of probabilities [P(s'|s,a) for all s']
            expected_next_value = 0.0
            for next_s in range(num_states):
                prob = transitions[s][a][next_s]
                expected_next_value += prob * values[next_s]
            
            # 3. Calculate Q(s, a)
            q_value = immediate_reward + gamma * expected_next_value
            
            # 4. Track the maximum
            if q_value > best_q_value:
                best_q_value = q_value
        
        new_values.append(best_q_value)
        
    return new_values