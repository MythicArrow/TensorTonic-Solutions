def sarsa_update(q_table: list, state: int, action: int, reward: float, next_state: int, next_action: int, alpha: float, gamma: float) -> list:
    """
    Returns a copied Q-table after one SARSA update.
    """
    # Write code here
    copy_q_table = [row[:] for row in q_table]
    td_error = reward + gamma * q_table[next_state][next_action] - q_table[state][action]
    copy_q_table[state][action] = copy_q_table[state][action] + alpha * td_error
    return copy_q_table
    