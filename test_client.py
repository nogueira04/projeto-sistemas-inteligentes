import numpy as np
from connection import connect, get_state_reward

ACTIONS = ["left", "right", "jump"]

def state_to_index(state_bin):
    return int(state_bin, 2)

def load_q_table(filename="resultado.txt"):
    q_table = []
    with open(filename, "r") as f:
        for line in f:
            q_table.append([float(v) for v in line.strip().split()])
    return np.array(q_table)

def run_test_episode(q_table):
    conn = connect(2037)
    state_bin, _ = get_state_reward(conn, "right")
    state_index = state_to_index(state_bin)

    done = False
    steps = 0
    total_reward = 0

    while not done and steps < 200:
        action_idx = np.argmax(q_table[state_index])
        action = ACTIONS[action_idx]

        next_state_bin, reward = get_state_reward(conn, action)
        next_state_index = state_to_index(next_state_bin)

        total_reward += reward
        state_index = next_state_index
        steps += 1

        print(f"Step {steps}: Action={action}, Reward={reward}")

        if reward <= -100:
            print("Perdeu")
            done = True
        elif reward == -1:
            print("Ganhou")
            done = True

    print(f"\nEpisódio de teste finalizado com {steps} passos. Recompensa total: {total_reward}")

def main():
    q_table = load_q_table("resultado.txt")
    run_test_episode(q_table)

if __name__ == "__main__":
    main()
