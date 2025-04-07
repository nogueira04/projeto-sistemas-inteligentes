import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from connection import connect, get_state_reward
import sys

# Q-learning parameters
ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor
EPSILON = 1.0     # Initial exploration rate
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
NUM_EPISODES = 5000
SAVE_FREQ = 100

# Available actions
ACTIONS = ["left", "right", "jump"]

# Convert binary state to index
def state_to_index(state_bin):
    return int(state_bin, 2)

# Epsilon-greedy action selection
def choose_action(q_table, state_index, epsilon):
    if random.random() < epsilon:
        print("Explorando ação aleatória")
        return random.randint(0, 2)
    print("Explotando ação com Q-table")
    return np.argmax(q_table[state_index])

# Save Q-table to a file
def save_q_table(q_table, filename="resultado.txt"):
    with open(filename, "w") as f:
        for row in q_table:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

def load_q_table(filename="resultado.txt"):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            row = [float(x) for x in line.split()]
            data.append(row)
    return np.array(data)

def main():
    resume = len(sys.argv) > 1 and sys.argv[1] == "resume"
    if resume:
        print("Resuming training from existing Q-table.")
        q_table = load_q_table()
    else:
        q_table = np.zeros((96, 3))  # 96 states x 3 actions

    reward_per_step_history = []

    epsilon = EPSILON

    # Live plot setup
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Recompensa média por passo')
    ax.set_xlabel('Episódios')
    ax.set_ylabel('Recompensa média por passo')
    ax.set_title('Aprendizado com Q-Learning')
    ax.legend()
    ax.grid(True)

    conn = connect(2037)

    for episode in range(NUM_EPISODES):
        state_bin, reward = get_state_reward(conn, "right")
        state_index = state_to_index(state_bin)

        done = False
        steps = 0
        total_reward = 0

        print("=" * 20)
        print(f"Episode {episode + 1} started.")

        while not done:
            action_idx = choose_action(q_table, state_index, epsilon)
            action = ACTIONS[action_idx]

            next_state_bin, reward = get_state_reward(conn, action)
            print(f"Action: {action}, State: {state_bin} -> {next_state_bin}, Reward: {reward}")
            next_state_index = state_to_index(next_state_bin)

            best_next = np.max(q_table[next_state_index])
            q_table[state_index, action_idx] += ALPHA * (
                reward + GAMMA * best_next - q_table[state_index, action_idx]
            )

            state_index = next_state_index
            steps += 1
            total_reward += reward

            if reward == -100 or reward == -1:
                done = True

        print(f"Episode {episode + 1} completed in {steps} steps.")
        print(f"Epsilon: {epsilon:.4f}")
        print("=" * 20)

        # Reward per step calculation
        reward_per_step = total_reward / steps if steps > 0 else total_reward
        reward_per_step_history.append(reward_per_step)

        # Decay epsilon
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
            epsilon = max(epsilon, EPSILON_MIN)

        if episode % 10 == 0:
            # Update plot
            line.set_xdata(np.arange(len(reward_per_step_history)))
            line.set_ydata(reward_per_step_history)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        if episode % SAVE_FREQ == 0 and episode > 0:
            save_q_table(q_table)
            print(f"Q-table saved at episode {episode}.")

    # Save Q-table after training
    save_q_table(q_table)
    print("Treinamento finalizado. Q-table salva em 'resultado.txt'.")

    # Keep plot visible at the end
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
