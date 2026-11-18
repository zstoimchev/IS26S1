import gymnasium as gym
import famnit_gym
import random
import csv
import time

random.seed(42)
env = gym.make('famnit_gym/Astronaut-v1', render_mode=None)


def get_state(fun_observation):
    height, velocity = fun_observation

    height_bin = int(height // 25)
    velocity_bin = int(velocity // 20)

    return height_bin, velocity_bin


q_table: dict[tuple[int, int], list[float]] = {}


def get_q_values(fun_state: tuple[int, int]) -> list[float]:
    if fun_state not in q_table:
        q_table[fun_state] = [0.0, 0.0]
    return q_table[fun_state]


def learn(
        fun_state: tuple[int, int],
        fun_action: int,
        fun_reward: float,
        fun_next_state: tuple[int, int],
        fun_terminated: bool):
    alpha = 0.10  # learning rate
    gamma = 0.99  # importance of future rewards

    current_values = get_q_values(fun_state)

    target = fun_reward
    if not fun_terminated:
        target += gamma * max(get_q_values(fun_next_state))

    error = target - current_values[fun_action]
    current_values[fun_action] += alpha * error
    return error ** 2  #


def choose_action(fun_state, epsilon=0.2):
    # choose randomly for 20% of the time...
    if random.random() < epsilon:
        return random.randrange(2)

    values = get_q_values(fun_state)
    if values[0] == values[1]:
        return random.randrange(2)

    return 0 if values[0] > values[1] else 1


training_results = []
training_start = time.perf_counter()

for episode in range(1000):
    observation, info = env.reset(seed=42 + episode)
    state = get_state(observation)
    done = False
    steps = 0
    total_loss = 0.0

    while not done:
        action = choose_action(state)
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(observation)

        reward = -100.0 if terminated else 1.0
        total_loss += learn(state, action, reward, next_state, terminated)

        state = next_state
        steps += 1
        done = terminated or truncated

    average_loss = total_loss / steps
    training_results.append((episode + 1, steps, average_loss))
    print(f"Episode {episode + 1}: survived {steps} steps, loss: {average_loss:.6f}")

print(f"Training time: {time.perf_counter() - training_start:.2f} seconds")

# save measurements
with open('training.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'steps', 'mean_squared_td_error'])
    writer.writerows(training_results)

env.close()

# evaluate
env = gym.make('famnit_gym/Astronaut-v1', render_mode="human")
observation, info = env.reset(seed=42)

done = False
steps = 0

while not done:
    state = get_state(observation)
    action = choose_action(state, epsilon=0.0)

    observation, reward, terminated, truncated, info = env.step(action)

    steps += 1
    done = terminated or truncated

print(f"Trained agent survived {steps} steps")

env.close()
