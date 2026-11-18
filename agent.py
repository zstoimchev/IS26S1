import gymnasium as gym
import random
import csv
import time
from collections import Counter

from famnit_gym.wrappers.astronaut import Asteroids, Oxygen, DebugOverlay, Video

random.seed(42)

# Gamma applies per simulated frame. Six frames correspond to ~0.988 discount
GAMMA = 0.998
FRAMES_PER_ACTION = 6
EPISODES = 15_000
q_table: dict[tuple[int, int, int, int], list[float]] = {}


def make_env(render_mode=None, record=False):
    fun_env = gym.make("famnit_gym/Astronaut-v1", render_mode=render_mode)
    fun_env = Asteroids(fun_env)
    if record:
        fun_env = Video(fun_env, filename="astronaut.webm")
        fun_env = DebugOverlay(fun_env)
    return fun_env


def get_state(fun_observation):
    height, velocity, asteroid_x, asteroid_y = fun_observation
    height_bin = int(height // 50)
    velocity_bin = int(velocity // 80)
    distance_bin = min(3, int((asteroid_x - 300) // 200))
    asteroid_height_bin = int(asteroid_y // 100)
    return height_bin, velocity_bin, distance_bin, asteroid_height_bin


def get_q_values(fun_state: tuple[int, int, int, int]) -> list[float]:
    if fun_state not in q_table:
        q_table[fun_state] = [0.0, 0.0]
    return q_table[fun_state]


def learn(fun_state, fun_action, fun_reward, fun_next_state, fun_terminated, transition_discount):
    alpha = 0.10  # learning rate
    current_values = get_q_values(fun_state)
    target = fun_reward
    if not fun_terminated:
        target += transition_discount * max(get_q_values(fun_next_state))
    error = target - current_values[fun_action]
    current_values[fun_action] += alpha * error
    return error ** 2


def choose_action(fun_state, fun_epsilon=0.2):
    if random.random() < fun_epsilon:
        return random.randrange(2)
    values = get_q_values(fun_state)
    if values[0] == values[1]:
        return random.randrange(2)
    return 0 if values[0] > values[1] else 1


def choose_eval_action(fun_state):
    values = q_table.get(fun_state)
    if values is None:
        return 0
    return 0 if values[0] >= values[1] else 1


def perform_action(env, action):
    cumulative_reward = 0.0
    discount = 1.0
    frames = 0
    for _ in range(FRAMES_PER_ACTION):
        observation, _env_reward, terminated, truncated, info = env.step(action)
        reward = -100.0 if terminated else 1.0
        cumulative_reward += discount * reward
        discount *= GAMMA
        frames += 1
        if terminated or truncated:
            break
    return observation, cumulative_reward, terminated, truncated, info, frames, discount


env = make_env()
training_results = []
training_start = time.perf_counter()

for episode in range(EPISODES):
    progress = episode / max(1, EPISODES - 1)
    epsilon = 0.5 * (0.02 / 0.5) ** progress
    observation, info = env.reset(seed=42 + episode)
    state = get_state(observation)
    steps = 0
    updates = 0
    total_loss = 0.0

    while True:
        action = choose_action(state, epsilon)
        observation, reward, terminated, truncated, info, frames, discount = (perform_action(env, action))
        next_state = get_state(observation)
        total_loss += learn(state, action, reward, next_state, terminated, discount)
        steps += frames
        updates += 1
        state = next_state
        if terminated or truncated:
            break

    average_loss = total_loss / updates
    training_results.append((episode + 1, steps, average_loss))
    if (episode + 1) % 100 == 0 or episode == 0:
        print(f"Episode {episode + 1}: survived {steps} steps, loss: {average_loss:.6f}")

print(f"Training time: {time.perf_counter() - training_start:.2f} seconds")
print(f"Learned Q-table states: {len(q_table)}")
with open("training.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["episode", "steps", "mean_squared_td_error"])
    writer.writerows(training_results)
env.close()

# Record one trained episode, preserving your original video and overlay.
env = make_env(render_mode="human", record=True)
observation, info = env.reset(seed=42)
steps = 0
while True:
    action = choose_eval_action(get_state(observation))
    observation, _, terminated, truncated, info, frames, _ = perform_action(env, action)
    steps += frames
    if terminated or truncated:
        break
print(f"Trained agent survived {steps} steps")
env.close()

# Test the trained policy on the same 50 unseen seeds as your remote branch.
env = make_env()
successes = 0
total_steps = 0
unseen_states = 0
collision_reasons = Counter()

for seed in range(20000, 20050):
    observation, info = env.reset(seed=seed)
    while True:
        state = get_state(observation)
        unseen_states += int(state not in q_table)
        action = choose_eval_action(state)
        observation, _, terminated, truncated, info, frames, _ = perform_action(env, action)
        if terminated or truncated:
            successes += int(truncated and not terminated)
            collision_reasons[info["collision"]] += 1
            total_steps += info["steps"]
            break

env.close()
print(f"Successful flights: {successes}/50")
print(f"Average survival: {total_steps / 50:.1f} / 1200")
print(f"Q-table states: {len(q_table)}")
print(f"Unseen evaluation states (decision visits): {unseen_states}")
print(f"Episode endings: {dict(collision_reasons)}")
