import gymnasium as gym
import famnit_gym

# Create and reset the environment
env = gym.make('famnit_gym/Astronaut-v1', render_mode='human')
observation, info = env.reset()


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


# Execute random actions
state = get_state(observation)
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    next_state = get_state(observation)

    reward = -100.00 if terminated else 1.0
    learn(state, action, reward, next_state, terminated)

    print(f"State: {state}, Q-values: {get_q_values(state)}")

    state = next_state
    done = terminated or truncated

# Close the environment
env.close()
