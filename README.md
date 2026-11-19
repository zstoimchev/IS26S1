# Astronaut

A reinforcement learning project based on Astronaut, a Flappy Bird-style game where an astronaut uses a jetpack to
navigate through obstacles.

The goal is to train an AI agent using Q-learning to control the astronaut, avoid collisions, dodge asteroids, and
collect oxygen tanks.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python src/example-game.py  # Play manually; SPACE activates thrust
python src/agent_p1.py      # Train and demonstrate basic flight
python src/agent_p2.py      # Train and evaluate asteroid avoidance
```

Hold SPACE to activate the jetpack.

- Training produces `training.csv`; both agents overwrite the same file, so save each run separately
- Subtask 2 two also records `astronaut.webm`
- Instructions for creating the loss graph are in `docs/PLOTTING.md`
