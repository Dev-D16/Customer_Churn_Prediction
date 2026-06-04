[README.md](https://github.com/user-attachments/files/28600083/README.md)# Multi-Agent Reinforcement Learning for Traffic Light Control

## Project Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system for adaptive traffic signal control in urban road networks. Each traffic light at an intersection acts as an independent learning agent that learns to optimize signal phases based on real-time traffic conditions.

### Key Features
- **Independent Multi-Agent DQN**: Each traffic light has its own neural network
- **Custom Traffic Simulator**: Built-in grid-based traffic simulation (no external dependencies)
- **Real-time Visualization**: Live pygame-based traffic visualization
- **Baseline Comparison**: Compare MARL against fixed-time traffic signals
- **Comprehensive Metrics**: Waiting time, throughput, and reward analysis

---

## Architecture

### Multi-Agent System Design

```
Traffic Network (NxN Grid)
    |
    |-- Agent TL_0_0 (Intersection [0,0])
    |-- Agent TL_0_1 (Intersection [0,1])
    |-- Agent TL_1_0 (Intersection [1,0])
    |-- Agent TL_1_1 (Intersection [1,1])
    ...
```

Each agent:
- **Observes**: Queue lengths and waiting times from each direction
- **Decides**: Keep current phase or switch to other direction
- **Learns**: From reward signal (negative total waiting time)

### State Space (7 dimensions)
| Feature | Description |
|---------|-------------|
| `queue_ns` | Number of vehicles in North-South direction |
| `queue_ew` | Number of vehicles in East-West direction |
| `wait_ns` | Total waiting time of NS vehicles |
| `wait_ew` | Total waiting time of EW vehicles |
| `ns_green` | Binary: NS has green (1) or not (0) |
| `ew_green` | Binary: EW has green (1) or not (0) |
| `phase_duration` | How long current phase has been active |

### Action Space (2 actions)
| Action | Description |
|--------|-------------|
| `0` | **Keep** current signal phase |
| `1` | **Switch** to other phase (via yellow) |

### Reward Function
```
Reward = - (Sum of all vehicle waiting times at intersection) / 100
```
The agent is penalized for vehicles waiting, encouraging it to minimize congestion.

---

## MARL Algorithm: Independent Q-Learning

We use **Independent Q-Learning** where each agent maintains its own Deep Q-Network:

1. **Experience Replay**: Each agent stores experiences (s, a, r, s') in a buffer
2. **Target Network**: Separate target network for stable learning
3. **Epsilon-Greedy**: Exploration decays over training
4. **Decentralized Execution**: Each agent acts independently at runtime

### Neural Network Architecture
```
Input (7) -> Linear(7, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 2) -> Q-values
```

---

## Project Structure

```
marl_traffic_project/
    environment.py      # Traffic simulation environment
    agents.py           # Multi-Agent DQN implementation
    train.py            # Training script
    evaluate.py         # Evaluation & comparison script
    visualize.py        # Real-time visualization
    run_all.py          # One-click pipeline
    requirements.txt    # Python dependencies
    README.md           # This file
```

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install numpy torch pygame matplotlib tqdm
```

### Run Everything (One Command)

```bash
python run_all.py
```

This will:
1. Train MARL agents (500 episodes, ~5 minutes)
2. Evaluate against fixed-time baseline
3. Generate comparison plots
4. Save all results for your report

### Manual Steps

```bash
# 1. Train agents
python train.py --grid-size 2 --episodes 500 --plot

# 2. Evaluate and compare
python evaluate.py --grid-size 2 --episodes 20

# 3. Visualize (opens pygame window)
python visualize.py --grid-size 2 --save-frames
```

---

## Expected Results

After training 500 episodes on a 2x2 grid:

| Metric | MARL | Fixed-Time | Improvement |
|--------|------|------------|-------------|
| Avg Waiting Time | ~15-25 steps | ~35-50 steps | **40-50% reduction** |
| Vehicles Arrived | Higher | Lower | **10-20% increase** |
| Adaptability | High (learns patterns) | None | Dynamic response |

The MARL system learns to:
- Give longer green to busier directions
- Coordinate implicitly through shared traffic state
- Adapt to varying traffic patterns

---

## Implementation Details

### Traffic Simulation
- Grid-based road network with configurable size
- Vehicles spawn at edges and travel to opposite sides
- Simple car-following model with collision avoidance
- Traffic lights with green/yellow/red phases

### Training Parameters
| Parameter | Value |
|-----------|-------|
| Episodes | 500 |
| Max Steps/Episode | 500 |
| Learning Rate | 0.001 |
| Discount Factor | 0.95 |
| Batch Size | 32 |
| Replay Buffer | 5000 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.05 |
| Epsilon Decay | 0.995 |
| Target Update | Every 100 steps |
| Hidden Layer | 128 units |

---

## Extensions & Future Work

1. **Communication Between Agents**: Share intended actions with neighbors
2. **Centralized Training with Decentralized Execution (CTDE)**: Use QMIX or MAPPO
3. **SUMO Integration**: Use realistic SUMO simulator with real road networks
4. **Larger Grids**: Scale to 4x4, 8x8, or city-scale networks
5. **Heterogeneous Traffic**: Add different vehicle types, pedestrians, emergency vehicles
6. **Multi-Objective**: Optimize for emissions and fuel consumption too

---

## References

1. Bakker et al. - "Traffic Light Control by Multiagent Reinforcement Learning Systems"
2. Wei et al. - "A Novel Multi-Agent Deep RL Approach for Traffic Signal Control" (2023)
3. SUMO-RL: https://github.com/LucasAlegre/sumo-rl
4. PettingZoo: https://pettingzoo.farama.org/
5. PyTSC: "A Unified Platform for Multi-Agent RL in Traffic Signal Control" (2025)

---

## For Your Project Report

### Suggested Report Structure
1. **Introduction** - Traffic congestion problem, need for adaptive control
2. **Literature Review** - MARL approaches to traffic signal control
3. **Methodology** - Your system design (use diagrams from this README)
4. **Implementation** - Code structure, algorithms used
5. **Experiments** - Training setup, parameters
6. **Results** - Include the generated plots (comparison.png, distributions.png)
7. **Conclusion** - MARL outperforms fixed-time control
8. **Future Work** - Extensions mentioned above

### Key Points to Highlight
- Each traffic light is an independent RL agent
- Uses Deep Q-Network for function approximation
- Reward function minimizes vehicle waiting time
- Outperforms traditional fixed-time traffic signals
- Scalable to larger networks

---

**Author**: BE 6th Semester Project
**Topic**: Multi-Agent Reinforcement Learning for Traffic Light Control
**Date**: 2026

