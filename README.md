# MSML642 Robot Search and Tracking

## Phase 1 submission

The [two-page progress report](docs/phase1/MSML642_Phase1_Report.pdf) and
[one-minute Gazebo video](docs/phase1/MSML642_Phase1_Demo.mp4) document the working setup.
The [editable report text](docs/phase1/MSML642_Phase1_Report.md) and measured results are
in docs/phase1. The capture clip uses the rule-based baseline with the NPC initially visible.
The following training clip shows early Q-learning behavior. The 12-episode debugging run
did not achieve capture. Full training and comparative experiments remain Phase 2 work.

The scene is a 16 脳 12 m indoor facility. A central corridor connects four areas through doorways. Partitions, barriers, crates, drums, cabinets and a workbench all take part in collisions and range sensing. A blue differential-drive robot starts in the middle of the corridor and carries a 360掳 laser rangefinder. Docker Desktop runs Ubuntu 24.04, ROS 2 Jazzy and Gazebo Harmonic in a Linux container. Gazebo uses headless software rendering, and ROS 2 commands run inside the container.

## Quick start for team members

Install Git and Docker with Docker Compose. Windows users should start Docker Desktop and use Linux containers. The first start needs an internet connection to download the image and dependencies.

In a terminal, run:

```sh
git clone https://github.com/Gjohnx/MSML642-Group-Project.git
cd MSML642-Group-Project
docker compose --profile gui up -d --build
```

Once the containers are up, open the [Gazebo GUI](http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale) in your browser. Compose starts both the simulation and the browser GUI; ROS 2 and Gazebo run inside the containers.

## Start and verify

To run only the headless simulation, run this from the repository root:

```sh
docker compose up -d --build
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && python3 smoke_test.py'
```

To check the complete task interface, including resets, target occlusion, capture and a
deliberate contact with a barrier, run this after other controllers have finished:

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m sim.task_check'
```

The smoke test sends forward, turn-left, turn-right and stop commands from the starting position. It checks the odometry and lidar readings and prints `PASS` on success. Before each re-run, run `docker compose restart sim` to put the robot back at its starting position.

View logs or shut down the simulation:

```powershell
docker compose logs sim
docker compose down
```

## Viewing the Gazebo GUI

### Windows desktop window

Start Docker Desktop, then run this in the project folder:

```powershell
docker compose up -d --build desktop
```

In a Docker Desktop WSL 2 environment with WSLg enabled, this opens a standalone Gazebo Sim window. The window shows the map from an angled overhead view by default. After closing the window, run the same command to reopen it, or run `docker compose stop desktop` to stop the desktop GUI.

### Browser

```powershell
docker compose --profile gui up -d --build
```

Open the [Gazebo GUI](http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale) in your Windows browser. Keep the window open and run the smoke test above in PowerShell to watch the robot move. To stop the browser GUI service, run `docker compose stop viewer`.


## ROS 2 interface

| Topic | Type | Purpose |
| --- | --- | --- |
| `/robot/cmd_vel` | `geometry_msgs/msg/Twist` | Forward, turn and stop |
| `/robot/odom` | `nav_msgs/msg/Odometry` | Robot odometry |
| `/robot/scan` | `sensor_msgs/msg/LaserScan` | Laser range scan |
| `/clock` | `rosgraph_msgs/msg/Clock` | Simulation clock |

A red NPC follows a fixed patrol route. The learner detects it inside a 4 m, 90-degree field of view with lidar-based occlusion checks. Episodes end on capture within 0.5 m, three separate collisions, or the step limit.

## Project structure

```
sim/                  Gazebo scene and ROS 2 bridge (mounted read-only in the container)
  room.sdf            Map and learner robot
  bridge.yaml         ROS 2 <-> Gazebo topics
  run.sh / gui.sh / view.sh
  smoke_test.py       Motion and lidar check
  npc_patrol.py       Scripted NPC waypoint controller
  npc_routes.yaml     NPC patrol routes
rl/                   Reinforcement learning code (mounted at /project/rl)
  core/               ROS-free: state discretization, Q-table, rewards,
                      lost-target search, collision recovery, baselines,
                      explore.py (explore-then-chase demo logic)
  ros/                env.py (reset/step interface), detector.py (limited-FOV detector),
                      odometry.py (wheel odometry for the explore state)
  config/             qlearning.yaml, rewards.yaml, explore.yaml
  tests/              Unit tests (no Gazebo needed)
  train.py / evaluate.py / experiment.py
  explore.py          Train or evaluate the exploration Q-table
  demo.py             Explore-then-chase demo with the Chase button
experiments/          Training outputs (not committed; mounted at /project/experiments)
analysis/             Plotting scripts for the report
docs/                 Reports and figures
```

`rl/core/` does not import `rclpy`, so it can be tested without starting the simulation. Training runs inside the container:

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.train'
docker compose exec sim bash -c 'cd /project && python3 -m pytest rl/tests'
```

## Train and evaluate

Run these from the repository root with the simulation running. Only one controller may operate the simulation at a time.

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.train --episodes 20 --epsilon-decay-episodes 15 --max-steps 120 --output experiments/train_seed0'
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.evaluate --policy experiments/train_seed0/q_table.npy --episodes 6 --output experiments/eval_q_seed0'
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.evaluate --policy rule --episodes 6 --output experiments/eval_rule_seed0'
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.evaluate --policy random --episodes 6 --output experiments/eval_random_seed0'
```

The 20-episode command is a short integration run. The proposal configuration defaults to 2,000 training episodes. Use multiple seeds and longer runs to assess learning. Each run saves its effective configuration, per-step traces, episode CSV and a metrics report. Training also saves q_table.npy after every completed episode. --resume loads an existing table. --epsilon-decay-episodes sets the decay schedule for the requested run.

Default evaluation cycles through three learner spawns and two NPC routes. Six episodes cover every pair once. Ground-truth coordinates are excluded from policy input. All policies share safety checks, collision recovery and lost-target search. A rediscovery requires at least five lost steps, with a maximum of three rewarded rediscoveries per episode to limit visibility-bonus farming.

Run both reward ablations and the two baselines with multiple seeds:

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.experiment --seeds 0 1 --episodes 20 --eval-episodes 6 --max-steps 120 --ablations'
```

Add `--heldout` to also evaluate every policy on the held-out start set. Figures (`learning_curves.png`, `policy_comparison.png`) and tables (`summary.csv` with mean and spread across seeds, `comparison.csv` per run) are saved under experiments/suite/figures. Reward ablations have different reward definitions, so compare capture, detection and collision metrics as well as return.

### Two-stage training

Stage 1 trains tracking only: the `tracking` start set places the learner where the NPC is in view at reset. Stage 2 continues from that Q-table on the default starts, where the NPC must be found first. `--epsilon-start` lowers the starting exploration rate so stage 2 does not overwrite what stage 1 learned.

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.train --spawn-set tracking --episodes 300 --epsilon-decay-episodes 225 --output experiments/stage1_seed0'
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.train --resume experiments/stage1_seed0/q_table.npy --epsilon-start 0.3 --episodes 500 --epsilon-decay-episodes 375 --output experiments/stage2_seed0'
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.evaluate --policy experiments/stage2_seed0/q_table.npy --spawn-set heldout --output experiments/eval_stage2_heldout_seed0'
```

Start sets are listed under `spawn_sets` in rl/config/qlearning.yaml. `heldout` is for evaluation only. `--no-lost-search` turns off the shared lost-target search, which shows what a policy does on its own after losing the NPC.

## Explore-then-chase demo

The robot explores with its own exploration Q-table until someone clicks **Chase** in the Gazebo window, then chases the NPC with the trained chase Q-table. The exploration state has only the lidar obstacle levels and how long since the robot reached a new area (from its own wheel odometry), so it never chases on its own. The last-seen direction of the NPC is still tracked, so the chase starts knowing which way to turn.

The Gazebo window includes a **Chase** button in both the standard and demo views. Rebuild the viewer to install the GUI plugin:

```sh
docker compose --profile gui up -d --build
```

Start rl.demo with the exploration and chase tables using the command below. The button becomes available when the demo enters exploration. Seeing the NPC does not switch modes. Clicking **Chase** sends one request and starts pursuit. The panel shows the mode reported by the running demo and disables the button during chase or when no demo is running. Restarting the demo returns to exploration.

The learner moves forward at 0.35 m/s. Both NPC patrol routes use 0.15 m/s. The learner's lidar covers 360 degrees with 360 samples. The target detector uses its configured range and field of view.

The button sends a Gazebo Transport message on /demo/chase, bridged to ROS 2. /demo/mode carries demo status back to the GUI. For headless testing, send the same request from a terminal:

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && ros2 topic pub --once /demo/chase std_msgs/msg/String "{data: chase}"'
```

Train the exploration Q-table, then run the demo (`--chase-table rule` tests it with the rule-based tracker, `--chase-after N` presses Chase automatically):

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.explore --episodes 150 --output experiments/explore_seed0'
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.demo --explore-table experiments/explore_seed0/q_table.npy --chase-table experiments/stage2_seed0/q_table.npy'
```

## Record a Gazebo demonstration

Start the browser GUI. In one terminal, record its virtual display for 90 seconds:

```sh
docker compose exec viewer bash record.sh /project/experiments/gazebo-demo.mp4 90
```

In another terminal, run a policy. This example starts with the NPC visible and demonstrates the rule-based tracking baseline:

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.evaluate --policy rule --episodes 1 --spawn -3 1.6 0 --route workshop_visit --max-steps 240 --output experiments/demo_rule'
```

The recording is saved to experiments/gazebo-demo.mp4 on the host. Replace rule with a saved Q-table path to demonstrate a learned policy. Include the policy name and training budget when presenting a recording.
