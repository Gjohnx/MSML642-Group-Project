# MSML642 Robot Search and Tracking

The scene is a 16 × 12 m indoor facility. A central corridor connects four areas through doorways. Partitions, barriers, crates, drums, cabinets and a workbench all take part in collisions and range sensing. A blue differential-drive robot starts in the middle of the corridor and carries a 180° laser rangefinder. Docker Desktop runs Ubuntu 24.04, ROS 2 Jazzy and Gazebo Harmonic in a Linux container. Gazebo uses headless software rendering, and ROS 2 commands run inside the container.

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

The next stage adds a patrolling NPC and limited field-of-view detection to this scene, then connects actions, observations and episode reset to the training code.

## Project structure

```
sim/                  Gazebo scene and ROS 2 bridge (mounted read-only in the container)
  room.sdf            Map and learner robot
  bridge.yaml         ROS 2 <-> Gazebo topics
  run.sh / gui.sh / view.sh
  smoke_test.py       Motion and lidar check
  npc_patrol.py       NPC patrol node (to be implemented)
  npc_routes.yaml     NPC patrol routes
rl/                   Reinforcement learning code (mounted at /project/rl)
  core/               ROS-free: state discretization, Q-table, rewards,
                      lost-target search, collision recovery, baselines
  ros/                env.py (reset/step interface), detector.py (limited-FOV detector)
  config/             qlearning.yaml, rewards.yaml
  tests/              Unit tests for core/
  train.py / evaluate.py
experiments/          Training outputs (not committed; mounted at /project/experiments)
analysis/             Plotting scripts for the report
docs/                 Proposal, report, figures
```

`rl/core/` does not import `rclpy`, so it can be tested without starting the simulation. Training runs inside the container:

```sh
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && python3 -m rl.train'
docker compose exec sim bash -c 'cd /project && python3 -m pytest rl/tests'
```
