# Interactive demo and RL validation

This update extends the existing team RL implementation. It provides a working demo and a short validation of the training and evaluation pipeline. It does not establish convergence or a performance advantage for a learned policy.

## Changes

- The Gazebo panel now includes Chase, Stop and Reset. The robot starts in exploration mode and ignores the NPC as an exploration objective. Chase enables pursuit. Stop halts both robots; Reset returns to exploration. Capture and episode limits wait for Reset in interactive mode.
- The panel reports visibility, collisions, steps, simulation time and policy type. All source comments, messages and interface text are English.
- The existing 360-degree, 360-sample lidar and 0.35 m/s learner speed are retained. NPC patrol speed is 0.15 m/s.
- Static obstacles now block target visibility and capture. A robot within 0.5 m of the NPC cannot capture through a wall. Geometry stays in the sensor simulator and never enters the policy state.
- Exploration enforces its own step budget even when it ignores a nearby-target capture signal.
- Training, evaluation, exploration and demo processes share a workspace lock, preventing competing controllers. Transient Gazebo pose-service timeouts are retried up to three times.
- Saved Q-tables have configuration metadata and checksums. Incompatible settings, damaged pairs and invalid arrays are rejected. Historical tables remain usable with a compatibility warning.
- Reports count safety shielding, lost-target search and collision recovery. These interventions can overlap. Old reports without these fields retain missing values.
- The experiment suite supports an optional tracking curriculum before full-map training for each reward variant. Plot legends also work with the Matplotlib version in the project image.

## Run the demo

```sh
git pull
docker compose --profile demo up -d --build
```

Open <http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale>.
The default demo uses rule-based policies and requires no model files. Learned policies can be selected using the `.env` settings in the README. Stop the demo before running training or evaluation:

```sh
docker compose stop demo
```

## Verification

- 53 automated tests passed in the Docker runtime.
- Gazebo task checks passed, including a 0.48 m separation across the north partition that must not count as capture, ordinary capture, and contact recovery.
- The browser GUI was operated with the mouse: Stop displayed the stopped state, Reset restarted exploration with a fresh step count, and Chase led to a captured state. This GUI check used rule policies.
- A second controller lock acquisition from the simulation container was rejected while the demo container owned the lock.
- A short seed-0 suite completed: three reward variants, each with four tracking and four full-map training episodes; six default-start and six held-out-start evaluations for each variant and each of the random and rule baselines. The limit was 60 steps per episode: 84 episodes in total. A separate exploration check ran two 20-step episodes.

Compact suite outputs are in `upgrade_validation/`. These are pilot results from a very small budget, not final comparisons. Early tracking rows may lack the intervention fields added during development. Baseline evaluations were rerun serially after identifying an overlapping controller during development; the controller lock prevents that overlap in future runs.

## Remaining work

Run longer training with several independent seeds, inspect learning curves and compare equal-budget policies on held-out starts and routes. Report capture, detection, rediscovery, collisions and intervention rates together. The short run validates execution and output generation; it is insufficient to conclude that Q-learning outperforms either baseline or that a reward ablation helps.
