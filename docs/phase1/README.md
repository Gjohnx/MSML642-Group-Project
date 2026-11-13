# Phase 1 evidence

MSML642_Phase1_Report.pdf is the two-page progress report. The matching Markdown file is editable source. MSML642_Phase1_Demo.mp4 is a 54-second recording of the Gazebo GUI. It contains 34 seconds from a rule-based capture run followed by 20 seconds from an early Q-learning training episode. There are no captions, title cards or narration.

The rule-based video run starts at [-3, 1.6, 0] with the NPC initially visible on workshop_visit. It captured the NPC after 83 actions and 25.636 simulation seconds, with zero separate collisions. The report JSON and episode CSV are in evidence/phase1_demo_rule.

The Q-learning debugging run used seed 0, 12 episodes, a 140-step limit, the same initial pose and route, and epsilon decay over eight episodes. It produced a Q-table but no captures. Its measurements are in evidence/track_train_seed0. The training video excerpt comes from an early exploratory episode in that run.

The suite subdirectory contains pipeline checks with seeds 0 and 1. Each Q-learning variant trained for two episodes and was evaluated for two episodes, with a 60-step limit. The variants are full rewards, no new-region bonus, and no rediscovery bonus. Random and rule-based policies were also evaluated. These runs validate the experiment commands and output files. They are too short to assess convergence or rank policies. The effective configurations are included in each report JSON.

The spawn_route_check measurements exercise all six combinations of the three configured learner spawns and two patrol routes, with a 20-step limit. These are reset and interface checks, not a full-route benchmark.

unit-tests.txt records 22 passing tests. gazebo-check.txt records the integration check for visibility, hidden observations, occlusion, resets, moving-target capture, contacts and recovery.
