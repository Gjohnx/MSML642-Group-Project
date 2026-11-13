# Simulation integration

The blue learner and red NPC are differential-drive robots. The learner has a 180-degree lidar and a body contact sensor. The NPC follows a configured waypoint loop.

The world loads Physics, SceneBroadcaster, Sensors, UserCommands and Contact. UserCommands provides /world/room/set_pose for episode resets. The bridge publishes wheel odometry, lidar, contact events and world-frame ground truth. Ground truth is used by the limited-FOV sensor simulator, reset checks and experiment metrics. Policy state contains only the discrete local observations from the proposal.

ChaseEnv pauses patrol, stops both robots, sets their poses, waits for the ROS bridge to report those poses, then settles the physics and resumes patrol. Post-action observations must have fresh timestamps. A failed reset or stalled simulation raises an error.

The patrol node accepts route names or "pause" on /npc/route. Only one training or evaluation process should control the simulation at a time. The browser GUI may remain open during experiments.
