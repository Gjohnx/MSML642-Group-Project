# Sim changes needed before training

Three additions to `sim/room.sdf` and `sim/bridge.yaml` that `rl/ros/env.py` depends on.

**1. World systems.** `Contact` is required for contact sensors; `UserCommands` provides `/world/room/set_pose` for fast episode resets. Add next to the other `<plugin>` lines in `room.sdf`:

```xml
<plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
<plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
```

**2. Contact sensor on the learner.** Inside `<link name="base">` of the `robot` model:

```xml
<sensor name="contact" type="contact">
  <always_on>true</always_on>
  <update_rate>50</update_rate>
  <topic>/robot/contact</topic>
  <contact><collision>body</collision></contact>
</sensor>
```

and in `bridge.yaml`:

```yaml
- ros_topic_name: /robot/contact
  gz_topic_name: /robot/contact
  ros_type_name: ros_gz_interfaces/msg/Contacts
  gz_type_name: gz.msgs.Contacts
  direction: GZ_TO_ROS
```

**3. True-pose odometry on the NPC**, so the detector and capture check get its position without wheel drift:

```xml
<plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
  <odom_frame>world</odom_frame>
  <robot_base_frame>npc</robot_base_frame>
  <odom_topic>/npc/odom</odom_topic>
  <odom_publish_frequency>30</odom_publish_frequency>
  <dimensions>2</dimensions>
</plugin>
```

plus a `/npc/odom` entry in `bridge.yaml` (`nav_msgs/msg/Odometry` / `gz.msgs.Odometry`, `GZ_TO_ROS`).

**Optional: 360° lidar** (`min_angle -3.14159`, `max_angle 3.14159`, ~360 samples). This gives collision recovery a rear reading; without it the robot turns in place instead of backing up.
