"""Scripted NPC patrol (stage-1 TODO from the README).

Needs first: an `npc` model in room.sdf (copy of `robot`, different color, its own
DiffDrive on /npc/cmd_vel and /npc/odom, no lidar) and matching entries in bridge.yaml.

Then: follow one route from npc_routes.yaml with a simple go-to-waypoint
controller on /npc/odom -> /npc/cmd_vel. It moves but never evades.

    python3 npc_patrol.py --route corridor_loop
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
