# MSML642 机器人搜索与追踪

场景是一个 16 × 12 米的室内设施，中间走廊连通四个带门洞的区域。隔墙、挡板、木箱、油桶、柜子和工作台可参与碰撞及测距。蓝色差速机器人从走廊中间起步，带有 180° 激光测距。Docker Desktop 使用 Linux 容器运行 Ubuntu 24.04、ROS 2 Jazzy 和 Gazebo Harmonic。Gazebo 使用无界面软件渲染，ROS 2 命令在容器内执行。

## 组员快速开始

先安装 Git 和带有 Docker Compose 的 Docker。Windows 用户请启动 Docker Desktop，并使用 Linux 容器。首次启动需要联网下载镜像和依赖。

在终端中执行：

```sh
git clone https://github.com/Gjohnx/MSML642-Group-Project.git
cd MSML642-Group-Project
docker compose --profile gui up -d --build
```

容器启动后，在本机浏览器打开 [Gazebo 图形界面](http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale)。Compose 会启动仿真和浏览器界面，ROS 2 和 Gazebo 都在容器内运行。

## 启动与验证

如果只需要运行无界面仿真，在仓库根目录执行：

```sh
docker compose up -d --build
docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && python3 smoke_test.py'
```

验证脚本从初始位置发出前进、左转、右转和停止命令。它检查里程计和雷达读数，成功时打印 `PASS`。每次重新验证前，执行 `docker compose restart sim` 把机器人放回初始位置。

查看日志或关闭仿真：

```powershell
docker compose logs sim
docker compose down
```

## 查看 Gazebo 图形界面

### Windows 桌面窗口

启动 Docker Desktop，在项目目录执行：

```powershell
docker compose up -d --build desktop
```

在启用了 WSLg 的 Docker Desktop WSL 2 环境中，这条命令会弹出独立的 Gazebo Sim 窗口。窗口默认从斜上方俯瞰地图。关闭窗口后可再运行同一命令打开，也可以用 `docker compose stop desktop` 结束桌面界面。

### 浏览器

```powershell
docker compose --profile gui up -d --build
```

在 Windows 浏览器打开 [Gazebo 图形界面](http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale)。保留窗口，在 PowerShell 中运行上面的验证脚本即可观察机器人动作。关闭浏览器界面服务可执行 `docker compose stop viewer`。


## ROS 2 接口

| 话题 | 类型 | 用途 |
| --- | --- | --- |
| `/robot/cmd_vel` | `geometry_msgs/msg/Twist` | 前进、转向和停止 |
| `/robot/odom` | `nav_msgs/msg/Odometry` | 机器人里程计 |
| `/robot/scan` | `sensor_msgs/msg/LaserScan` | 激光测距 |
| `/clock` | `rosgraph_msgs/msg/Clock` | 仿真时钟 |

下一阶段在这个场景中加入巡逻 NPC 和有限视野检测，再把动作、观测及回合重置接入训练代码。
