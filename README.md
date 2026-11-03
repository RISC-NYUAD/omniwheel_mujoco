# omniwheel_mujoco

Docker-based MuJoCo simulator for the omniwheel robot. The container uses ROS 2 Humble and CycloneDDS;

## Clone
```
git clone --recursive https://github.com/Abanesjo/omniwheel_mujoco
```

## Run

```bash
./build_and_run.sh
```

![omniwheel_mujoco](docs/mujoco.png)

The script builds the image and starts the MuJoCo viewer with ROS 2 publishers/subscribers active.
It uses host networking and CycloneDDS on multicast-enabled loopback, matching the
`omniwheel_ros` Docker setup.

The sensors publish at 100Hz and the physics step runs at 500Hz

![publish_rate](docs/hz.png)

## Control Input

Publish a base-frame wrench, matching the real `omni/` motor torque controller:

```bash
ros2 topic pub -r 100 /omni/cmd_wrench geometry_msgs/msg/Wrench \
"{force: {x: 1.0, y: 0.0, z: 0.0}, torque: {x: 0.0, y: 0.0, z: 0.0}}"
```

The simulator uses `force.x`, `force.y`, and `torque.z`, converts them to wheel torques
with the same Jacobian constants as `omni_motor_bulkctrl_torque.cpp`, saturates each
wheel to `+/-7.1 Nm` (`torque_limit_nm`), and drops the command to zero if none arrives
within `0.2s`. While a wheel is motoring, its torque also follows the XH540-W150-R
torque-speed envelope `|tau| <= 7.1 * V/12 - 7.1 * |w| / 7.33` (`simulate/config.yaml`),
the same model `omniwheel_mjlab` trains with.

Contacts use an elliptic friction cone with `impratio="50"` (`mujoco/scene*.xml`), matching
`omniwheel_mjlab`. With MuJoCo's defaults (pyramidal, 1) the robot turns about 3.4x faster
under the same yaw torque than in training.

## Published Interfaces

- `/odometry/filtered` (`nav_msgs/msg/Odometry`), localization-equivalent MuJoCo truth in `odom -> base_footprint`
- `/tf`, including dynamic `odom -> base_footprint`
- `/omni/wheel_odometry` (`geometry_msgs/msg/TwistWithCovarianceStamped`)
- `/joint_states` (`sensor_msgs/msg/JointState`)
- `/omni/encoder_angles` (`std_msgs/msg/Float32MultiArray`)
- `/clock` (`rosgraph_msgs/msg/Clock`)

The sim does not publish camera or LiDAR topics and does not run `robot_localization`.

## DDS
The project assumes CycloneDDS, which you can edit the config of in the `cyclonedds.xml` file before running the `build_and_run.sh`
