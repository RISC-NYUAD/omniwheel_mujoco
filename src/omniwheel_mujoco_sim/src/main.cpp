// Copyright 2026 NYUAD
//
// Licensed under the Apache License, Version 2.0.

#define private public
#include "glfw_adapter.h"
#undef private

#include <mujoco/mujoco.h>

#include "array_safety.h"
#include "simulate.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/wrench.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/transform_broadcaster.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

constexpr double kPi = 3.14159265358979323846;
constexpr int kErrorLength = 1024;
constexpr double kSyncMisalign = 0.1;
constexpr double kSimRefreshFraction = 0.7;
using Seconds = std::chrono::duration<double>;

mjModel* g_model = nullptr;
mjData* g_data = nullptr;
mjtNum* g_ctrl_noise = nullptr;

struct SimulationConfig {
  std::filesystem::path scene_file{"/workspace/omniwheel_mujoco/mujoco/scene.xml"};
  double publish_rate_hz{100.0};
  double command_timeout_sec{0.2};
  double torque_limit_nm{4.7};
  int encoder_ticks_per_rev{4096};
  bool fixed_pendulum{false};
  bool print_scene_information{true};
};

SimulationConfig g_config;

std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#else
  const char* path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == static_cast<std::size_t>(-1)) {
        if (errno == EINVAL) {
          return path;
        }
        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }
  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }
  return "";
}

void scanPluginLibraries() {
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = executable_dir + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}

void loadConfig(const std::filesystem::path& config_file) {
  if (config_file.empty() || !std::filesystem::exists(config_file)) {
    std::cerr << "Config file not found, using defaults: " << config_file << "\n";
    return;
  }

  const YAML::Node cfg = YAML::LoadFile(config_file.string());
  if (cfg["scene_file"]) {
    g_config.scene_file = cfg["scene_file"].as<std::string>();
  }
  if (cfg["publish_rate_hz"]) {
    g_config.publish_rate_hz = cfg["publish_rate_hz"].as<double>();
  }
  if (cfg["command_timeout_sec"]) {
    g_config.command_timeout_sec = cfg["command_timeout_sec"].as<double>();
  }
  if (cfg["torque_limit_nm"]) {
    g_config.torque_limit_nm = cfg["torque_limit_nm"].as<double>();
  }
  if (cfg["encoder_ticks_per_rev"]) {
    g_config.encoder_ticks_per_rev = cfg["encoder_ticks_per_rev"].as<int>();
  }
  if (cfg["fixed_pendulum"]) {
    g_config.fixed_pendulum = cfg["fixed_pendulum"].as<bool>();
  }
  if (cfg["print_scene_information"]) {
    g_config.print_scene_information = cfg["print_scene_information"].as<bool>();
  }
}

void parseArguments(int argc, char** argv, std::filesystem::path* config_file) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--config" && i + 1 < argc) {
      *config_file = argv[++i];
    } else if (arg == "--scene" && i + 1 < argc) {
      g_config.scene_file = argv[++i];
    }
  }
}

double clamp(double value, double low, double high) {
  return std::min(std::max(value, low), high);
}

std::array<double, 3> baseWrenchToMotorTorquesLeftRightBack(double force_x, double force_y,
                                                            double torque_z) {
  // Closed form of omni_motor_bulkctrl_torque.cpp's J_torque matrix.
  // Output order is XML actuator order: rim_left_motor, rim_right_motor, rim_back_motor.
  constexpr double wheel_radius_m = 0.101;
  constexpr double base_radius_m = 0.1285;
  const double torque_z_coeff = wheel_radius_m / (3.0 * base_radius_m);

  return {
      (-wheel_radius_m / std::sqrt(3.0)) * force_x + (wheel_radius_m / 3.0) * force_y +
          torque_z_coeff * torque_z,
      (wheel_radius_m / std::sqrt(3.0)) * force_x + (wheel_radius_m / 3.0) * force_y +
          torque_z_coeff * torque_z,
      -(2.0 * wheel_radius_m / 3.0) * force_y + torque_z_coeff * torque_z,
  };
}

builtin_interfaces::msg::Time stampFromSimTime(double sim_time) {
  builtin_interfaces::msg::Time stamp;
  const double clamped = std::max(0.0, sim_time);
  stamp.sec = static_cast<int32_t>(std::floor(clamped));
  stamp.nanosec = static_cast<uint32_t>((clamped - static_cast<double>(stamp.sec)) * 1e9);
  return stamp;
}

std::array<double, 4> yawQuaternion(double yaw) {
  const double half = 0.5 * yaw;
  return {0.0, 0.0, std::sin(half), std::cos(half)};
}

double yawFromMujocoQuat(const mjtNum* quat_wxyz) {
  const double w = quat_wxyz[0];
  const double x = quat_wxyz[1];
  const double y = quat_wxyz[2];
  const double z = quat_wxyz[3];
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

double quantizePosition(double position, int ticks_per_rev) {
  if (ticks_per_rev <= 0) {
    return position;
  }
  const double ticks = std::round(position / (2.0 * kPi) * ticks_per_rev);
  return ticks * (2.0 * kPi) / ticks_per_rev;
}

int nameToIdOrThrow(const mjModel* model, int object_type, const char* name) {
  const int id = mj_name2id(model, object_type, name);
  if (id < 0) {
    throw std::runtime_error(std::string("MuJoCo object not found: ") + name);
  }
  return id;
}

mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);
  if (!filename[0]) {
    return nullptr;
  }

  char loadError[kErrorLength] = "";
  mjModel* mnew = nullptr;
  if (mju::strlen_arr(filename) > 4 &&
      !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                    mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }

  mju::strcpy_arr(sim.load_error, loadError);
  if (!mnew) {
    std::printf("%s\n", loadError);
    return nullptr;
  }

  if (loadError[0]) {
    std::printf("Model compiled, but simulation warning:\n  %s\n", loadError);
  }
  return mnew;
}

class OmniwheelRosBridge : public rclcpp::Node {
 public:
  explicit OmniwheelRosBridge(mj::Simulate* sim)
      : Node("omniwheel_mujoco_bridge"), sim_(sim) {
    using namespace std::chrono_literals;

    wrench_sub_ = create_subscription<geometry_msgs::msg::Wrench>(
        "/omni/cmd_wrench", rclcpp::QoS(10),
        [this](const geometry_msgs::msg::Wrench::SharedPtr msg) {
          const auto motor_torque_lrb =
              baseWrenchToMotorTorquesLeftRightBack(msg->force.x, msg->force.y, msg->torque.z);

          std::lock_guard<std::mutex> lock(command_mutex_);
          for (std::size_t i = 0; i < 3; ++i) {
            target_torque_nm_[i] =
                clamp(motor_torque_lrb[i], -g_config.torque_limit_nm, g_config.torque_limit_nm);
          }
          last_command_time_ = std::chrono::steady_clock::now();
          command_received_ = true;
        });

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/filtered", rclcpp::QoS(10));
    wheel_odom_pub_ =
        create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/omni/wheel_odometry",
                                                                         rclcpp::QoS(10));
    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", rclcpp::QoS(10));
    encoder_pub_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/omni/encoder_angles", rclcpp::QoS(10));
    clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    const auto period =
        std::chrono::duration<double>(1.0 / std::max(1.0, g_config.publish_rate_hz));
    publish_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&OmniwheelRosBridge::publishState, this));
  }

  void applyControls(mjModel* model, mjData* data) {
    if (!initialized_) {
      initialize(model);
    }

    std::array<double, 3> desired{};
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      desired = target_torque_nm_;
      if (!command_received_ ||
          std::chrono::duration<double>(std::chrono::steady_clock::now() - last_command_time_)
                  .count() > g_config.command_timeout_sec) {
        desired = {0.0, 0.0, 0.0};
      }
    }

    for (std::size_t i = 0; i < 3; ++i) {
      applied_torque_nm_[i] =
          clamp(desired[i], -g_config.torque_limit_nm, g_config.torque_limit_nm);
      data->ctrl[actuator_ids_[i]] = applied_torque_nm_[i];
    }
  }

  void enforceFixedPendulum(mjModel* model, mjData* data, bool recompute) {
    if (!g_config.fixed_pendulum) {
      return;
    }
    if (!initialized_) {
      initialize(model);
    }

    const int qpos = model->jnt_qposadr[pendulum_ball_joint_id_];
    const int qvel = model->jnt_dofadr[pendulum_ball_joint_id_];
    const auto zero_pendulum_state = [&]() {
      data->qpos[qpos + 0] = 1.0;
      data->qpos[qpos + 1] = 0.0;
      data->qpos[qpos + 2] = 0.0;
      data->qpos[qpos + 3] = 0.0;

      data->qvel[qvel + 0] = 0.0;
      data->qvel[qvel + 1] = 0.0;
      data->qvel[qvel + 2] = 0.0;
      data->qacc[qvel + 0] = 0.0;
      data->qacc[qvel + 1] = 0.0;
      data->qacc[qvel + 2] = 0.0;
      data->qacc_warmstart[qvel + 0] = 0.0;
      data->qacc_warmstart[qvel + 1] = 0.0;
      data->qacc_warmstart[qvel + 2] = 0.0;
    };

    zero_pendulum_state();
    if (recompute) {
      mj_forward(model, data);
      zero_pendulum_state();
    }
  }

  void publishClock(mjtNum sim_time) {
    rosgraph_msgs::msg::Clock clock_msg;
    clock_msg.clock = stampFromSimTime(sim_time);
    clock_pub_->publish(clock_msg);
  }

 private:
  void initialize(mjModel* model) {
    actuator_ids_[0] = nameToIdOrThrow(model, mjOBJ_ACTUATOR, "rim_left_motor");
    actuator_ids_[1] = nameToIdOrThrow(model, mjOBJ_ACTUATOR, "rim_right_motor");
    actuator_ids_[2] = nameToIdOrThrow(model, mjOBJ_ACTUATOR, "rim_back_motor");

    wheel_joint_ids_[0] = nameToIdOrThrow(model, mjOBJ_JOINT, "rim_left_joint");
    wheel_joint_ids_[1] = nameToIdOrThrow(model, mjOBJ_JOINT, "rim_right_joint");
    wheel_joint_ids_[2] = nameToIdOrThrow(model, mjOBJ_JOINT, "rim_back_joint");
    pendulum_ball_joint_id_ = nameToIdOrThrow(model, mjOBJ_JOINT, "pendulum_ball_joint");
    base_freejoint_id_ = nameToIdOrThrow(model, mjOBJ_JOINT, "base_freejoint");

    if (g_config.print_scene_information) {
      printSceneInformation(model);
    }

    initialized_ = true;
  }

  void printSceneInformation(const mjModel* model) {
    std::cout << "<<------------- Actuators ------------->>\n";
    for (int i = 0; i < model->nu; ++i) {
      if (const char* name = mj_id2name(model, mjOBJ_ACTUATOR, i)) {
        std::cout << "Actuator_index: " << i << ", name: " << name << "\n";
      }
    }
    std::cout << "<<------------- Joints ------------->>\n";
    for (int i = 0; i < model->njnt; ++i) {
      if (const char* name = mj_id2name(model, mjOBJ_JOINT, i)) {
        std::cout << "Joint_index: " << i << ", name: " << name << "\n";
      }
    }
  }

  void publishState() {
    if (!sim_ || !g_model || !g_data) {
      return;
    }

    mj::MutexLock lock(sim_->mtx);
    if (!g_model || !g_data) {
      return;
    }
    if (!initialized_) {
      initialize(g_model);
    }

    const auto stamp = stampFromSimTime(g_data->time);

    const int base_qpos = g_model->jnt_qposadr[base_freejoint_id_];
    const int base_qvel = g_model->jnt_dofadr[base_freejoint_id_];

    const double x = g_data->qpos[base_qpos + 0];
    const double y = g_data->qpos[base_qpos + 1];
    const double yaw = yawFromMujocoQuat(&g_data->qpos[base_qpos + 3]);
    const auto odom_quat = yawQuaternion(yaw);

    const double vx_world = g_data->qvel[base_qvel + 0];
    const double vy_world = g_data->qvel[base_qvel + 1];
    const double yaw_rate = g_data->qvel[base_qvel + 5];
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double vx_body = cos_yaw * vx_world + sin_yaw * vy_world;
    const double vy_body = -sin_yaw * vx_world + cos_yaw * vy_world;

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = "odom";
    tf_msg.child_frame_id = "base_footprint";
    tf_msg.transform.translation.x = x;
    tf_msg.transform.translation.y = y;
    tf_msg.transform.translation.z = 0.0;
    tf_msg.transform.rotation.x = odom_quat[0];
    tf_msg.transform.rotation.y = odom_quat[1];
    tf_msg.transform.rotation.z = odom_quat[2];
    tf_msg.transform.rotation.w = odom_quat[3];
    tf_broadcaster_->sendTransform(tf_msg);

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_footprint";
    odom_msg.pose.pose.position.x = x;
    odom_msg.pose.pose.position.y = y;
    odom_msg.pose.pose.position.z = 0.0;
    odom_msg.pose.pose.orientation.x = odom_quat[0];
    odom_msg.pose.pose.orientation.y = odom_quat[1];
    odom_msg.pose.pose.orientation.z = odom_quat[2];
    odom_msg.pose.pose.orientation.w = odom_quat[3];
    odom_msg.twist.twist.linear.x = vx_body;
    odom_msg.twist.twist.linear.y = vy_body;
    odom_msg.twist.twist.angular.z = yaw_rate;
    setDiagonalCovariance(odom_msg.pose.covariance, 1e-4, 1e-4, 1.0, 1.0, 1.0, 1e-4);
    setDiagonalCovariance(odom_msg.twist.covariance, 1e-3, 1e-3, 1.0, 1.0, 1.0, 1e-3);
    odom_pub_->publish(odom_msg);

    publishWheelOdometry(stamp);
    publishJointStates(stamp);
    publishEncoderAngles();
  }

  template <typename CovarianceArray>
  void setDiagonalCovariance(CovarianceArray& covariance, double x, double y, double z, double roll,
                             double pitch, double yaw) {
    covariance.fill(0.0);
    covariance[0] = x;
    covariance[7] = y;
    covariance[14] = z;
    covariance[21] = roll;
    covariance[28] = pitch;
    covariance[35] = yaw;
  }

  std::array<double, 3> wheelPositionsLeftRightBack() const {
    std::array<double, 3> positions{};
    for (std::size_t i = 0; i < 3; ++i) {
      const int qpos = g_model->jnt_qposadr[wheel_joint_ids_[i]];
      positions[i] = quantizePosition(g_data->qpos[qpos], g_config.encoder_ticks_per_rev);
    }
    return positions;
  }

  std::array<double, 3> wheelVelocitiesLeftRightBack() const {
    std::array<double, 3> velocities{};
    for (std::size_t i = 0; i < 3; ++i) {
      const int qvel = g_model->jnt_dofadr[wheel_joint_ids_[i]];
      velocities[i] = g_data->qvel[qvel];
    }
    return velocities;
  }

  std::array<double, 2> pendulumAngles() const {
    if (g_config.fixed_pendulum) {
      return {0.0, 0.0};
    }

    const int qpos = g_model->jnt_qposadr[pendulum_ball_joint_id_];
    mjtNum quat[4] = {g_data->qpos[qpos + 0], g_data->qpos[qpos + 1], g_data->qpos[qpos + 2],
                      g_data->qpos[qpos + 3]};
    mju_normalize4(quat);

    mjtNum angle_axis[3];
    mju_quat2Vel(angle_axis, quat, 1.0);

    // The old exported model used two hinge encoders. In pendulum_link1's local frame,
    // Pendulum_Joint_1 was -Z and Pendulum_Joint_2 was -X.
    return {-angle_axis[2], -angle_axis[0]};
  }

  std::array<double, 2> pendulumVelocities() const {
    if (g_config.fixed_pendulum) {
      return {0.0, 0.0};
    }

    const int qvel = g_model->jnt_dofadr[pendulum_ball_joint_id_];
    return {-g_data->qvel[qvel + 2], -g_data->qvel[qvel + 0]};
  }

  void publishWheelOdometry(const builtin_interfaces::msg::Time& stamp) {
    const auto w_lrb = wheelVelocitiesLeftRightBack();

    // Real joint-state order is Wheel_1, Wheel_2, Wheel_3. The sim command order is
    // left, right, back, so Wheel_1 maps to right and Wheel_2 maps to left.
    const double w1_right = w_lrb[1];
    const double w2_left = w_lrb[0];
    const double w3_back = w_lrb[2];

    constexpr double alpha = 0.101;
    constexpr double L = 0.1285;

    geometry_msgs::msg::TwistWithCovarianceStamped wheel_odom;
    wheel_odom.header.stamp = stamp;
    wheel_odom.header.frame_id = "base_link";
    wheel_odom.twist.twist.linear.x =
        alpha * (-(std::sqrt(3.0) / 3.0) * w1_right + (std::sqrt(3.0) / 3.0) * w2_left);
    wheel_odom.twist.twist.linear.y =
        alpha * ((2.0 / 3.0) * w3_back - (1.0 / 3.0) * w1_right - (1.0 / 3.0) * w2_left);
    wheel_odom.twist.twist.angular.z =
        -alpha * ((w1_right + w2_left + w3_back) / (3.0 * L));
    setDiagonalCovariance(wheel_odom.twist.covariance, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
    wheel_odom_pub_->publish(wheel_odom);
  }

  void publishJointStates(const builtin_interfaces::msg::Time& stamp) {
    const auto wheel_pos_lrb = wheelPositionsLeftRightBack();
    const auto wheel_vel_lrb = wheelVelocitiesLeftRightBack();
    const auto pendulum_pos = pendulumAngles();
    const auto pendulum_vel = pendulumVelocities();

    sensor_msgs::msg::JointState joint_state;
    joint_state.header.stamp = stamp;
    joint_state.name = {"Wheel_1_RotJoint", "Wheel_2_RotJoint", "Wheel_3_RotJoint",
                        "Pendulum_Joint_1", "Pendulum_Joint_2"};
    joint_state.position.resize(joint_state.name.size(), 0.0);
    joint_state.velocity.resize(joint_state.name.size(), 0.0);
    joint_state.effort.resize(joint_state.name.size(), 0.0);

    joint_state.position[0] = wheel_pos_lrb[1];
    joint_state.position[1] = wheel_pos_lrb[0];
    joint_state.position[2] = wheel_pos_lrb[2];
    joint_state.velocity[0] = wheel_vel_lrb[1];
    joint_state.velocity[1] = wheel_vel_lrb[0];
    joint_state.velocity[2] = wheel_vel_lrb[2];
    joint_state.effort[0] = applied_torque_nm_[1];
    joint_state.effort[1] = applied_torque_nm_[0];
    joint_state.effort[2] = applied_torque_nm_[2];

    joint_state.position[3] = pendulum_pos[0];
    joint_state.position[4] = pendulum_pos[1];
    joint_state.velocity[3] = pendulum_vel[0];
    joint_state.velocity[4] = pendulum_vel[1];

    joint_state_pub_->publish(joint_state);
  }

  void publishEncoderAngles() {
    constexpr std::array<std::array<float, 2>, 2> encoder_limits = {
        std::array<float, 2>{0.353988F, 0.705674F},
        std::array<float, 2>{2.261226F, 2.611762F}};
    const float zero0 = encoder_limits[1][0] + (encoder_limits[1][1] - encoder_limits[1][0]) / 2.0F;
    const float zero1 = encoder_limits[0][0] + (encoder_limits[0][1] - encoder_limits[0][0]) / 2.0F;

    std_msgs::msg::Float32MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].size = 2;
    msg.layout.dim[0].stride = 1;
    msg.layout.dim[0].label = "encoder_values";
    msg.data.resize(6, 0.0F);

    const auto pendulum_pos = pendulumAngles();
    msg.data[0] = static_cast<float>(pendulum_pos[0]) + zero0;
    msg.data[1] = static_cast<float>(pendulum_pos[1]) + zero1;
    msg.data[2] = encoder_limits[1][0];
    msg.data[3] = encoder_limits[1][1];
    msg.data[4] = encoder_limits[0][0];
    msg.data[5] = encoder_limits[0][1];
    encoder_pub_->publish(msg);
  }

  mj::Simulate* sim_;
  bool initialized_{false};
  std::array<int, 3> actuator_ids_{-1, -1, -1};
  std::array<int, 3> wheel_joint_ids_{-1, -1, -1};
  int pendulum_ball_joint_id_{-1};
  int base_freejoint_id_{-1};

  std::mutex command_mutex_;
  std::array<double, 3> target_torque_nm_{0.0, 0.0, 0.0};
  std::array<double, 3> applied_torque_nm_{0.0, 0.0, 0.0};
  std::chrono::steady_clock::time_point last_command_time_{std::chrono::steady_clock::now()};
  bool command_received_{false};

  rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr wrench_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr wheel_odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr encoder_pub_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

std::shared_ptr<OmniwheelRosBridge> g_bridge;

void stepPhysics(mjModel* model, mjData* data) {
  if (g_bridge) {
    g_bridge->enforceFixedPendulum(model, data, false);
    g_bridge->applyControls(model, data);
  }
  mj_step(model, data);
  if (g_bridge) {
    g_bridge->enforceFixedPendulum(model, data, true);
    g_bridge->publishClock(data->time);
  }
}

void PhysicsLoop(mj::Simulate& sim) {
  std::chrono::time_point<mj::Simulate::Clock> sync_cpu;
  mjtNum sync_sim = 0;

  while (!sim.exitrequest.load() && rclcpp::ok()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) {
        dnew = mj_makeData(mnew);
      }
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);
        mj_deleteData(g_data);
        mj_deleteModel(g_model);
        g_model = mnew;
        g_data = dnew;
        mj_forward(g_model, g_data);
        if (g_bridge) {
          g_bridge->enforceFixedPendulum(g_model, g_data, true);
        }
        free(g_ctrl_noise);
        g_ctrl_noise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * g_model->nu));
        mju_zero(g_ctrl_noise, g_model->nu);
      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel* mnew = LoadModel(sim.filename, sim);
      mjData* dnew = nullptr;
      if (mnew) {
        dnew = mj_makeData(mnew);
      }
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);
        mj_deleteData(g_data);
        mj_deleteModel(g_model);
        g_model = mnew;
        g_data = dnew;
        mj_forward(g_model, g_data);
        if (g_bridge) {
          g_bridge->enforceFixedPendulum(g_model, g_data, true);
        }
        free(g_ctrl_noise);
        g_ctrl_noise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * g_model->nu));
        mju_zero(g_ctrl_noise, g_model->nu);
      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      mj::MutexLock lock(sim.mtx);
      if (!g_model || !g_data) {
        continue;
      }

      if (sim.run) {
        bool stepped = false;
        const auto start_cpu = mj::Simulate::Clock::now();
        const auto elapsed_cpu = start_cpu - sync_cpu;
        const double elapsed_sim = g_data->time - sync_sim;
        const double slowdown = 100.0 / sim.percentRealTime[sim.real_time_index];
        const bool misaligned =
            std::abs(Seconds(elapsed_cpu).count() / slowdown - elapsed_sim) > kSyncMisalign;

        if (elapsed_sim < 0 || elapsed_cpu.count() < 0 ||
            sync_cpu.time_since_epoch().count() == 0 || misaligned || sim.speed_changed) {
          sync_cpu = start_cpu;
          sync_sim = g_data->time;
          sim.speed_changed = false;

          stepPhysics(g_model, g_data);
          stepped = true;
        } else {
          bool measured = false;
          const mjtNum previous_sim = g_data->time;
          const double refresh_time = kSimRefreshFraction / sim.refresh_rate;

          while (Seconds((g_data->time - sync_sim) * slowdown) <
                     mj::Simulate::Clock::now() - sync_cpu &&
                 mj::Simulate::Clock::now() - start_cpu < Seconds(refresh_time)) {
            if (!measured && elapsed_sim) {
              sim.measured_slowdown = Seconds(elapsed_cpu).count() / elapsed_sim;
              measured = true;
            }

            stepPhysics(g_model, g_data);
            stepped = true;

            if (g_data->time < previous_sim) {
              break;
            }
          }
        }

        if (stepped) {
          sim.AddToHistory();
        }
      } else {
        if (g_bridge) {
          g_bridge->enforceFixedPendulum(g_model, g_data, false);
        }
        mj_forward(g_model, g_data);
        if (sim.pause_update) {
          mju_copy(g_data->qacc_warmstart, g_data->qacc, g_model->nv);
        }
        if (g_bridge) {
          g_bridge->enforceFixedPendulum(g_model, g_data, false);
        }
        sim.speed_changed = true;
      }
    }
  }
}

void PhysicsThread(mj::Simulate* sim, const char* filename) {
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    g_model = LoadModel(filename, *sim);
    if (g_model) {
      g_data = mj_makeData(g_model);
    }
    if (g_data) {
      sim->Load(g_model, g_data, filename);
      mj_forward(g_model, g_data);
      if (g_bridge) {
        g_bridge->enforceFixedPendulum(g_model, g_data, true);
      }
      free(g_ctrl_noise);
      g_ctrl_noise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * g_model->nu));
      mju_zero(g_ctrl_noise, g_model->nu);
    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  free(g_ctrl_noise);
  g_ctrl_noise = nullptr;
  mj_deleteData(g_data);
  g_data = nullptr;
  mj_deleteModel(g_model);
  g_model = nullptr;
}

void userKeyCallback(GLFWwindow* /*window*/, int key, int /*scancode*/, int act, int /*mods*/) {
  if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE && g_model && g_data) {
    mj_resetData(g_model, g_data);
    if (g_bridge) {
      g_bridge->enforceFixedPendulum(g_model, g_data, false);
    }
    mj_forward(g_model, g_data);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path config_file{"/workspace/omniwheel_mujoco/simulate/config.yaml"};
  parseArguments(argc, argv, &config_file);
  loadConfig(config_file);

  rclcpp::init(argc, argv);

  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);
  mjvOption opt;
  mjv_defaultOption(&opt);
  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  auto sim = std::make_unique<mj::Simulate>(
      std::make_unique<mj::GlfwAdapter>(), &cam, &opt, &pert, false);

  g_bridge = std::make_shared<OmniwheelRosBridge>(sim.get());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(g_bridge);
  std::thread ros_thread([&executor]() { executor.spin(); });

  std::thread physics_thread(&PhysicsThread, sim.get(), g_config.scene_file.c_str());

  glfwSetKeyCallback(static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_, userKeyCallback);
  sim->RenderLoop();
  sim->exitrequest.store(true);

  if (physics_thread.joinable()) {
    physics_thread.join();
  }
  executor.cancel();
  if (ros_thread.joinable()) {
    ros_thread.join();
  }
  executor.remove_node(g_bridge);
  g_bridge.reset();
  rclcpp::shutdown();
  return 0;
}
