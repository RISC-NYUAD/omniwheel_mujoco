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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iterator>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <random>
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
  double torque_limit_nm{7.1};
  // DC-motor torque-speed envelope (XH540-W150-R datasheet at 12 V), matching
  // omniwheel_mjlab training. While motoring:
  //   |tau| <= stall * V / V_n - stall * |w| / w_n.
  bool torque_speed_limit{true};
  double supply_voltage_v{12.0};
  double motor_nominal_voltage_v{12.0};
  double motor_stall_torque_nm{7.1};
  double motor_no_load_speed_rad_s{70.0 * 2.0 * M_PI / 60.0};
  int encoder_ticks_per_rev{4096};
  bool fixed_pendulum{false};
  bool print_scene_information{true};
};

SimulationConfig g_config;

constexpr double kDegToRad = kPi / 180.0;
// Policy action units are wrench / scale (ActionScaling in omniwheel_ros policy.py).
constexpr double kActionForceScaleN = 100.0;
constexpr double kActionYawTorqueScaleNm = 13.5;
constexpr int kRollersPerWheel = 36;

// Domain-randomization knobs shown as viewer sliders, mirroring the omniwheel_mjlab training
// randomizations (run ft_t7). Each default is the midpoint of its slider range: for per-episode
// values that is the midpoint of the training range, for per-step noise it is the training
// amplitude (the sliders go to twice that). Guarded by Simulate::mtx: the viewer writes it while
// handling events, the physics thread reads it every step and the ROS thread on every publish.
struct DomainParams {
  // Robot physics.
  mjtNum mass_scale = 1.0;                      // base mass and inertia
  mjtNum com_offset_cm[3] = {0.0, 0.0, 1.0};    // base COM, body frame
  mjtNum pendulum_max_angle_deg = 10.0;         // ball-joint cone limit
  mjtNum pendulum_damping = 0.05;
  mjtNum pendulum_friction = 0.01;
  mjtNum rim_damping = 0.025;
  mjtNum rim_friction = 0.01;
  mjtNum roller_damping = 0.0005;
  mjtNum roller_friction = 0.01;
  mjtNum ground_friction = 1.2;                 // wheel-ground sliding friction

  // Motors. Per-wheel arrays are left, right, back; action arrays are fx, fy, tz.
  mjtNum torque_strength[3] = {1.0, 1.0, 1.0};
  mjtNum torque_offset_nm[3] = {0.0, 0.0, 0.0};
  mjtNum motor_lag_ms = 10.0;
  mjtNum action_delay_ms = 1000.0 / 60.0;
  mjtNum action_offset[3] = {0.0, 0.0, 0.0};    // policy action units
  mjtNum action_jitter_std = 0.02;              // policy action units, Gaussian
  mjtNum action_hold_pct = 1.0;

  // Sensors. Pendulum arrays follow Pendulum_Joint_1, Pendulum_Joint_2.
  mjtNum pendulum_angle_noise_deg = 0.35;       // uniform +-
  mjtNum pendulum_rate_noise_deg_s = 3.0;       // uniform +-
  mjtNum pendulum_angle_offset_deg[2] = {0.0, 0.0};
  mjtNum pendulum_rate_offset_deg_s[2] = {0.0, 0.0};
  mjtNum pendulum_delay_ms = 25.0;
  mjtNum pendulum_hold_pct = 1.0;
  mjtNum base_vel_noise = 0.03;                 // uniform +-, m/s and rad/s
  mjtNum base_vel_offset[3] = {0.0, 0.0, 0.0};  // vx, vy (m/s), wz (rad/s)
  mjtNum base_vel_delay_ms = 1000.0 / 60.0;
  mjtNum base_vel_hold_pct = 0.5;
};

DomainParams g_params;

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
  if (cfg["torque_speed_limit"]) {
    g_config.torque_speed_limit = cfg["torque_speed_limit"].as<bool>();
  }
  if (cfg["supply_voltage_v"]) {
    g_config.supply_voltage_v = cfg["supply_voltage_v"].as<double>();
  }
  if (cfg["motor_nominal_voltage_v"]) {
    g_config.motor_nominal_voltage_v = cfg["motor_nominal_voltage_v"].as<double>();
  }
  if (cfg["motor_stall_torque_nm"]) {
    g_config.motor_stall_torque_nm = cfg["motor_stall_torque_nm"].as<double>();
  }
  if (cfg["motor_no_load_speed_rad_s"]) {
    g_config.motor_no_load_speed_rad_s = cfg["motor_no_load_speed_rad_s"].as<double>();
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

// Writes the "Robot physics" sliders into the model, the way training's reset events do: base
// mass and inertia scaled and the COM offset relative to the MJCF (then mj_setConst, as mjlab
// does), and absolute joint damping, friction loss and pendulum cone limit. Ground friction goes
// to both the wheel contact geoms and the floor, because MuJoCo uses the larger of the two
// geoms' values and the floor would otherwise mask low settings.
class ModelRandomizer {
 public:
  ~ModelRandomizer() {
    if (scratch_) {
      mj_deleteData(scratch_);
    }
  }

  void apply(mjModel* m, const DomainParams& p) {
    if (m != model_) {
      capture(m);
    }

    if (pendulum_joint_ >= 0) {
      m->jnt_range[2 * pendulum_joint_ + 1] = p.pendulum_max_angle_deg * kDegToRad;
      const int dof = m->jnt_dofadr[pendulum_joint_];
      for (int i = 0; i < 3; ++i) {
        m->dof_damping[dof + i] = p.pendulum_damping;
        m->dof_frictionloss[dof + i] = p.pendulum_friction;
      }
    }
    for (const int dof : rim_dofs_) {
      m->dof_damping[dof] = p.rim_damping;
      m->dof_frictionloss[dof] = p.rim_friction;
    }
    for (const int dof : roller_dofs_) {
      m->dof_damping[dof] = p.roller_damping;
      m->dof_frictionloss[dof] = p.roller_friction;
    }
    for (const int geom : contact_geoms_) {
      m->geom_friction[3 * geom] = p.ground_friction;
    }

    const bool inertial_changed =
        p.mass_scale != applied_mass_scale_ ||
        !std::equal(std::begin(p.com_offset_cm), std::end(p.com_offset_cm),
                    std::begin(applied_com_offset_cm_));
    if (base_body_ >= 0 && (inertial_dirty_ || inertial_changed)) {
      m->body_mass[base_body_] = nominal_mass_ * p.mass_scale;
      for (int k = 0; k < 3; ++k) {
        m->body_inertia[3 * base_body_ + k] = nominal_inertia_[k] * p.mass_scale;
        m->body_ipos[3 * base_body_ + k] = nominal_ipos_[k] + p.com_offset_cm[k] / 100.0;
      }
      mj_setConst(m, scratch_);  // scratch data: mj_setConst overwrites the state it is given
      applied_mass_scale_ = p.mass_scale;
      std::copy(std::begin(p.com_offset_cm), std::end(p.com_offset_cm),
                std::begin(applied_com_offset_cm_));
      inertial_dirty_ = false;
    }
  }

 private:
  void capture(mjModel* m) {
    model_ = m;
    if (scratch_) {
      mj_deleteData(scratch_);
    }
    scratch_ = mj_makeData(m);
    inertial_dirty_ = true;

    base_body_ = mj_name2id(m, mjOBJ_BODY, "base_link");
    if (base_body_ >= 0) {
      nominal_mass_ = m->body_mass[base_body_];
      for (int k = 0; k < 3; ++k) {
        nominal_inertia_[k] = m->body_inertia[3 * base_body_ + k];
        nominal_ipos_[k] = m->body_ipos[3 * base_body_ + k];
      }
    }
    pendulum_joint_ = mj_name2id(m, mjOBJ_JOINT, "pendulum_ball_joint");

    rim_dofs_.clear();
    roller_dofs_.clear();
    contact_geoms_.clear();
    std::vector<bool> wheel_body(m->nbody, false);
    const auto add_dof = [m](const std::string& joint, std::vector<int>* dofs) {
      const int id = mj_name2id(m, mjOBJ_JOINT, joint.c_str());
      if (id >= 0) {
        dofs->push_back(m->jnt_dofadr[id]);
      }
    };
    const auto mark_body = [m, &wheel_body](const std::string& body) {
      const int id = mj_name2id(m, mjOBJ_BODY, body.c_str());
      if (id >= 0) {
        wheel_body[id] = true;
      }
    };
    for (const char* wheel : {"left", "right", "back"}) {
      const std::string rim = std::string("rim_") + wheel;
      add_dof(rim + "_joint", &rim_dofs_);
      mark_body(rim + "_link");
      for (int i = 0; i < kRollersPerWheel; ++i) {
        const std::string roller = "roller_" + std::to_string(i) + "_" + rim;
        add_dof(roller + "_joint", &roller_dofs_);
        mark_body(roller + "_link");
      }
    }
    for (int g = 0; g < m->ngeom; ++g) {
      const bool collides = m->geom_contype[g] != 0 || m->geom_conaffinity[g] != 0;
      if (collides && wheel_body[m->geom_bodyid[g]]) {
        contact_geoms_.push_back(g);
      }
    }
    const int ground = mj_name2id(m, mjOBJ_GEOM, "ground");
    if (ground >= 0) {
      contact_geoms_.push_back(ground);
    }
  }

  const mjModel* model_ = nullptr;  // compared only; a reload allocates before freeing the old
  mjData* scratch_ = nullptr;
  int base_body_ = -1;
  mjtNum nominal_mass_ = 0.0;
  mjtNum nominal_inertia_[3] = {0.0, 0.0, 0.0};
  mjtNum nominal_ipos_[3] = {0.0, 0.0, 0.0};
  int pendulum_joint_ = -1;
  std::vector<int> rim_dofs_;
  std::vector<int> roller_dofs_;
  std::vector<int> contact_geoms_;
  bool inertial_dirty_ = true;
  mjtNum applied_mass_scale_ = 0.0;
  mjtNum applied_com_offset_cm_[3] = {0.0, 0.0, 0.0};
};

ModelRandomizer g_model_randomizer;

class OmniwheelRosBridge : public rclcpp::Node {
 public:
  explicit OmniwheelRosBridge(mj::Simulate* sim)
      : Node("omniwheel_mujoco_bridge"), sim_(sim) {
    using namespace std::chrono_literals;

    wrench_sub_ = create_subscription<geometry_msgs::msg::Wrench>(
        "/omni/cmd_wrench", rclcpp::QoS(10),
        [this](const geometry_msgs::msg::Wrench::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(command_mutex_);
          command_wrench_ = {msg->force.x, msg->force.y, msg->torque.z};
          ++command_seq_;
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

  // Actuation in the order training applies it (BaseWrenchWheelTorqueAction): the received
  // wrench in policy action units gets an offset and Gaussian jitter and is clipped, a new
  // command is occasionally dropped (the previous one is held), and the result is delayed. Then
  // wheel torques get the per-wheel strength and offset, first-order motor lag and the
  // torque-speed envelope. Runs on the physics thread with Simulate::mtx held.
  void applyControls(mjModel* model, mjData* data) {
    if (!initialized_) {
      initialize(model);
    }
    syncTime(data->time);
    const DomainParams& p = g_params;

    std::array<double, 3> wrench{};
    std::uint64_t seq = 0;
    bool timed_out = false;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      wrench = command_wrench_;
      seq = command_seq_;
      timed_out = !command_received_ ||
                  std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                last_command_time_)
                          .count() > g_config.command_timeout_sec;
    }
    if (timed_out) {
      // No controller: motors off, and no noise or offsets that would push the robot around.
      resetActuation();
      for (std::size_t i = 0; i < 3; ++i) {
        applied_torque_nm_[i] = 0.0;
        data->ctrl[actuator_ids_[i]] = 0.0;
      }
      return;
    }

    if (seq != last_command_seq_) {
      last_command_seq_ = seq;
      std::array<double, 3> packet = {wrench[0] / kActionForceScaleN,
                                      wrench[1] / kActionForceScaleN,
                                      wrench[2] / kActionYawTorqueScaleNm};
      for (std::size_t i = 0; i < 3; ++i) {
        packet[i] = clamp(packet[i], -1.0, 1.0) + p.action_offset[i] +
                    p.action_jitter_std * gaussian_(physics_rng_);
        packet[i] = clamp(packet[i], -1.0, 1.0);
      }
      if (unit_(physics_rng_) < p.action_hold_pct / 100.0) {
        packet = held_packet_;
      }
      held_packet_ = packet;
      packet_queue_.emplace_back(data->time, packet);
    }

    // Newest command that is at least the delay old; zero until the first one is.
    const double cutoff = data->time - p.action_delay_ms / 1000.0;
    while (packet_queue_.size() >= 2 && packet_queue_[1].first <= cutoff) {
      packet_queue_.pop_front();
    }
    std::array<double, 3> action{};
    if (!packet_queue_.empty() && packet_queue_.front().first <= cutoff) {
      action = packet_queue_.front().second;
    }

    const double limit = g_config.torque_limit_nm;
    const auto target = baseWrenchToMotorTorquesLeftRightBack(
        action[0] * kActionForceScaleN, action[1] * kActionForceScaleN,
        action[2] * kActionYawTorqueScaleNm);
    const double tau = p.motor_lag_ms / 1000.0;
    const double alpha = tau > 0.0 ? 1.0 - std::exp(-model->opt.timestep / tau) : 1.0;
    for (std::size_t i = 0; i < 3; ++i) {
      double torque = clamp(target[i], -limit, limit);
      torque = clamp(torque * p.torque_strength[i] + p.torque_offset_nm[i], -limit, limit);
      lagged_torque_nm_[i] += alpha * (torque - lagged_torque_nm_[i]);
      torque = lagged_torque_nm_[i];
      if (g_config.torque_speed_limit) {
        const double wheel_vel = data->qvel[model->jnt_dofadr[wheel_joint_ids_[i]]];
        torque = limitByTorqueSpeedEnvelope(torque, wheel_vel);
      }
      applied_torque_nm_[i] = clamp(torque, -limit, limit);
      data->ctrl[actuator_ids_[i]] = applied_torque_nm_[i];
    }
  }

  // Samples the true sensor values after every physics step, for the sensor delays.
  void recordSensors() {
    if (!initialized_ || !g_model || !g_data) {
      return;
    }
    syncTime(g_data->time);
    sensor_history_.push_back(trueSensorSample());
    while (!sensor_history_.empty() &&
           sensor_history_.front().time < g_data->time - kSensorHistorySec) {
      sensor_history_.pop_front();
    }
  }

  // Back-EMF headroom of the motor's current loop: only motoring torque (same sign
  // as wheel velocity) is limited; braking passes through to the plain clamp.
  static double limitByTorqueSpeedEnvelope(double torque, double wheel_vel) {
    if (torque * wheel_vel <= 0.0) {
      return torque;
    }
    const double stall = g_config.motor_stall_torque_nm;
    const double available =
        std::max(0.0, stall * g_config.supply_voltage_v / g_config.motor_nominal_voltage_v -
                          stall * std::abs(wheel_vel) / g_config.motor_no_load_speed_rad_s);
    return clamp(torque, -available, available);
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
    const SensedState sensed = senseState();

    const int base_qpos = g_model->jnt_qposadr[base_freejoint_id_];
    const double x = g_data->qpos[base_qpos + 0];
    const double y = g_data->qpos[base_qpos + 1];
    const double yaw = yawFromMujocoQuat(&g_data->qpos[base_qpos + 3]);
    const auto odom_quat = yawQuaternion(yaw);

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
    odom_msg.twist.twist.linear.x = sensed.base_vel[0];
    odom_msg.twist.twist.linear.y = sensed.base_vel[1];
    odom_msg.twist.twist.angular.z = sensed.base_vel[2];
    setDiagonalCovariance(odom_msg.pose.covariance, 1e-4, 1e-4, 1.0, 1.0, 1.0, 1e-4);
    setDiagonalCovariance(odom_msg.twist.covariance, 1e-3, 1e-3, 1.0, 1.0, 1.0, 1e-3);
    odom_pub_->publish(odom_msg);

    publishWheelOdometry(stamp);
    publishJointStates(stamp, sensed);
    publishEncoderAngles(sensed);
  }

  struct SensorSample {
    double time{0.0};
    std::array<double, 2> pendulum_pos{};
    std::array<double, 2> pendulum_vel{};
    std::array<double, 3> base_vel{};  // vx, vy in the yaw-aligned base frame, wz
  };

  struct SensedState {
    std::array<double, 2> pendulum_pos{};
    std::array<double, 2> pendulum_vel{};
    std::array<double, 3> base_vel{};
  };

  SensorSample trueSensorSample() const {
    SensorSample sample;
    sample.time = g_data->time;
    sample.pendulum_pos = pendulumAngles();
    sample.pendulum_vel = pendulumVelocities();

    const int base_qpos = g_model->jnt_qposadr[base_freejoint_id_];
    const int base_qvel = g_model->jnt_dofadr[base_freejoint_id_];
    const double yaw = yawFromMujocoQuat(&g_data->qpos[base_qpos + 3]);
    const double vx_world = g_data->qvel[base_qvel + 0];
    const double vy_world = g_data->qvel[base_qvel + 1];
    sample.base_vel = {std::cos(yaw) * vx_world + std::sin(yaw) * vy_world,
                       -std::sin(yaw) * vx_world + std::cos(yaw) * vy_world,
                       g_data->qvel[base_qvel + 5]};
    return sample;
  }

  // Newest recorded sample at least `delay_s` old; the current state if none is.
  SensorSample delayedSample(double delay_s, const SensorSample& now) const {
    const double cutoff = now.time - delay_s;
    for (auto it = sensor_history_.rbegin(); it != sensor_history_.rend(); ++it) {
      if (it->time <= cutoff) {
        return *it;
      }
    }
    return sensor_history_.empty() ? now : sensor_history_.front();
  }

  // The policy's sensor inputs as training corrupts them (observation terms in ft_t7's
  // env.yaml): a fixed offset, a delay, an occasional repeated sample, then uniform noise. The
  // pendulum angle and rate share one delay and hold, as in training. Noise and hold are drawn
  // per published message here rather than per 60 Hz policy step.
  SensedState senseState() {
    syncTime(g_data->time);
    const DomainParams& p = g_params;
    const SensorSample now = trueSensorSample();

    const SensorSample pend = delayedSample(p.pendulum_delay_ms / 1000.0, now);
    std::array<double, 4> pend_value = {
        pend.pendulum_pos[0] + p.pendulum_angle_offset_deg[0] * kDegToRad,
        pend.pendulum_pos[1] + p.pendulum_angle_offset_deg[1] * kDegToRad,
        pend.pendulum_vel[0] + p.pendulum_rate_offset_deg_s[0] * kDegToRad,
        pend.pendulum_vel[1] + p.pendulum_rate_offset_deg_s[1] * kDegToRad};
    if (have_held_pendulum_ && unit_(sensor_rng_) < p.pendulum_hold_pct / 100.0) {
      pend_value = held_pendulum_;
    }
    held_pendulum_ = pend_value;
    have_held_pendulum_ = true;

    const SensorSample base = delayedSample(p.base_vel_delay_ms / 1000.0, now);
    std::array<double, 3> base_value{};
    for (std::size_t i = 0; i < 3; ++i) {
      base_value[i] = base.base_vel[i] + p.base_vel_offset[i];
    }
    if (have_held_base_vel_ && unit_(sensor_rng_) < p.base_vel_hold_pct / 100.0) {
      base_value = held_base_vel_;
    }
    held_base_vel_ = base_value;
    have_held_base_vel_ = true;

    const auto noise = [this](double amplitude) {
      return amplitude * (2.0 * unit_(sensor_rng_) - 1.0);
    };
    SensedState sensed;
    for (std::size_t i = 0; i < 2; ++i) {
      sensed.pendulum_pos[i] = pend_value[i] + noise(p.pendulum_angle_noise_deg * kDegToRad);
      sensed.pendulum_vel[i] =
          pend_value[2 + i] + noise(p.pendulum_rate_noise_deg_s * kDegToRad);
    }
    for (std::size_t i = 0; i < 3; ++i) {
      sensed.base_vel[i] = base_value[i] + noise(p.base_vel_noise);
    }
    return sensed;
  }

  // Sim time going backwards means a reset or reload: drop every delayed or held value.
  void syncTime(double time) {
    if (time < last_sim_time_) {
      resetActuation();
      sensor_history_.clear();
      have_held_pendulum_ = false;
      have_held_base_vel_ = false;
    }
    last_sim_time_ = time;
  }

  void resetActuation() {
    packet_queue_.clear();
    held_packet_ = {0.0, 0.0, 0.0};
    lagged_torque_nm_ = {0.0, 0.0, 0.0};
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

  void publishJointStates(const builtin_interfaces::msg::Time& stamp, const SensedState& sensed) {
    const auto wheel_pos_lrb = wheelPositionsLeftRightBack();
    const auto wheel_vel_lrb = wheelVelocitiesLeftRightBack();
    const auto& pendulum_pos = sensed.pendulum_pos;
    const auto& pendulum_vel = sensed.pendulum_vel;

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

  void publishEncoderAngles(const SensedState& sensed) {
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

    const auto& pendulum_pos = sensed.pendulum_pos;
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

  static constexpr double kSensorHistorySec = 0.25;  // longer than the longest sensor delay

  std::mutex command_mutex_;
  std::array<double, 3> command_wrench_{0.0, 0.0, 0.0};  // fx, fy, tz as received
  std::uint64_t command_seq_{0};
  std::chrono::steady_clock::time_point last_command_time_{std::chrono::steady_clock::now()};
  bool command_received_{false};

  // Physics-thread actuation state.
  std::uint64_t last_command_seq_{0};
  std::array<double, 3> held_packet_{0.0, 0.0, 0.0};
  std::deque<std::pair<double, std::array<double, 3>>> packet_queue_;  // (sim time, action)
  std::array<double, 3> lagged_torque_nm_{0.0, 0.0, 0.0};
  std::array<double, 3> applied_torque_nm_{0.0, 0.0, 0.0};
  std::mt19937 physics_rng_{std::random_device{}()};

  // Sensor state, shared by the physics thread (recording) and the ROS thread (publishing)
  // under Simulate::mtx; sensor_rng_ is used only on the ROS thread.
  double last_sim_time_{0.0};
  std::deque<SensorSample> sensor_history_;
  std::array<double, 4> held_pendulum_{};
  bool have_held_pendulum_{false};
  std::array<double, 3> held_base_vel_{};
  bool have_held_base_vel_{false};
  std::mt19937 sensor_rng_{std::random_device{}()};

  std::normal_distribution<double> gaussian_{0.0, 1.0};
  std::uniform_real_distribution<double> unit_{0.0, 1.0};

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
// Simulate's own key handler, which owns the default shortcuts (space pauses/resumes).
GLFWkeyfun g_simulate_key_callback = nullptr;

void stepPhysics(mjModel* model, mjData* data) {
  if (g_bridge) {
    g_bridge->enforceFixedPendulum(model, data, false);
    g_bridge->applyControls(model, data);
  }
  mj_step(model, data);
  if (g_bridge) {
    g_bridge->enforceFixedPendulum(model, data, true);
    g_bridge->recordSensors();
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
      g_model_randomizer.apply(g_model, g_params);

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

void userKeyCallback(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (key == GLFW_KEY_BACKSPACE) {
    if (act == GLFW_PRESS && g_model && g_data) {
      mj_resetData(g_model, g_data);
      if (g_bridge) {
        g_bridge->enforceFixedPendulum(g_model, g_data, false);
      }
      mj_forward(g_model, g_data);
    }
    return;
  }
  if (g_simulate_key_callback) {
    g_simulate_key_callback(window, key, scancode, act, mods);
  }
}

// Adds the domain-randomization sliders to the viewer's right panel, bound straight to
// g_params: Pendulum (the two most-used knobs), Robot physics, Motors and Sensors. Simulate
// rebuilds the right panel on every model load and trims it back to its own sections when it
// remakes the joint or control section, so the sections are re-appended whenever they go
// missing. PollEvents runs on the render thread with sim->mtx held and the GL context current,
// which the mjui calls require.
class DomainUiAdapter : public mj::GlfwAdapter {
 public:
  void attach(mj::Simulate* sim) { sim_ = sim; }

  void PollEvents() override {
    mj::GlfwAdapter::PollEvents();
    if (sim_ && sim_->m_ && !hasSections()) {
      addSections();
    }
  }

 private:
  static constexpr int kNumSections = 4;
  static constexpr const char* kLastSectionName = "Sensors";

  static mjuiDef section(const char* name) {
    mjuiDef def{};
    def.type = mjITEM_SECTION;
    mju::strcpy_arr(def.name, name);
    def.state = mjPRESERVE;
    return def;
  }

  static mjuiDef separator(const char* name) {
    mjuiDef def{};
    def.type = mjITEM_SEPARATOR;
    mju::strcpy_arr(def.name, name);
    def.state = 1;  // plain, not collapsible
    return def;
  }

  // `range` is "min max divisions"; state 2 enables the slider once a model is loaded.
  static mjuiDef slider(const char* name, mjtNum* value, const char* range) {
    mjuiDef def{};
    def.type = mjITEM_SLIDERNUM;
    mju::strcpy_arr(def.name, name);
    def.state = 2;
    def.pdata = value;
    mju::strcpy_arr(def.other, range);
    return def;
  }

  bool hasSections() const {
    const mjUI& ui = sim_->ui1;
    return ui.nsect > 0 && std::strcmp(ui.sect[ui.nsect - 1].name, kLastSectionName) == 0;
  }

  void addSections() {
    mjUI& ui = sim_->ui1;
    if (ui.nsect + kNumSections > mjMAXUISECT) {
      return;
    }
    // Pendulum starts open, the rest collapsed; afterwards mjPRESERVE keeps the user's choice.
    if (!added_once_) {
      for (int i = 0; i < kNumSections; ++i) {
        ui.sect[ui.nsect + i].state = i == 0 ? mjSECT_OPEN : mjSECT_CLOSED;
      }
      added_once_ = true;
    }

    DomainParams& p = g_params;
    mjuiDef end{};
    end.type = mjITEM_END;
    const std::vector<mjuiDef> defs = {
        section("Pendulum"),
        slider("Max angle (deg)", &p.pendulum_max_angle_deg, "0 10 20"),
        slider("Damping", &p.pendulum_damping, "0 0.1 20"),

        section("Robot physics"),
        separator("Base"),
        slider("Mass scale", &p.mass_scale, "0.85 1.15 30"),
        slider("COM x (cm)", &p.com_offset_cm[0], "-2 2 40"),
        slider("COM y (cm)", &p.com_offset_cm[1], "-2 2 40"),
        slider("COM z (cm)", &p.com_offset_cm[2], "-1 3 40"),
        separator("Joint damping"),
        slider("Rims", &p.rim_damping, "0 0.05 20"),
        slider("Rollers", &p.roller_damping, "0 0.001 20"),
        separator("Joint friction"),
        slider("Pendulum", &p.pendulum_friction, "0 0.02 20"),
        slider("Rims", &p.rim_friction, "0 0.02 20"),
        slider("Rollers", &p.roller_friction, "0 0.02 20"),
        separator("Ground contact"),
        slider("Friction", &p.ground_friction, "0.4 2.0 32"),

        section("Motors"),
        separator("Torque strength"),
        slider("Left", &p.torque_strength[0], "0.8 1.2 40"),
        slider("Right", &p.torque_strength[1], "0.8 1.2 40"),
        slider("Back", &p.torque_strength[2], "0.8 1.2 40"),
        separator("Torque offset (Nm)"),
        slider("Left", &p.torque_offset_nm[0], "-0.2 0.2 40"),
        slider("Right", &p.torque_offset_nm[1], "-0.2 0.2 40"),
        slider("Back", &p.torque_offset_nm[2], "-0.2 0.2 40"),
        separator("Timing (ms)"),
        slider("Motor lag", &p.motor_lag_ms, "0 20 20"),
        slider("Command delay", &p.action_delay_ms, "0 33.3333 20"),
        separator("Command noise (action units)"),
        slider("Offset Fx", &p.action_offset[0], "-0.03 0.03 30"),
        slider("Offset Fy", &p.action_offset[1], "-0.03 0.03 30"),
        slider("Offset Tz", &p.action_offset[2], "-0.03 0.03 30"),
        slider("Jitter std", &p.action_jitter_std, "0 0.04 20"),
        slider("Hold (%)", &p.action_hold_pct, "0 2 20"),

        section(kLastSectionName),
        separator("Pendulum angle (deg)"),
        slider("Noise", &p.pendulum_angle_noise_deg, "0 0.7 14"),
        slider("Offset 1", &p.pendulum_angle_offset_deg[0], "-0.25 0.25 20"),
        slider("Offset 2", &p.pendulum_angle_offset_deg[1], "-0.25 0.25 20"),
        separator("Pendulum rate (deg/s)"),
        slider("Noise", &p.pendulum_rate_noise_deg_s, "0 6 12"),
        slider("Offset 1", &p.pendulum_rate_offset_deg_s[0], "-1 1 20"),
        slider("Offset 2", &p.pendulum_rate_offset_deg_s[1], "-1 1 20"),
        separator("Pendulum timing"),
        slider("Delay (ms)", &p.pendulum_delay_ms, "0 50 30"),
        slider("Hold (%)", &p.pendulum_hold_pct, "0 2 20"),
        separator("Base velocity (m/s, rad/s)"),
        slider("Noise", &p.base_vel_noise, "0 0.06 12"),
        slider("Offset vx", &p.base_vel_offset[0], "-0.01 0.01 20"),
        slider("Offset vy", &p.base_vel_offset[1], "-0.01 0.01 20"),
        slider("Offset wz", &p.base_vel_offset[2], "-0.01 0.01 20"),
        slider("Delay (ms)", &p.base_vel_delay_ms, "0 33.3333 20"),
        slider("Hold (%)", &p.base_vel_hold_pct, "0 1 20"),
        end,
    };
    mjui_add(&ui, defs.data());

    // Same steps as simulate's UiModify, which is not exported.
    mjrContext& con = mjr_context();
    mjui_resize(&ui, &con);
    mjr_addAux(ui.auxid, ui.width, ui.maxheight, ui.spacing.samples, &con);
    mjui_update(-1, -1, &ui, &sim_->uistate, &con);
  }

  mj::Simulate* sim_ = nullptr;
  bool added_once_ = false;
};

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

  auto ui_adapter = std::make_unique<DomainUiAdapter>();
  DomainUiAdapter* domain_ui = ui_adapter.get();
  auto sim = std::make_unique<mj::Simulate>(std::move(ui_adapter), &cam, &opt, &pert, false);
  domain_ui->attach(sim.get());
  sim->run = 0;  // start paused; press space to run

  g_bridge = std::make_shared<OmniwheelRosBridge>(sim.get());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(g_bridge);
  std::thread ros_thread([&executor]() { executor.spin(); });

  std::thread physics_thread(&PhysicsThread, sim.get(), g_config.scene_file.c_str());

  g_simulate_key_callback = glfwSetKeyCallback(
      static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_, userKeyCallback);
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
