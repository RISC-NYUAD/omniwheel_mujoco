#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <onnxruntime_cxx_api.h>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace {

constexpr std::size_t kObservationSize = 13;
constexpr std::size_t kActionSize = 3;
constexpr std::chrono::milliseconds kControlPeriod{20};
constexpr float kMaxLinearVelocityMps = 1.0F;
constexpr float kMaxYawVelocityRadS = 2.0F;
constexpr double kPi = 3.14159265358979323846;

constexpr char kInputName[] = "obs";
constexpr char kOutputName[] = "actions";
constexpr char kJointStatesTopic[] = "/joint_states";
constexpr char kOdometryTopic[] = "/odometry/filtered";
constexpr char kCommandTopic[] = "/nav_vel";
constexpr char kPendulumJoint1Name[] = "Pendulum_Joint_1";
constexpr char kPendulumJoint2Name[] = "Pendulum_Joint_2";

double wrapToPi(double angle) {
  double wrapped = std::fmod(angle + kPi, 2.0 * kPi);
  if (wrapped < 0.0) {
    wrapped += 2.0 * kPi;
  }
  return wrapped - kPi;
}

std::string shapeToString(const std::vector<int64_t>& shape) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0U) {
      stream << ',';
    }
    stream << shape[i];
  }
  stream << ']';
  return stream.str();
}

class OmniwheelPolicyNode : public rclcpp::Node {
 public:
  OmniwheelPolicyNode()
      : Node("omniwheel_policy_node"),
        ort_environment_(ORT_LOGGING_LEVEL_WARNING, "omniwheel_policy"),
        cpu_memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    model_file_ = declare_parameter<std::string>("model_file", "");
    if (model_file_.empty()) {
      throw std::invalid_argument(
          "parameter 'model_file' must point to the exported policy.onnx");
    }
    std::ifstream model_stream(model_file_, std::ios::binary);
    if (!model_stream.good()) {
      throw std::runtime_error("cannot read ONNX model: " + model_file_);
    }

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ =
        std::make_unique<Ort::Session>(ort_environment_, model_file_.c_str(), session_options_);
    validateModelContract();

    command_publisher_ =
        create_publisher<geometry_msgs::msg::Twist>(kCommandTopic, rclcpp::QoS(10));
    joint_states_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
        kJointStatesTopic, rclcpp::QoS(10),
        std::bind(&OmniwheelPolicyNode::onJointStates, this, std::placeholders::_1));
    odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
        kOdometryTopic, rclcpp::QoS(10),
        std::bind(&OmniwheelPolicyNode::onOdometry, this, std::placeholders::_1));

    // This timer is attached to the node's ROS clock. With use_sim_time=true it therefore
    // advances from /clock, rather than from wall time.
    control_timer_ = rclcpp::create_timer(
        get_node_base_interface(), get_node_timers_interface(), get_clock(), kControlPeriod,
        std::bind(&OmniwheelPolicyNode::onControlTimer, this));

    RCLCPP_INFO(get_logger(),
                "Loaded %s; waiting for %s and %s before publishing policy commands at 50 Hz",
                model_file_.c_str(), kJointStatesTopic, kOdometryTopic);
  }

 private:
  struct RobotState {
    std::array<double, 2> pendulum_position{};
    std::array<double, 2> pendulum_velocity{};
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
    double vx_body{0.0};
    double vy_body{0.0};
    double yaw_rate{0.0};
    double target_x{0.0};
    double target_y{0.0};
    double target_yaw{0.0};
    bool joint_states_received{false};
    bool odometry_received{false};
    bool target_initialized{false};
  };

  void validateModelContract() {
    if (session_->GetInputCount() != 1U || session_->GetOutputCount() != 1U) {
      throw std::runtime_error("policy must have exactly one input and one output");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session_->GetInputNameAllocated(0, allocator);
    const auto output_name = session_->GetOutputNameAllocated(0, allocator);
    if (input_name.get() == nullptr || std::string(input_name.get()) != kInputName) {
      throw std::runtime_error("policy input must be named 'obs'");
    }
    if (output_name.get() == nullptr || std::string(output_name.get()) != kOutputName) {
      throw std::runtime_error("policy output must be named 'actions'");
    }

    // TensorTypeAndShapeInfo is a non-owning view here, so retain each parent TypeInfo
    // until all metadata has been consumed.
    const auto input_type_info = session_->GetInputTypeInfo(0);
    const auto output_type_info = session_->GetOutputTypeInfo(0);
    const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
    const auto input_shape = input_info.GetShape();
    const auto output_shape = output_info.GetShape();
    const std::vector<int64_t> expected_input_shape{1, static_cast<int64_t>(kObservationSize)};
    const std::vector<int64_t> expected_output_shape{1, static_cast<int64_t>(kActionSize)};

    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_shape != expected_input_shape) {
      throw std::runtime_error("policy input 'obs' must be float32[1,13], got shape " +
                               shapeToString(input_shape));
    }
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_shape != expected_output_shape) {
      throw std::runtime_error("policy output 'actions' must be float32[1,3], got shape " +
                               shapeToString(output_shape));
    }
  }

  void onJointStates(const sensor_msgs::msg::JointState::SharedPtr message) {
    const auto joint1 = std::find(message->name.begin(), message->name.end(), kPendulumJoint1Name);
    const auto joint2 = std::find(message->name.begin(), message->name.end(), kPendulumJoint2Name);
    if (joint1 == message->name.end() || joint2 == message->name.end()) {
      state_.joint_states_received = false;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Waiting for joints '%s' and '%s' on %s", kPendulumJoint1Name,
                           kPendulumJoint2Name, kJointStatesTopic);
      return;
    }

    const std::size_t index1 =
        static_cast<std::size_t>(std::distance(message->name.begin(), joint1));
    const std::size_t index2 =
        static_cast<std::size_t>(std::distance(message->name.begin(), joint2));
    if (index1 >= message->position.size() || index2 >= message->position.size() ||
        index1 >= message->velocity.size() || index2 >= message->velocity.size()) {
      state_.joint_states_received = false;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "%s is missing pendulum position or velocity values",
                           kJointStatesTopic);
      return;
    }

    // The MuJoCo joint-state convention is the negative of the analytical alpha/beta
    // convention used to train this policy.
    const std::array<double, 2> position{-message->position[index1],
                                         -message->position[index2]};
    const std::array<double, 2> velocity{-message->velocity[index1],
                                         -message->velocity[index2]};
    if (!std::isfinite(position[0]) || !std::isfinite(position[1]) ||
        !std::isfinite(velocity[0]) || !std::isfinite(velocity[1])) {
      state_.joint_states_received = false;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Ignoring non-finite pendulum state on %s", kJointStatesTopic);
      return;
    }

    state_.pendulum_position = position;
    state_.pendulum_velocity = velocity;
    state_.joint_states_received = true;
  }

  void onOdometry(const nav_msgs::msg::Odometry::SharedPtr message) {
    const auto& pose = message->pose.pose;
    const auto& twist = message->twist.twist;
    const auto& quaternion = pose.orientation;
    const double quaternion_norm_squared =
        quaternion.x * quaternion.x + quaternion.y * quaternion.y +
        quaternion.z * quaternion.z + quaternion.w * quaternion.w;

    if (!std::isfinite(pose.position.x) || !std::isfinite(pose.position.y) ||
        !std::isfinite(twist.linear.x) || !std::isfinite(twist.linear.y) ||
        !std::isfinite(twist.angular.z) || !std::isfinite(quaternion_norm_squared) ||
        quaternion_norm_squared <= std::numeric_limits<double>::epsilon()) {
      state_.odometry_received = false;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Ignoring invalid state on %s", kOdometryTopic);
      return;
    }

    const double inverse_norm = 1.0 / std::sqrt(quaternion_norm_squared);
    const double qx = quaternion.x * inverse_norm;
    const double qy = quaternion.y * inverse_norm;
    const double qz = quaternion.z * inverse_norm;
    const double qw = quaternion.w * inverse_norm;
    const double yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                  1.0 - 2.0 * (qy * qy + qz * qz));
    if (!std::isfinite(yaw)) {
      state_.odometry_received = false;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Ignoring invalid orientation on %s", kOdometryTopic);
      return;
    }

    state_.x = pose.position.x;
    state_.y = pose.position.y;
    state_.yaw = yaw;
    // The simulator publishes this odometry twist in base_footprint, matching training.
    state_.vx_body = twist.linear.x;
    state_.vy_body = twist.linear.y;
    state_.yaw_rate = twist.angular.z;
    state_.odometry_received = true;

    // This test runner intentionally holds the pose at which the first valid odometry
    // sample arrives. The target is never updated after initialization.
    if (!state_.target_initialized) {
      state_.target_x = state_.x;
      state_.target_y = state_.y;
      state_.target_yaw = state_.yaw;
      state_.target_initialized = true;
      RCLCPP_INFO(get_logger(), "Fixed policy target initialized at x=%.3f, y=%.3f, yaw=%.3f",
                  state_.target_x, state_.target_y, state_.target_yaw);
    }
  }

  std::array<float, kObservationSize> buildObservation() const {
    const double error_x_world = state_.target_x - state_.x;
    const double error_y_world = state_.target_y - state_.y;
    const double cosine_yaw = std::cos(state_.yaw);
    const double sine_yaw = std::sin(state_.yaw);
    const double error_x_body =
        cosine_yaw * error_x_world + sine_yaw * error_y_world;
    const double error_y_body =
        -sine_yaw * error_x_world + cosine_yaw * error_y_world;

    const std::array<double, kObservationSize> observation_double{
        state_.pendulum_position[0],
        state_.pendulum_position[1],
        state_.pendulum_velocity[0],
        state_.pendulum_velocity[1],
        error_x_body,
        error_y_body,
        wrapToPi(state_.target_yaw - state_.yaw),
        state_.vx_body,
        state_.vy_body,
        state_.yaw_rate,
        previous_action_[0],
        previous_action_[1],
        previous_action_[2],
    };

    std::array<float, kObservationSize> observation{};
    for (std::size_t i = 0; i < observation.size(); ++i) {
      observation[i] = static_cast<float>(observation_double[i]);
      if (!std::isfinite(observation[i])) {
        throw std::runtime_error("policy observation contains a non-finite value");
      }
    }
    return observation;
  }

  std::array<float, kActionSize> runPolicy(
      std::array<float, kObservationSize>& observation) {
    const std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(kObservationSize)};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        cpu_memory_info_, observation.data(), observation.size(), input_shape.data(),
        input_shape.size());
    const std::array<const char*, 1> input_names{kInputName};
    const std::array<const char*, 1> output_names{kOutputName};
    Ort::RunOptions run_options{nullptr};
    auto output_tensors = session_->Run(run_options, input_names.data(), &input_tensor, 1,
                                        output_names.data(), 1);
    if (output_tensors.size() != 1U || !output_tensors[0].IsTensor()) {
      throw std::runtime_error("policy did not return its actions tensor");
    }

    const auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    if (output_info.GetElementCount() != kActionSize) {
      throw std::runtime_error("policy returned an action tensor with the wrong size");
    }
    const float* output = output_tensors[0].GetTensorData<float>();
    std::array<float, kActionSize> action{};
    for (std::size_t i = 0; i < action.size(); ++i) {
      if (!std::isfinite(output[i])) {
        throw std::runtime_error("policy returned a non-finite action");
      }
      action[i] = std::clamp(output[i], -1.0F, 1.0F);
    }
    return action;
  }

  void publishAction(const std::array<float, kActionSize>& action) {
    geometry_msgs::msg::Twist command;
    command.linear.x = static_cast<double>(action[0] * kMaxLinearVelocityMps);
    command.linear.y = static_cast<double>(action[1] * kMaxLinearVelocityMps);
    command.angular.z = static_cast<double>(action[2] * kMaxYawVelocityRadS);
    command_publisher_->publish(command);
    previous_action_ = action;
  }

  void publishSafeStop() {
    publishAction({0.0F, 0.0F, 0.0F});
  }

  void onControlTimer() {
    if (!state_.joint_states_received || !state_.odometry_received ||
        !state_.target_initialized) {
      publishSafeStop();
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Policy is waiting for valid joint states and odometry");
      return;
    }

    try {
      auto observation = buildObservation();
      publishAction(runPolicy(observation));
    } catch (const Ort::Exception& error) {
      publishSafeStop();
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "ONNX policy inference failed; commanding zero velocity: %s",
                            error.what());
    } catch (const std::exception& error) {
      publishSafeStop();
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "Policy update failed; commanding zero velocity: %s", error.what());
    }
  }

  std::string model_file_;
  Ort::Env ort_environment_;
  Ort::SessionOptions session_options_;
  Ort::MemoryInfo cpu_memory_info_;
  std::unique_ptr<Ort::Session> session_;

  RobotState state_;
  std::array<float, kActionSize> previous_action_{};

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr command_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

}  // namespace

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  int exit_code = EXIT_SUCCESS;
  try {
    rclcpp::spin(std::make_shared<OmniwheelPolicyNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("omniwheel_policy_node"), "%s", error.what());
    exit_code = EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return exit_code;
}
