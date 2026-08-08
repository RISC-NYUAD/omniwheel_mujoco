cmake_minimum_required(VERSION 3.8)
project(omniwheel_mujoco_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(onnxruntime_vendor REQUIRED)

set(MUJOCO_ROOT /workspace/omniwheel_mujoco/dependencies/mujoco)

find_library(MUJOCO_LIBRARY mujoco REQUIRED
  PATHS
    /usr/local/lib
    ${MUJOCO_ROOT}/build/lib
)

file(GLOB MUJOCO_SIMULATE_SOURCES
  ${MUJOCO_ROOT}/simulate/glfw_*.cc
  ${MUJOCO_ROOT}/simulate/platform_*.cc
  ${MUJOCO_ROOT}/simulate/simulate.cc
)

add_executable(omniwheel_mujoco_sim
  ${MUJOCO_SIMULATE_SOURCES}
  src/lodepng/lodepng.cpp
  src/main.cpp
)

target_compile_features(omniwheel_mujoco_sim PUBLIC cxx_std_17)

target_include_directories(omniwheel_mujoco_sim PUBLIC
  ${MUJOCO_ROOT}/include
  ${MUJOCO_ROOT}/simulate
  src/lodepng
)

target_link_libraries(omniwheel_mujoco_sim
  ${MUJOCO_LIBRARY}
  glfw
  pthread
  yaml-cpp
)

ament_target_dependencies(omniwheel_mujoco_sim
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  rosgraph_msgs
  tf2_ros
)

add_executable(omniwheel_policy_node
  src/policy_node.cpp
)

target_compile_features(omniwheel_policy_node PUBLIC cxx_std_17)

target_link_libraries(omniwheel_policy_node
  onnxruntime::onnxruntime
)

ament_target_dependencies(omniwheel_policy_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
)

install(TARGETS
  omniwheel_mujoco_sim
  omniwheel_policy_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch urdf models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
