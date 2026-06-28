#include <pinocchio/fwd.hpp> // forward declarations must be included first.
#include <algorithm>
#include <array>
#include <cmath>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include "std_srvs/SetBool.h"
#include "humanoid_controllers/humanoidController.h"
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <std_srvs/Trigger.h>
#include <angles/angles.h>
#include <humanoid_estimation/FromTopiceEstimate.h>
#include <humanoid_estimation/LinearKalmanFilter.h>
#ifdef KUAVO_CONTROL_LIB_FOUND
#include <kuavo_estimation/base_filter/InEkfBaseFilter.h>
#endif
#include <humanoid_wbc/WeightedWbc.h>
#include <humanoid_wbc/StandUpWbc.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <humanoid_wbc/HierarchicalWbc.h>
#include "humanoid_interface_drake/common/utils.h"
#include "humanoid_interface_drake/common/sensor_data.h"
#include "humanoid_interface_drake/humanoid_interface_drake.h"
#include <fstream>
#include <sstream>

namespace humanoid_controller
{
  using namespace ocs2;
  using namespace humanoid;
  using Duration = std::chrono::duration<double>;
  using Clock = std::chrono::high_resolution_clock;
  std::mutex head_mtx;

  namespace
  {
  constexpr int kRlActionRampSteps = 25;      // 0.5s @ 50Hz — pure RL only; avoid clip-saturated first step
  constexpr double kRlActionRampMax = 1.0;    // |action| cap at end of ramp

  vector_t drakeQToMujocoQpos(const vector_t &drake_q)
  {
    vector_t mujoco_q(drake_q.size());
    mujoco_q << drake_q.segment(4, 3), drake_q.head(4), drake_q.tail(drake_q.size() - 7);
    return mujoco_q;
  }
  // Pure RL handoff: WBC torque crossfade + short action ramp; PD base stays training defaultJointPos.
  constexpr double kRlWbcHoldSec = 0.0;
  constexpr double kRlTorqueCrossfadeSec = 0.4;
  constexpr int kRlSpawnSettleSteps = 2500;   // 2.5s @ 1kHz — WBC hold spawn pose before inference
  constexpr int kRlSpawnUprightFrames = 500; // 0.5s consecutive upright before RL on
  // scene_rl default qpos: pelvis upright (no forward pitch in quat), legs = MuJoCo passive stand
  constexpr std::array<double, 19> kMujocoS49DefaultQpos = {
      -0.00505, 0.00000, 0.84414, 1.00000, 0.00000, 0.00000, 0.00000, -0.01825, -0.00190, -0.52421,
      0.73860,  -0.31872, 0.01835,  0.01825,  0.00190,  -0.52421, 0.73860,  -0.31872, -0.01835};

  vector_t buildMujocoTrainingStandQpos(const vector_t &drake_template,
                                        const vector_t &initial_state,
                                        const vector_t &default_joints)
  {
    vector_t mujoco_q = drakeQToMujocoQpos(drake_template);
    const double z = initial_state(8);
    const double pitch = initial_state(10);
    const double half = 0.5 * pitch;
    mujoco_q(0) = 0.0;
    mujoco_q(1) = 0.0;
    mujoco_q(2) = z;
    mujoco_q(3) = std::cos(half);
    mujoco_q(4) = 0.0;
    mujoco_q(5) = std::sin(half);
    mujoco_q(6) = 0.0;
    for (int i = 7; i < mujoco_q.size(); ++i)
    {
      mujoco_q(i) = 0.0;
    }
    const int n = std::min(static_cast<int>(default_joints.size()), static_cast<int>(mujoco_q.size()) - 7);
    for (int i = 0; i < n; ++i)
    {
      mujoco_q(7 + i) = default_joints(i);
    }
    return mujoco_q;
  }

  void initSimEquilibriumJoints(Eigen::VectorXd &eq_joints)
  {
    eq_joints.setZero();
    constexpr std::array<double, 12> k_legs = {
        -0.01825, -0.00190, -0.52421, 0.73860, -0.31872, 0.01835,
        0.01825,  0.00190,  -0.52421, 0.73860, -0.31872, -0.01835};
    const int n = std::min(static_cast<int>(eq_joints.size()), 12);
    for (int i = 0; i < n; ++i)
    {
      eq_joints[i] = k_legs[static_cast<size_t>(i)];
    }
  }

  Eigen::Vector3d gravityBodyFromSensor(const SensorData &sensor_data)
  {
    // Raw buffer entries only have quat_; quat_offset_ is set in applySensorData().
    Eigen::Quaternion<scalar_t> q = sensor_data.quat_offset_;
    if (q.norm() < 0.5)
    {
      q = sensor_data.quat_;
    }
    q.normalize();
    const Eigen::Matrix3d R = q.matrix();
    return R.transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);
  }

  bool isGravityUpright(const SensorData &sensor_data)
  {
    const Eigen::Vector3d gravity_body = gravityBodyFromSensor(sensor_data);
    return gravity_body.z() < -0.92 && std::abs(gravity_body.x()) < 0.12 && std::abs(gravity_body.y()) < 0.30;
  }
  bool loadDanceReferenceCsv(const std::string &csv_path,
                             const Eigen::VectorXd &default_joint_pos,
                             std::vector<Eigen::VectorXd> &trajectory_gym)
  {
    std::ifstream file(csv_path);
    if (!file.is_open())
    {
      std::cerr << "[humanoidController] Cannot open dance CSV: " << csv_path << std::endl;
      return false;
    }
    trajectory_gym.clear();
    std::string line;
    while (std::getline(file, line))
    {
      if (line.empty())
      {
        continue;
      }
      if (line.rfind("time", 0) == 0)
      {
        continue;
      }
      std::vector<double> vals;
      std::stringstream ss(line);
      std::string token;
      while (std::getline(ss, token, ','))
      {
        if (!token.empty())
        {
          vals.push_back(std::stod(token));
        }
      }
      if (vals.size() < 26)
      {
        continue;
      }
      // CSV columns are gym order (leg_l1..l6, leg_r1..r6, zarm_*), same as
      // punch_env_cfg / rewards.py _parse_csv_joint_rows — not Isaac lab order.
      std::vector<double> row;
      if (vals.size() >= 27)
      {
        row.assign(vals.begin() + 1, vals.begin() + 27);
      }
      else
      {
        row.assign(vals.begin(), vals.begin() + 26);
      }
      Eigen::VectorXd frame(default_joint_pos.size());
      frame.setZero();
      const int joint_count = std::min<int>(26, default_joint_pos.size());
      for (int gym_idx = 0; gym_idx < joint_count; ++gym_idx)
      {
        frame[gym_idx] = row[gym_idx] - default_joint_pos[gym_idx];
      }
      trajectory_gym.push_back(frame);
    }
    if (trajectory_gym.empty())
    {
      std::cerr << "[humanoidController] Dance CSV has no valid rows: " << csv_path << std::endl;
      return false;
    }
    std::cout << "[humanoidController] Loaded dance CSV frames: " << trajectory_gym.size()
              << " from " << csv_path << std::endl;
    return true;
  }
  } // namespace

  static void mujocoSimStart(ros::NodeHandle &nh_)
  {
    std_srvs::SetBool srv;
    srv.request.data = true;
    // 等待服务可用
    bool service_available = ros::service::waitForService("sim_start", ros::Duration(5.0)); // 5秒超时
    if (service_available)
    {
      ros::ServiceClient sim_start_client = nh_.serviceClient<std_srvs::SetBool>("sim_start");
      if (sim_start_client.call(srv))
      {
        if (srv.response.success)
        {
          ROS_INFO("sim_start Service call succeeded with message: %s", srv.response.message.c_str());
          return;
        }
        else
        {
          ROS_ERROR("sim_start Service call failed");
        }
      }
      else
      {
        ROS_ERROR("Failed to call sim_start service");
      }
    }
    else
    {
      ROS_ERROR("sim_start Service not available");
    }
    exit(1);
  }
  void humanoidController::keyboard_thread_func()
  {
    usleep(100000);
    struct sched_param param;
    param.sched_priority = 0;
    auto result = pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);
    if (result != 0)
    {
      std::cerr << "Failed to set keyboard_thread_func's scheduling parameters. Error: " << strerror(result) << std::endl;
    }
    ros::NodeHandle nh;
    char Walk_Command = '\0';
    while (ros::ok())
    {
      CommandData CommandData_;
      CommandData_ = getCommandData();
      if (hardware_status_ != 1)
      {
        usleep(100000);
        continue;
      }
      if (kbhit())
      {
        Walk_Command = getchar();
        if (Walk_Command == '\n' || Walk_Command == '\0')
        {
          continue;
        }
        // 首先，将光标移动到上一行的开始
        std::cout << "\033[F";
        // 然后，清除当前行（从光标位置到行尾）
        std::cout << "\033[K";
        std::cout << "[keyboard command]: " << Walk_Command << std::endl;
        if (Walk_Command == 'x')
        {
          std::cout << "x" << std::endl;
          for (int i = 0; i < 5; i++)
          {
            std::cout << "publish stop message" << std::endl;
            std_msgs::Bool stop_msg;
            stop_msg.data = true;
            stop_pub.publish(stop_msg);
            ros::Duration(0.1).sleep();
          }
        }
        // else if (Walk_Command == 'f')
        // {
        //   wbc_only_ = !wbc_only_;
        //   std::cout << "start using mpc: " << !wbc_only_ << std::endl;
        // }
        // else if (Walk_Command == 'v')
        // {
        //   std::cout << "v" << std::endl;
        //   control_mode_ = (control_mode_ == 1) ? 2 : 1;
        //   std::cout << "control mode:" << control_mode_ << std::endl;
        // }
        // else if (Walk_Command == 't')
        // {
        //   control_mode_ = (control_mode_ == 0) ? 2 : 0;
        //   std::cout << "control mode:" << control_mode_ << std::endl;
        // }
        else if (Walk_Command == 'r' || Walk_Command == 'l')
        {
          activateRlController();
        }
        else if (Walk_Command == 'w')
        {
          CommandData_.cmdStance_ = 0;
          CommandData_.cmdVelLineX_ += 0.1;
          ROS_INFO_STREAM("Velocity linear x: " << CommandData_.cmdVelLineX_);
        }
        else if (Walk_Command == 's')
        {
          CommandData_.cmdStance_ = 0;
          CommandData_.cmdVelLineX_ -= 0.1;
          ROS_INFO_STREAM("Velocity linear x: " << CommandData_.cmdVelLineX_);
        }
        else if (Walk_Command == 'a')
        {
          CommandData_.cmdStance_ = 0;
          CommandData_.cmdVelLineY_ += 0.1;
          ROS_INFO_STREAM("Velocity linear y: " << CommandData_.cmdVelLineY_);
        }
        else if (Walk_Command == 'd')
        {
          CommandData_.cmdStance_ = 0;
          CommandData_.cmdVelLineY_ -= 0.1;
          ROS_INFO_STREAM("Velocity linear y: " << CommandData_.cmdVelLineY_);
        }
        else if (Walk_Command == 'q')
        {
          CommandData_.cmdStance_ = 0;
          CommandData_.cmdVelAngularZ_ += 0.1;
          ROS_INFO_STREAM("Velocity angular z: " << CommandData_.cmdVelAngularZ_);
        }
        else if (Walk_Command == 'e')
        {
          CommandData_.cmdStance_ = 0;
          CommandData_.cmdVelAngularZ_ -= 0.1;
          ROS_INFO_STREAM("Velocity angular z: " << CommandData_.cmdVelAngularZ_);
        }
        else if (Walk_Command == ' ')
        {
          CommandData_.setzero();
          CommandData_.cmdStance_ = initialCommandData_.cmdStance_;
          ROS_INFO_STREAM("Velocity linear x: " << CommandData_.cmdVelLineX_);
          ROS_INFO_STREAM("Velocity linear y: " << CommandData_.cmdVelLineY_);
          ROS_INFO_STREAM("Velocity angular z: " << CommandData_.cmdVelAngularZ_);
        }
        else if (Walk_Command == 'm')
        {
          CommandData_.setzero();
          ROS_INFO_STREAM("Velocity linear x: " << CommandData_.cmdVelLineX_);
          ROS_INFO_STREAM("Velocity linear y: " << CommandData_.cmdVelLineY_);
          ROS_INFO_STREAM("Velocity angular z: " << CommandData_.cmdVelAngularZ_);
        }
        setCommandData(CommandData_);
        Walk_Command = '\0';
      }
      usleep(50000);
    }
  }
  void humanoidController::printRLparam()
  {
    std::cout << "[RL param]:Inital Command: " << initialCommandData_.getCommand().transpose() << std::endl;
    std::cout << "[RL param]:Start Using RL: " << is_rl_controller_ << std::endl;
    std::cout << "[RL param]:Joint Control Mode(0:CST, 1:CSV, 2:CSP):" << JointControlMode_.transpose() << std::endl;
    std::cout << "[RL param]:Joint PD mode:" << JointPDMode_.transpose() << std::endl;
    std::cout << "[RL param]:Joint Kp:" << jointKp_.transpose() << std::endl;
    std::cout << "[RL param]:Joint Kd:" << jointKd_.transpose() << std::endl;
    std::cout << "[RL param]:Initial State:" << initialState_.transpose() << std::endl;
    std::cout << "[RL param]:Torque Limits:" << torqueLimits_.transpose() << std::endl;
    std::cout << "[RL param]:Action Scale Test:" << actionScaleTest_.transpose() << std::endl;
    std::cout << "=============================================================================" << std::endl;
  }
  void humanoidController::captureSimObsOffsetFromState(const vector_t &rbd_state)
  {
    const int n = static_cast<int>(jointNum_ + jointArmNum_);
    sim_joint_obs_offset_.resize(n);
    for (int i = 0; i < n; ++i)
    {
      sim_joint_obs_offset_[i] = rbd_state(6 + i) - defalutJointPos_[i];
    }
    sim_handoff_obs_offset_ = !is_real_;
  }
  void humanoidController::beginPureRlTorqueCrossfade()
  {
    if (is_real_ || sim_keep_wbc_with_rl_ || kRlTorqueCrossfadeSec <= 0.0)
    {
      rl_torque_crossfade_active_ = false;
      return;
    }
    if (wbc_tau_at_rl_takeover_.cwiseAbs().maxCoeff() < 1.0)
    {
      rl_torque_crossfade_active_ = false;
      return;
    }
    rl_torque_crossfade_active_ = true;
    rl_torque_crossfade_start_time_ = ros::Time::now();
  }
  void humanoidController::finalizeRlWbcHandoff()
  {
    const vector_t rbd = getRobotState();
    captureSimObsOffsetFromState(rbd);
    episodeLength_ = 0;
    rl_takeover_step_ = 0;
    rl_obs_reset_pending_ = true;
    rl_action_ramp_logged_ = false;
    jointCmdFilter_.reset();
    {
      std::lock_guard<std::mutex> lock(action_mtx_);
      actions_.setZero();
      rl_takeover_seed_actions_.setZero();
    }
    const SensorData sd = getRobotSensorData();
    const Eigen::Vector3d g = gravityBodyFromSensor(sd);
    beginPureRlTorqueCrossfade();
    ROS_INFO("[RL mode] WBC hold done — RL PD (defaultJointPos), crossfade %.0fms: gravity_body=[%.3f,%.3f,%.3f], |obs jointPos|=%.4f",
             kRlTorqueCrossfadeSec * 1000.0, g.x(), g.y(), g.z(), sim_joint_obs_offset_.cwiseAbs().maxCoeff());
  }
  void humanoidController::activateRlController()
  {
    CommandData cmd = getCommandData();
    cmd.setzero();
    setCommandData(cmd);
    is_rl_controller_ = true;
    episodeLength_ = 0;
    rl_takeover_step_ = 0;
    rl_torque_blend_step_ = 0;
    rl_obs_reset_pending_ = true;
    wbc_tau_at_rl_takeover_ = last_wbc_tau_;
    rl_wbc_torque_handoff_ = (!is_real_ && !sim_keep_wbc_with_rl_ && kRlWbcHoldSec > 0.0);
    rl_torque_crossfade_active_ = false;
    if (!rl_wbc_torque_handoff_)
    {
      beginPureRlTorqueCrossfade();
    }
    if (rl_wbc_torque_handoff_)
    {
      rl_wbc_handoff_end_time_ = ros::Time::now() + ros::Duration(kRlWbcHoldSec);
    }
    const SensorData sd = getRobotSensorData();
    const Eigen::Vector3d g = gravityBodyFromSensor(sd);
    const vector_t rbd_state = getRobotState();
    const int n = static_cast<int>(jointNum_ + jointArmNum_);
    double handoff_joint_err = 0.0;
    if (!is_real_ && !sim_keep_wbc_with_rl_)
    {
      captureSimObsOffsetFromState(rbd_state);
      handoff_joint_err = sim_joint_obs_offset_.cwiseAbs().maxCoeff();
    }
    {
      std::lock_guard<std::mutex> lock(action_mtx_);
      actions_.setZero();
      rl_takeover_seed_actions_.setZero();
    }
    rl_action_ramp_logged_ = false;
    jointCmdFilter_.reset();
    if (!is_real_ && sim_keep_wbc_with_rl_)
    {
      sim_wbc_ref_filtered_.resize(n);
      for (int i = 0; i < n; ++i)
      {
        sim_wbc_ref_filtered_[i] = initial_status_(12 + i);
      }
      ROS_INFO("[RL mode] RL activated: WBC tracks RL targets (blend=%.2f, filter=%.0fms) — full-speed policy, no ramp",
               sim_wbc_rl_action_blend_, sim_wbc_ref_filter_tau_ * 1000.0);
    }
    else if (rl_wbc_torque_handoff_)
    {
      ROS_INFO("[RL mode] RL activated (WBC->R): gravity_body=[%.3f,%.3f,%.3f], |obs jointPos|=%.4f — inference NOW, WBC %.0f ms then RL PD at handoff pose",
               g.x(), g.y(), g.z(), handoff_joint_err, kRlWbcHoldSec * 1000.0);
    }
    else
    {
      ROS_INFO("[RL mode] RL activated (WBC->R): gravity_body=[%.3f,%.3f,%.3f], |obs jointPos|=%.4f — RL PD + WBC crossfade %.0fms, action ramp %d steps",
               g.x(), g.y(), g.z(), handoff_joint_err, kRlTorqueCrossfadeSec * 1000.0, kRlActionRampSteps);
    }
    printRLparam();
  }
  void humanoidController::beginRlSpawnSettle()
  {
    CommandData cmd = getCommandData();
    cmd.setzero();
    setCommandData(cmd);
    is_rl_controller_ = false;
    episodeLength_ = 0;
    rl_spawn_settling_ = true;
    rl_spawn_settle_step_ = 0;
    rl_spawn_upright_count_ = 0;
    {
      std::lock_guard<std::mutex> lock(action_mtx_);
      actions_.setZero();
      rl_takeover_seed_actions_.setZero();
    }
    ROS_INFO("[RL mode] rl_from_start: settling %.1fs — WBC holds spawn pose (not RL PD), inference paused",
             static_cast<double>(kRlSpawnSettleSteps) / 1000.0);
  }
  void humanoidController::activateRlControllerFromStart(bool blend_from_wbc)
  {
    CommandData cmd = getCommandData();
    cmd.setzero();
    setCommandData(cmd);
    is_rl_controller_ = true;
    episodeLength_ = 0;
    rl_takeover_step_ = 0;
    rl_obs_reset_pending_ = true;
    if (blend_from_wbc)
    {
      wbc_tau_at_rl_takeover_ = last_wbc_tau_;
      rl_wbc_torque_handoff_ = (!is_real_ && !sim_keep_wbc_with_rl_ && kRlWbcHoldSec > 0.0);
      rl_torque_crossfade_active_ = false;
      if (rl_wbc_torque_handoff_)
      {
        rl_wbc_handoff_end_time_ = ros::Time::now() + ros::Duration(kRlWbcHoldSec);
      }
    }
    else
    {
      rl_wbc_torque_handoff_ = false;
      rl_torque_crossfade_active_ = false;
      wbc_tau_at_rl_takeover_.setZero();
    }
    const vector_t rbd_state = getRobotState();
    const int n = static_cast<int>(jointNum_ + jointArmNum_);
    if (!is_real_ && !sim_keep_wbc_with_rl_ && !blend_from_wbc)
    {
      sim_joint_obs_offset_.setZero(n);
      sim_handoff_obs_offset_ = true;
    }
    {
      std::lock_guard<std::mutex> lock(action_mtx_);
      actions_.setZero();
      rl_takeover_seed_actions_.setZero();
    }
    ROS_INFO("[RL mode] RL from start: inference on after settle, full policy, no ramp (blend_from_wbc=%d)",
             static_cast<int>(blend_from_wbc));
    printRLparam();
  }
  bool humanoidController::init(HybridJointInterface *robot_hw, ros::NodeHandle &controller_nh, bool is_nodelet_node)
  {
    int robot_version_int;
    RobotVersion robot_version(3, 4);
    controllerNh_ = controller_nh;
    ros_logger_ = new TopicLogger(controllerNh_);
    if (controllerNh_.hasParam("/robot_version"))
    {
      controllerNh_.getParam("/robot_version", robot_version_int);
      int major = robot_version_int / 10;
      int minor = robot_version_int % 10;
      robot_version = RobotVersion(major, minor);
    }
    is_nodelet_node_ = is_nodelet_node;
    drake_interface_ = HighlyDynamic::HumanoidInterfaceDrake::getInstancePtr(robot_version, true, 2e-3);
    kuavo_settings_ = drake_interface_->getKuavoSettings();

    auto &motor_info = kuavo_settings_.hardware_settings;
    headNum_ = motor_info.num_head_joints;
    armNumReal_ = motor_info.num_arm_joints;
    jointNumReal_ = motor_info.num_joints - headNum_ - armNumReal_;
    actuatedDofNumReal_ = jointNumReal_ + armNumReal_ + headNum_;
    ros::param::set("/armRealDof", static_cast<int>(armNumReal_));
    ros::param::set("/legRealDof", static_cast<int>(jointNumReal_));
    ros::param::set("/headRealDof", static_cast<int>(headNum_));
    output_tau_ = vector_t::Zero(actuatedDofNumReal_);
    output_pos_ = vector_t::Zero(actuatedDofNumReal_);
    output_vel_ = vector_t::Zero(actuatedDofNumReal_);

    auto robot_config = drake_interface_->getRobotConfig();
    ruiwo_motor_velocities_factor_ = robot_config->getValue<double>("motor_velocities_factor");
    AnkleSolverType ankleSolverType = static_cast<AnkleSolverType>(robot_config->getValue<int>("ankle_solver_type"));
    ankleSolver.getconfig(ankleSolverType);

    // Initialize OCS2
    inferenceFrequency_ = 100.0;      // 默认100Hz
    double controlFrequency_ = 500.0; // 默认500Hz
    std::string urdfFile;
    std::string taskFile;
    std::string referenceFile;
    std::string gaitCommandFile;
    std::string rlParamFile;
    controllerNh_.getParam("/network_model_file", networkModelPath_);
    controllerNh_.getParam("/rl_param", rlParamFile);
    controllerNh_.getParam("/urdfFile", urdfFile);
    controllerNh_.getParam("/taskFile", taskFile);
    controllerNh_.getParam("/referenceFile", referenceFile);
    controllerNh_.getParam("/gaitCommandFile", gaitCommandFile);
    controllerNh_.getParam("/use_external_mpc", use_external_mpc_);
    controllerNh_.getParam("/wbc_frequency", controlFrequency_);
    dt_ = 1.0 / controlFrequency_;
    if (controllerNh_.hasParam("/real"))
    {
      controllerNh_.getParam("/real", is_real_);
      controllerNh_.getParam("/cali", is_cali_);
      if (is_real_)
      {
        std::cout << "real robot controller" << std::endl;
        ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(controllerNh_);
        hardware_interface_ptr_ = new KuavoHardwareInterface(nh_ptr, jointNum_);
      }
    }
    if (controllerNh_.hasParam("wbc_only"))
    {
      controllerNh_.getParam("/wbc_only", wbc_only_);
    }
    if (controllerNh_.hasParam("play_back"))
    {
      controllerNh_.getParam("/play_back", is_play_back_mode_);
    }
    if (controllerNh_.hasParam("/rl_from_start"))
    {
      controllerNh_.getParam("/rl_from_start", rl_from_start_);
    }
    if (controllerNh_.hasParam("/sim_keep_wbc_with_rl"))
    {
      controllerNh_.getParam("/sim_keep_wbc_with_rl", sim_keep_wbc_with_rl_);
    }
    if (controllerNh_.hasParam("/sim_rl_debug_zero_action"))
    {
      controllerNh_.getParam("/sim_rl_debug_zero_action", sim_rl_debug_zero_action_);
    }
    if (sim_rl_debug_zero_action_ && !is_real_)
    {
      ROS_WARN("[RL debug] sim_rl_debug_zero_action=true: policy output forced to 0 (PD hold at defaultJointPos only)");
    }
    if (controllerNh_.hasParam("/sim_wbc_rl_action_blend"))
    {
      controllerNh_.getParam("/sim_wbc_rl_action_blend", sim_wbc_rl_action_blend_);
    }
    if (controllerNh_.hasParam("/sim_wbc_ref_filter_tau"))
    {
      controllerNh_.getParam("/sim_wbc_ref_filter_tau", sim_wbc_ref_filter_tau_);
    }
    if (controllerNh_.hasParam("/sim_wbc_rl_blend_start"))
    {
      controllerNh_.getParam("/sim_wbc_rl_blend_start", sim_wbc_rl_blend_start_);
    }
    if (controllerNh_.hasParam("/sim_wbc_rl_blend_ramp_sec"))
    {
      controllerNh_.getParam("/sim_wbc_rl_blend_ramp_sec", sim_wbc_rl_blend_ramp_sec_);
    }
    if (controllerNh_.hasParam("/sim_wbc_max_rl_joint_offset_rad"))
    {
      controllerNh_.getParam("/sim_wbc_max_rl_joint_offset_rad", sim_wbc_max_rl_joint_offset_rad_);
    }
    if (controllerNh_.hasParam("channel_map_path"))
    {
      std::string channel_map_path;
      controllerNh_.getParam("channel_map_path", channel_map_path);
      ROS_INFO_STREAM("Loading joystick mapping from " << channel_map_path);
      loadJoyJsonConfig(channel_map_path);
    }
    else
    {
      ROS_WARN_STREAM("No channel_map_path parameter found, using default joystick mapping.");
    }
    if (controllerNh_.hasParam("joystick_sensitivity"))
    {
      controllerNh_.getParam("joystick_sensitivity", joystickSensitivity);
      ROS_INFO_STREAM("Loading joystick sensitivity: " << joystickSensitivity);
    }
    else
    {
      ROS_WARN_STREAM("No input sensitivity parameter found, using default joystick sensitivity.");
    }
    Eigen::Vector4d joystickFilterCutoffFreq_(joystickSensitivity, joystickSensitivity,
                                              joystickSensitivity, joystickSensitivity);
    joystickFilter_.setParams(0.01, joystickFilterCutoffFreq_);
    oldJoyMsg_.axes = std::vector<float>(8, 0.0); // 假设有 8 个轴，默认值为 0.0
    oldJoyMsg_.buttons = std::vector<int32_t>(12, 0);
    size_t buffer_size = (is_play_back_mode_) ? 20 : 5;
    sensors_data_buffer_ptr_ = new KuavoDataBuffer<SensorData>("humanoid_sensors_data_buffer", buffer_size, dt_);

    bool verbose = false;
    loadData::loadCppDataType(taskFile, "humanoid_interface.verbose", verbose);
    setupHumanoidInterface(taskFile, urdfFile, referenceFile, gaitCommandFile, verbose, robot_version_int);
    ros::NodeHandle nh;
    CentroidalModelPinocchioMapping pinocchioMapping(HumanoidInterface_->getCentroidalModelInfo());
    std::cout << "HumanoidInterface_->getCentroidalModelInfo().robotMass:" << HumanoidInterface_->getCentroidalModelInfo().robotMass << std::endl;
    eeKinematicsPtr_ = std::make_shared<PinocchioEndEffectorKinematics>(HumanoidInterface_->getPinocchioInterface(), pinocchioMapping,
                                                                        HumanoidInterface_->modelSettings().contactNames3DoF);

    pinocchioInterface_ptr_ = new PinocchioInterface(HumanoidInterface_->getPinocchioInterface());
    eeKinematicsPtr_->setPinocchioInterface(*pinocchioInterface_ptr_);
    auto &info = HumanoidInterface_->getCentroidalModelInfo();
    centroidalModelInfoWBC_ = info;
    centroidalModelInfo_ = info; // TODO:适配旧的代码
    jointArmNum_ = info.actuatedDofNum - jointNum_;
    jointTorqueCmd_.resize(jointNum_ + jointArmNum_);
    jointTorqueCmd_.setZero();
    initialState_.resize(12 + jointNum_ + jointArmNum_);
    defalutJointPos_.resize(jointNum_ + jointArmNum_);
    JointControlMode_.resize(jointNum_ + jointArmNum_);
    JointControlMode_.setZero();
    JointPDMode_.resize(jointNum_ + jointArmNum_);
    JointPDMode_.setZero();
    jointKp_.resize(jointNum_ + jointArmNum_);
    jointKd_.resize(jointNum_ + jointArmNum_);
    torqueLimits_.resize(jointNum_ + jointArmNum_);
    actionScaleTest_.resize(jointNum_ + jointArmNum_);
    jointCmdFilterState_.resize(jointNum_ + jointArmNum_);
    sensor_data_head_.resize_joint(headNum_);
    head_kp_.resize(headNum_);
    head_kd_.resize(headNum_);
    output_tau_.resize(jointNum_ + jointArmNum_);
    output_tau_.setZero();
    jointPos_ = vector_t::Zero(jointNum_ + jointArmNum_);
    jointVel_ = vector_t::Zero(jointNum_ + jointArmNum_);
    jointAcc_ = vector_t::Zero(jointNum_ + jointArmNum_);
    quat_ = Eigen::Quaternion<scalar_t>(1, 0, 0, 0);
    loadSettings(rlParamFile, verbose, dt_);
    if (headNum_ > 0)
    {
      loadData::loadEigenMatrix(referenceFile, "head_kp_", head_kp_);
      loadData::loadEigenMatrix(referenceFile, "head_kd_", head_kd_);
    }
    auto squat_initial_state_ = drake_interface_->getSquatInitialState();
    std::vector<double> squat_initial_state_vector(squat_initial_state_.data(), squat_initial_state_.data() + squat_initial_state_.size());
    std::cout << "controller squat_initial_state_:" << squat_initial_state_.transpose() << std::endl;

    std::cout << "controller initialState: " << initialState_.transpose() << std::endl;
    std::vector<double> initial_state_vector(initialState_.data(), initialState_.data() + initialState_.size());
    nh.setParam("/initial_state", initial_state_vector);
    nh.setParam("/squat_initial_state", squat_initial_state_vector);

    // Publish before waitForReady: MuJoCo blocks on this param; sensors come only after physics starts.
    if (rl_from_start_ && !is_real_)
    {
      sim_joint_obs_offset_ = Eigen::VectorXd::Zero(defalutJointPos_.size());
      sim_handoff_obs_offset_ = true;
      ROS_INFO("[RL mode] rl_from_start spawn: obs offset zero (training defaultJointPos)");
      spawn_centroidal_status_ = initialState_;
      spawn_centroidal_status_.segment(12, jointNum_ + jointArmNum_) = defalutJointPos_;
      spawn_centroidal_status_[8] = kMujocoS49DefaultQpos[2];
      spawn_centroidal_status_[10] = 0.0;
      spawn_centroidal_status_[11] = 0.0;
    }
    {
      const vector_t drake_template = drake_interface_->getDrakeSquatState();
      vector_t mujoco_q;
      if (!is_real_)
      {
        mujoco_q = buildMujocoTrainingStandQpos(drake_template, initialState_, defalutJointPos_);
        isPreUpdateComplete = true;
        ROS_INFO("[sim] MuJoCo spawn at training default z=%.3f pitch=%.3f — skip squat stand-up",
                 initialState_(8), initialState_(10));
      }
      else
      {
        mujoco_q = drakeQToMujocoQpos(drake_template);
      }
      std::vector<double> mujoco_init_state(mujoco_q.data(), mujoco_q.data() + mujoco_q.size());
      ros::param::set("mujoco_init_state", mujoco_init_state);
    }

    sensorsDataSub_ = controllerNh_.subscribe<kuavo_msgs::sensorsData>("/sensors_data_raw", 10, &humanoidController::sensorsDataCallback, this);
    joy_sub_ = controllerNh_.subscribe<sensor_msgs::Joy>("/joy", 10, &humanoidController::joyCallback, this);
    jointCmdPub_ = controllerNh_.advertise<kuavo_msgs::jointCmd>("/joint_cmd", 10);
    targetTorquePub_ = controllerNh_.advertise<std_msgs::Float32MultiArray>("/targetTorque", 10);
    wbcFrequencyPub_ = controllerNh_.advertise<std_msgs::Float64>("/monitor/frequency/wbc", 10);
    wbcTimeCostPub_ = controllerNh_.advertise<std_msgs::Float64>("/monitor/time_cost/wbc", 10);
    stop_pub = controllerNh_.advertise<std_msgs::Bool>("/stop_robot", 10);

    // State estimation
    setupStateEstimate(taskFile, verbose);
    sensors_data_buffer_ptr_->waitForReady();
    wbc_ = std::make_shared<WeightedWbc>(HumanoidInterface_->getPinocchioInterface(), HumanoidInterface_->getCentroidalModelInfo(),
                                         *eeKinematicsPtr_);
    wbc_->setArmNums(jointArmNum_);
    wbc_->loadTasksSetting(taskFile, verbose, is_real_);

    standUpWbc_ = std::make_shared<StandUpWbc>(*pinocchioInterface_ptr_, info,
                                               *eeKinematicsPtr_);
    standUpWbc_->setArmNums(armNumReal_);
    standUpWbc_->loadTasksSetting(taskFile, verbose, is_real_);

    singleInputData.resize(numSingleObs_);
    networkInputData_.resize(numSingleObs_ * frameStack_);
    commandPhase_.resize(2);
    actions_.resize(jointNum_ + jointArmNum_);
    rl_takeover_seed_actions_.resize(jointNum_ + jointArmNum_);
    last_wbc_tau_.resize(jointNum_ + jointArmNum_);
    wbc_tau_at_rl_takeover_.resize(jointNum_ + jointArmNum_);
    singleInputData.setZero();
    networkInputData_.setZero();
    actions_.setZero();
    rl_takeover_seed_actions_.setZero();
    last_wbc_tau_.setZero();
    wbc_tau_at_rl_takeover_.setZero();
    for (int i = 0; i < frameStack_; i++)
    {
      input_deque.push_back(singleInputData);
    }
    compiled_model_ =
        core_.compile_model(networkModelPath_, "CPU"); // 创建编译模型

    keyboardThread_ = std::thread(&humanoidController::keyboard_thread_func, this);
    if (!keyboardThread_.joinable())
    {
      std::cerr << "Failed to start keyboard thread" << std::endl;
      exit(1);
    }

    return true;
  }
  



  void humanoidController::loadJoyJsonConfig(const std::string &config_file)
  {
    std::ifstream ifs(config_file);
    if (!ifs.is_open())
    {
      std::cerr << "Failed to open config file: " << config_file << std::endl;
      return;
    }
    nlohmann::json data_;
    ifs >> data_;
    auto updateMap = [](auto &map, const auto &items)
    {
      for (const auto &[key, value] : items)
      {
        std::cout << "button:" << key << " value:" << value << std::endl;
        map[key] = value;
      }
    };
    updateMap(joyButtonMap, data_["JoyButton"].items());
    updateMap(joyAxisMap, data_["JoyAxis"].items());
  }
  void humanoidController::loadSettings(const std::string &rlParamFile, bool verbose, double dt)
  {
    bool enable_ = false;
    double index_ = 0, startIdx_ = 0, mumIdx_ = 0, obsScale_ = 0;
    int num_ = 0;
    std::string networkModelFile_;
    Eigen::Vector3d accFilterCutoffFreq_, freeAccFilterCutoffFreq_, gyroFilterCutoffFreq_;
    Eigen::VectorXd defaultBaseState_(jointNum_), jointCmdFilterCutoffFreq_(jointNum_ + jointArmNum_);
    CommandData commandData_;
    boost::property_tree::ptree pt;
    boost::property_tree::read_info(rlParamFile, pt);
    // 使用新的 helper 函数来简化数据加载
    auto loadEigenMatrix = [&](const std::string &key, auto &matrix)
    {
      loadData::loadEigenMatrix(rlParamFile, key, matrix);
    };
    loadEigenMatrix("defaultJointState", defalutJointPos_);
    loadEigenMatrix("defaultBaseState", defaultBaseState_);
    loadEigenMatrix("JointControlMode", JointControlMode_);
    loadEigenMatrix("JointPDMode", JointPDMode_);
    loadEigenMatrix("jointKp", jointKp_);
    loadEigenMatrix("jointKd", jointKd_);
    loadEigenMatrix("torqueLimits", torqueLimits_);
    loadEigenMatrix("actionScaleTest", actionScaleTest_);
    loadEigenMatrix("accFilterCutoffFreq", accFilterCutoffFreq_);
    loadEigenMatrix("freeAccFilterCutoffFreq", freeAccFilterCutoffFreq_);
    loadEigenMatrix("gyroFilterCutoffFreq", gyroFilterCutoffFreq_);
    loadEigenMatrix("jointCmdFilterCutoffFreq", jointCmdFilterCutoffFreq_);
    loadEigenMatrix("jointCmdFilterState", jointCmdFilterState_);
    loadEigenMatrix("accFilterState", accFilterState_);
    loadEigenMatrix("freeAccFilterState", freeAccFilterState_);
    loadEigenMatrix("gyroFilterState", gyroFilterState_);
    loadEigenMatrix("velocityLimits", velocityLimits_);
    // loadEigenMatrix("initalCommand", initalCommand_);
    // loadEigenMatrix("commandScale", commandScale_);
    loadData::loadCppDataType(rlParamFile, "actionScale", actionScale_);
    loadData::loadCppDataType(rlParamFile, "frameStack", frameStack_);
    loadData::loadCppDataType(rlParamFile, "numSingleObs", numSingleObs_);
    loadData::loadCppDataType(rlParamFile, "cycleTime", cycleTime_);
    loadData::loadCppDataType(rlParamFile, "phase", phase_);
    loadData::loadCppDataType(rlParamFile, "episodeLength", episodeLength_);
    loadData::loadCppDataType(rlParamFile, "clipObservations", clipObservations_);
    loadData::loadCppDataType(rlParamFile, "clipActions", clipActions_);
    loadData::loadCppDataType(rlParamFile, "withArm", withArm_);
    loadData::loadCppDataType(rlParamFile, "inferenceFrequency", inferenceFrequency_);
    loadData::loadCppDataType(rlParamFile, "networkModelFile", networkModelFile_);
    // loadData::loadCppDataType(rlParamFile, "ankleSolverType", ankleSolverType_);

    loadData::loadCppDataType(rlParamFile, "danceCsvFile", danceCsvFile_);
    referenceJointPos_.resize(defalutJointPos_.size());
    referenceJointPos_.setZero();
    hasDanceReference_ = false;
    if (!danceCsvFile_.empty())
    {
      std::string csv_path = danceCsvFile_;
      if (csv_path.front() != '/')
      {
        const auto pos = rlParamFile.find_last_of("/\\");
        if (pos != std::string::npos)
        {
          csv_path = rlParamFile.substr(0, pos + 1) + danceCsvFile_;
        }
      }
      hasDanceReference_ = loadDanceReferenceCsv(csv_path, defalutJointPos_, danceReferenceTrajectoryGym_);
    }

    // ankleSolver.getconfig(ankleSolverType_);
    initialState_ << defaultBaseState_, defalutJointPos_;
    networkModelPath_ = networkModelPath_ + networkModelFile_;
    std::cout<<"networkModelPath_"<<networkModelPath_<<std::endl;
    accFilter_.setParams(dt, accFilterCutoffFreq_);
    freeAccFilter_.setParams(dt, freeAccFilterCutoffFreq_);
    gyroFilter_.setParams(dt, gyroFilterCutoffFreq_);
    jointCmdFilter_.setParams(dt, jointCmdFilterCutoffFreq_);

    // 加载命令数据
    const std::string prefixCommandData_ = "commandData";
    const std::vector<std::pair<std::string, double CommandData::*>> cmdInitalList = {
        {"cmdVelLineX", &CommandData::cmdVelLineX_},
        {"cmdVelLineY", &CommandData::cmdVelLineY_},
        {"cmdVelLineZ", &CommandData::cmdVelLineZ_},
        {"cmdVelAngularX", &CommandData::cmdVelAngularX_},
        {"cmdVelAngularY", &CommandData::cmdVelAngularY_},
        {"cmdVelAngularZ", &CommandData::cmdVelAngularZ_},
        {"cmdStance", &CommandData::cmdStance_},
    };
    const std::vector<std::pair<std::string, double CommandData::*>> cmdScaleList = {
        {"cmdVelLineX", &CommandData::cmdVelScaleLineX_},
        {"cmdVelLineY", &CommandData::cmdVelScaleLineY_},
        {"cmdVelLineZ", &CommandData::cmdVelScaleLineZ_},
        {"cmdVelAngularX", &CommandData::cmdVelScaleAngularX_},
        {"cmdVelAngularY", &CommandData::cmdVelScaleAngularY_},
        {"cmdVelAngularZ", &CommandData::cmdVelScaleAngularZ_},
        {"cmdStance", &CommandData::cmdScaleStance_}};
    for (const auto &[cmdName, cmdMember] : cmdInitalList)
    {
      loadData::loadPtreeValue(pt, commandData_.*cmdMember, prefixCommandData_ + ".inital." + cmdName, verbose);
    }
    for (const auto &[cmdName, cmdMember] : cmdScaleList)
    {
      loadData::loadPtreeValue(pt, commandData_.*cmdMember, prefixCommandData_ + ".scale." + cmdName, verbose);
    }
    initialCommandData_ = commandData_;
    setCommandData(initialCommandData_);
    // 加载单输入数据
    const std::string prefixSingleInputData_ = "singleInputData";
    if (verbose)
    {
      std::cerr << "\n #### singleInputData:";
      std::cerr << "\n #### =============================================================================\n";
    }
    for (const auto &pair : pt)
    {
      if (pair.first == prefixSingleInputData_)
      {
        for (const auto &pair2 : pair.second)
        {
          singleInputDataKeys.push_back(pair2.first);
          loadData::loadPtreeValue(pt, startIdx_, prefixSingleInputData_ + "." + pair2.first + ".startIdx", verbose);
          loadData::loadPtreeValue(pt, mumIdx_, prefixSingleInputData_ + "." + pair2.first + ".numIdx", verbose);
          loadData::loadPtreeValue(pt, obsScale_, prefixSingleInputData_ + "." + pair2.first + ".obsScales", verbose);
          num_ += mumIdx_;
          singleInputDataID_[pair2.first] = {startIdx_, mumIdx_, obsScale_};
        }
      }
    }
    if (num_ != numSingleObs_)
    {
      std::cerr << "Error: singleInputData number is not equal to 'numSingleObs_'" << std::endl;
      std::cerr << "Single Obs Number: " << num_ << std::endl;
      std::cerr << "'numSingleObs_' Number: " << numSingleObs_ << std::endl;
      exit(1);
    }
  }
  void humanoidController::headCmdCallback(const kuavo_msgs::robotHeadMotionData::ConstPtr &msg)
  {
    if (msg->joint_data.size() == 2)
    {
      if (msg->joint_data[0] < -30 || msg->joint_data[0] > 30 || msg->joint_data[1] < -25 || msg->joint_data[1] > 25)
      {
        ROS_WARN("Invalid robot head motion data. Joint data must be in the range [-30, 30] and [-25, 25].");
        return;
      }
      head_mtx.lock();
      desire_head_pos_[0] = msg->joint_data[0] * M_PI / 180.0;
      desire_head_pos_[1] = msg->joint_data[1] * M_PI / 180.0;
      head_mtx.unlock();
    }
    else
    {
      ROS_WARN("Invalid robot head motion data. Expected 2 elements, but received %lu elements.", msg->joint_data.size());
    }
  }

  void humanoidController::sensorsDataCallback(const kuavo_msgs::sensorsData::ConstPtr &msg)
  {
    auto &joint_data = msg->joint_data;
    auto &imu_data = msg->imu_data;
    auto &end_effector_data = msg->end_effector_data; // TODO: add end_effector_data to the observation
    SensorData sensor_data;
    sensor_data.resize_joint(jointNum_ + jointArmNum_);
    // JOINT DATA
    for (size_t i = 0; i < jointNum_ + jointArmNum_; ++i)
    {
      sensor_data.jointPos_(i) = joint_data.joint_q[i];
      sensor_data.jointVel_(i) = joint_data.joint_v[i];
      sensor_data.jointAcc_(i) = joint_data.joint_vd[i];
      sensor_data.jointCurrent_(i) = joint_data.joint_current[i];
    }
    ros::Time ros_time = msg->header.stamp;
    sensor_data.timeStamp_ = msg->sensor_time;
    double sensor_time_diff = (ros::Time::now() - ros_time).toSec() * 1000;
    ros_logger_->publishValue("/monitor/time_cost/sensor_to_controller", sensor_time_diff);
    // IMU
    sensor_data.quat_.coeffs().w() = imu_data.quat.w;
    sensor_data.quat_.coeffs().x() = imu_data.quat.x;
    sensor_data.quat_.coeffs().y() = imu_data.quat.y;
    sensor_data.quat_.coeffs().z() = imu_data.quat.z;
    sensor_data.angularVel_ << imu_data.gyro.x, imu_data.gyro.y, imu_data.gyro.z;
    sensor_data.linearAccel_ << imu_data.acc.x, imu_data.acc.y, imu_data.acc.z;
    sensor_data.freeLinearAccel_ << imu_data.free_acc.x, imu_data.free_acc.y, imu_data.free_acc.z;
    sensor_data.orientationCovariance_ << Eigen::Matrix<scalar_t, 3, 3>::Zero();
    sensor_data.angularVelCovariance_ << Eigen::Matrix<scalar_t, 3, 3>::Zero();
    sensor_data.linearAccelCovariance_ << Eigen::Matrix<scalar_t, 3, 3>::Zero();
    Eigen::Vector3d acc_filtered = accFilter_.update(sensor_data.linearAccel_);
    Eigen::Vector3d free_acc_filtered = freeAccFilter_.update(sensor_data.freeLinearAccel_);
    Eigen::Vector3d gyro_filtered = gyroFilter_.update(sensor_data.angularVel_);
    for (int i = 0; i < 3; i++)
    {
      sensor_data.linearAccel_(i) = accFilterState_(i) * acc_filtered(i) + (1 - accFilterState_(i)) * sensor_data.linearAccel_(i);
      sensor_data.freeLinearAccel_(i) = freeAccFilterState_(i) * free_acc_filtered(i) + (1 - freeAccFilterState_(i)) * sensor_data.freeLinearAccel_(i);
      sensor_data.angularVel_(i) = gyroFilterState_(i) * gyro_filtered(i) + (1 - gyroFilterState_(i)) * sensor_data.angularVel_(i);
    }
    ros_logger_->publishVector("/imu_data_filtered/linearAccel", acc_filtered);
    ros_logger_->publishVector("/imu_data_filtered/angularVel", gyro_filtered);
    ros_logger_->publishVector("/imu_data_filtered/freeLinearAccel", free_acc_filtered);
    sensors_data_buffer_ptr_->addData(sensor_data.timeStamp_.toSec(), sensor_data);
    if (headNum_ > 0 && joint_data.joint_q.size() == jointNumReal_ + armNumReal_ + headNum_)
    {
      int head_start_index = joint_data.joint_q.size() - headNum_;
      for (size_t i = 0; i < headNum_; ++i)
      {
        sensor_data_head_.jointPos_(i) = joint_data.joint_q[i + head_start_index];
        sensor_data_head_.jointVel_(i) = joint_data.joint_v[i + head_start_index];
        sensor_data_head_.jointAcc_(i) = joint_data.joint_vd[i + head_start_index];
        sensor_data_head_.jointCurrent_(i) = joint_data.joint_current[i + head_start_index];
      }
      if (!is_real_ && !head_desire_initialized_)
      {
        head_mtx.lock();
        desire_head_pos_ = sensor_data_head_.jointPos_;
        head_mtx.unlock();
        head_desire_initialized_ = true;
        ROS_INFO("[sim] Head hold: lock desire to current yaw=%.3f pitch=%.3f rad",
                 desire_head_pos_(0), desire_head_pos_(1));
      }
    }
    if (!is_initialized_)
      is_initialized_ = true;
  }
  void humanoidController::checkAndPublishCommandLine(const vector_t &joystick_origin_axis)
  {
    geometry_msgs::Twist cmdVel_;
    cmdVel_.linear.x = 0;
    cmdVel_.linear.y = 0;
    cmdVel_.linear.z = 0;
    cmdVel_.angular.x = 0;
    cmdVel_.angular.y = 0;
    cmdVel_.angular.z = 0;
    static bool send_zero_twist = false;
    CommandData CommandData_;
    CommandData_ = getCommandData();
    auto updated = commandLineToTargetTrajectories(joystick_origin_axis, cmdVel_);
    if (!Walkenable_)
    {
      CommandData_.setzero();
      CommandData_.cmdStance_ = initialCommandData_.cmdStance_;
      setCommandData(CommandData_);
      return;
    }
    if (!std::any_of(updated.begin(), updated.end(), [](bool x)
                     { return x; }))
    {
      if (!send_zero_twist)
      {
        std::cout << "[JoyControl] send zero twist" << std::endl;
        send_zero_twist = true;
        CommandData_.setzero();
      }

      CommandData_.cmdStance_ = cmdTrotgait_ || contactTrotgait_ ? 0 : initialCommandData_.cmdStance_;
      setCommandData(CommandData_);
      return;
    }
    send_zero_twist = false;
    cmdTrotgait_ = false;
    CommandData_.cmdVelLineX_ = cmdVel_.linear.x;
    CommandData_.cmdVelLineY_ = cmdVel_.linear.y;
    CommandData_.cmdVelLineZ_ = cmdVel_.linear.z;
    CommandData_.cmdVelAngularX_ = cmdVel_.angular.x;
    CommandData_.cmdVelAngularY_ = cmdVel_.angular.y;
    CommandData_.cmdVelAngularZ_ = cmdVel_.angular.z;
    CommandData_.cmdStance_ = 0;
    setCommandData(CommandData_);
  }
  std::vector<bool> humanoidController::commandLineToTargetTrajectories(const vector_t &joystick_origin_axis, geometry_msgs::Twist &cmdVel)
  {
    std::vector<bool> updated(6, false);
    Eigen::Vector4d limit_vector;
    limit_vector = velocityLimits_;
    // limit_vector(0) *= 0.5;/
    double dead_zone = 0.05;
    if (joystick_origin_axis.cwiseAbs().maxCoeff() < dead_zone)
      return updated; // command line is zero, do nothing
    // if (abs(joystick_origin_axis(0)) > 0.5)
    // {
    //   limit_vector(3) *= 0.5;
    //   if (joystick_origin_axis(0) > 0)
    //   {
    //     limit_vector(0) *= 2;
    //   }
    // }
    commadLineTarget_.head(4) = joystick_origin_axis.head(4).cwiseProduct(limit_vector);
    if (joystick_origin_axis.head(2).cwiseAbs().maxCoeff() > dead_zone)
    {
      cmdVel.linear.x = commadLineTarget_(0);
      cmdVel.linear.y = commadLineTarget_(1);
      updated[0] = true;
      updated[1] = true;
    }
    if (std::abs(joystick_origin_axis(2)) > dead_zone)
    {
      updated[2] = true;
      cmdVel.linear.z = commadLineTarget_(2);
    }
    else
    {
      cmdVel.linear.z = 0.0;
    }
    if (std::abs(commadLineTarget_(3)) > dead_zone)
    {
      updated[3] = true;
      cmdVel.angular.z = commadLineTarget_(3);
    }
    return updated;
  }

  void humanoidController::joyCallback(const sensor_msgs::Joy::ConstPtr &joy_msg)
  {
    // double alpha_ = joystickSensitivity / 1000;
    vector_t joystickOriginAxisFilter_ = vector_t::Zero(6);
    vector_t joystickOriginAxisTemp_ = vector_t::Zero(6);
    if (std::any_of(joy_msg->buttons.begin(), joy_msg->buttons.end(), [](float button)
                    { return std::abs(button) > 1; }))
    {
      std::cout << "invalide joy msg" << std::endl;
      return;
    }
    if (joy_msg->buttons.size() <= joyButtonMap["BUTTON_START"])
    {
      std::cerr << "[JoyController]:joy_msg has a error length, check your joystick_type!" << std::endl;
      return;
    }
    joystickOriginAxisTemp_.head(4) << joy_msg->axes[joyAxisMap["AXIS_LEFT_STICK_X"]], joy_msg->axes[joyAxisMap["AXIS_LEFT_STICK_Y"]], joy_msg->axes[joyAxisMap["AXIS_RIGHT_STICK_Z"]], joy_msg->axes[joyAxisMap["AXIS_RIGHT_STICK_YAW"]];
    joystickOriginAxisFilter_ = joystickOriginAxisTemp_;
    // joystickOriginAxisFilter_.head(4) = joystickFilter_.update(joystickOriginAxisTemp_.head(4));
    joystickOriginAxisFilter_.head(4) = joystickOriginAxisTemp_.head(4);
    joystickOriginAxisFilter_(3) = joystickOriginAxisTemp_(3);
    if (!Walkenable_)
    {
      joystickOriginAxisFilter_ = vector_t::Zero(6);
    }
    setJoyCmdState(joystickOriginAxisFilter_);
    joystickOriginAxisPre_ = joystickOriginAxisFilter_;
    if (!oldJoyMsg_.buttons[joyButtonMap["BUTTON_START"]] && joy_msg->buttons[joyButtonMap["BUTTON_START"]])
    {
      ros::ServiceClient client = controllerNh_.serviceClient<std_srvs::Trigger>("/humanoid_controller/real_initial_start");
      std_srvs::Trigger srv;
      if (client.call(srv))
      {
        ROS_INFO("[JoyControl] Service call successful");
      }
      else
      {
        ROS_ERROR("Failed to callRealInitializeSrv service");
      }
    }
    CommandData CommandData_;
    CommandData_ = getCommandData();
    if (!oldJoyMsg_.buttons[joyButtonMap["BUTTON_STANCE"]] && joy_msg->buttons[joyButtonMap["BUTTON_STANCE"]])
    {
      Walkenable_ = false;
      cmdTrotgait_ = false;
      ROS_INFO("[RL mode] Walkenable: False");
      ROS_INFO("[RL mode] Switch to Stance");
    }
    else if (!oldJoyMsg_.buttons[joyButtonMap["BUTTON_LB"]] && joy_msg->buttons[joyButtonMap["BUTTON_LB"]])
    {
      Walkenable_ = true;
      ROS_INFO("[RL mode] Walkenable: True");
    }
    else if (!oldJoyMsg_.buttons[joyButtonMap["BUTTON_TROT"]] && joy_msg->buttons[joyButtonMap["BUTTON_TROT"]])
    {
      if (Walkenable_)
      {
        if (CommandData_.cmdStance_)
        {
          cmdTrotgait_ = true;
          ROS_INFO("[RL mode] Switch to Trot");
        }
        else
        {
          cmdTrotgait_ = false;
          ROS_INFO("[RL mode] Switch to Stance");
        }
      }
      else
      {
        ROS_INFO("[RL mode] Walkenable is false, cannot switch to Trot");
      }
    }
    else if (!oldJoyMsg_.buttons[joyButtonMap["BUTTON_RL"]] && joy_msg->buttons[joyButtonMap["BUTTON_RL"]])
    {
      activateRlController();
    }
    // setCommandData(CommandData_);
    if (joy_msg->buttons[joyButtonMap["BUTTON_BACK"]])
      for (int i = 0; i < 5; i++)
      {
        std::cout << "publish stop message" << std::endl;
        std_msgs::Bool stop_msg;
        stop_msg.data = true;
        stop_pub.publish(stop_msg);
        ros::Duration(0.1).sleep();
      }
    oldJoyMsg_ = *joy_msg;
  }
  void humanoidController::starting(const ros::Time &time)
  {
    initial_status_ = initialState_;
    currentObservation_.state = initial_status_;
    while (!is_initialized_)
    {
      if (!is_nodelet_node_)
        ros::spinOnce();
      usleep(1000);
    }
    if (is_real_)
    {
      SensorData_t intial_sensor_data;
      hardware_interface_ptr_->init(intial_sensor_data);
      real_init_wait();
    }
    else
    {
      hardware_status_ = 1;
    }
    last_time_ = current_time_;
    updateStateEstimation(time, true);
    currentObservation_.mode = ModeNumber::SS;
    intail_input_ = vector_t::Zero(HumanoidInterface_->getCentroidalModelInfo().inputDim);
    for (int i = 0; i < 8; i++)
      intail_input_(3 * i + 2) = HumanoidInterface_->getCentroidalModelInfo().robotMass * 9.81 / 8; // 48.7*g/8
    optimizedInput_mrt_ = intail_input_;
    if (!is_real_ && !is_play_back_mode_)
      mujocoSimStart(controllerNh_);
    last_time_ = current_time_;
    if (!is_play_back_mode_)
      sensors_data_buffer_ptr_->sync();
    // 网络推理线程
    inferenceThread_ = std::thread(&humanoidController::inference_thread_func, this);
    if (!inferenceThread_.joinable())
    {
      ROS_ERROR_STREAM("Failed to start inference thread");
      exit(1);
    }
    if (rl_from_start_ && !is_real_ && !is_play_back_mode_)
    {
      if (!sim_keep_wbc_with_rl_)
      {
        activateRlControllerFromStart(false);
      }
      else
      {
        beginRlSpawnSettle();
      }
    }
  }
  void humanoidController::real_init_wait()
  {
    while (ros::ok())
    {
      if (ros::param::get("/hardware/is_ready", hardware_status_))
      {
        if (hardware_status_ == 1)
        {
          std::cerr << "real robot is ready\n";
          break;
        }
      }
      usleep(1000);
    }
  }

  void humanoidController::preUpdate(const ros::Time &time)
  {
    /*******************输入蹲姿和站姿**********************/
    auto &infoWBC = centroidalModelInfoWBC_;
    vector_t squatState = vector_t::Zero(infoWBC.stateDim);
    squatState.head(12 + jointNum_) = drake_interface_->getSquatInitialState();
    vector_t standState = vector_t::Zero(infoWBC.stateDim);
    standState.head(12 + jointNum_) = initialState_.head(12 + jointNum_);
    /*******采用 standUp_controller 从蹲姿运动到站姿*********/
    stateEstimate_->setFixFeetHeights(true);
    updateStateEstimation(time, false);

    static vector_t startObservation_state = currentObservation_.state;

    static double startTime = time.toSec();
    double endTime = startTime + (standState[8] - squatState[8]) / 0.11; // 以 0.11m/s 速度起立

    scalar_array_t timeTrajectory;
    timeTrajectory.push_back(startTime);
    timeTrajectory.push_back(endTime);
    vector_array_t stateTrajectory;
    stateTrajectory.push_back(startObservation_state);
    stateTrajectory.push_back(standState);
    vector_t standUpState = LinearInterpolation::interpolate(time.toSec(), timeTrajectory, stateTrajectory);
    vector_t measuredRbdState_;
    measuredRbdState_ = getRobotState();
    vector_t torque = standUpWbc_->update(standUpState, intail_input_, measuredRbdState_, ModeNumber::SS, dt_, false).tail(infoWBC.actuatedDofNum);

    kuavo_msgs::jointCmd jointCmdMsg;
    for (int i1 = 0; i1 < jointNumReal_; ++i1)
    {
      jointCmdMsg.joint_q.push_back(standUpState(12 + i1));
      jointCmdMsg.joint_v.push_back(0);
      jointCmdMsg.tau.push_back(torque(i1));
      jointCmdMsg.tau_ratio.push_back(1);
      jointCmdMsg.joint_kp.push_back(jointKp_[i1]);
      jointCmdMsg.joint_kd.push_back(jointKd_[i1]);
      jointCmdMsg.tau_max.push_back(kuavo_settings_.hardware_settings.max_current[i1]);
      jointCmdMsg.control_modes.push_back(2);
    }
    for (int i2 = 0; i2 < armNumReal_; ++i2)
    {
      jointCmdMsg.joint_q.push_back(output_pos_(jointNumReal_ + i2));
      jointCmdMsg.joint_v.push_back(output_vel_(jointNumReal_ + i2));
      jointCmdMsg.tau.push_back(output_tau_(jointNumReal_ + i2));
      jointCmdMsg.tau_ratio.push_back(1);
      jointCmdMsg.tau_max.push_back(kuavo_settings_.hardware_settings.max_current[jointNumReal_ + i2]);
      jointCmdMsg.control_modes.push_back(2);
    }
    for (int i3 = 0; i3 < headNum_; ++i3)
    {
      jointCmdMsg.joint_q.push_back(0);
      jointCmdMsg.joint_v.push_back(0);
      jointCmdMsg.tau.push_back(0);
      jointCmdMsg.tau_ratio.push_back(1);
      jointCmdMsg.tau_max.push_back(10);
      jointCmdMsg.control_modes.push_back(2);
    }
    jointCmdPub_.publish(jointCmdMsg);
    /*******************超过设置时间，退出******************/
    // 延迟启动, 避免切换不稳定
    if (time.toSec() > endTime + 0.3)
    {
      stateEstimate_->setFixFeetHeights(false);
      isPreUpdateComplete = true;
    }
  }

  void humanoidController::update(const ros::Time &time, const ros::Duration &dfd)
  {
    const auto t1 = Clock::now();
    ros::Duration period = ros::Duration(dt_);
    vector_t joyCmd = vector_t::Zero(6);
    joyCmd = getJoyCmdState();
    checkAndPublishCommandLine(joyCmd);
    updateStateEstimation(time, false);
    vector_t measuredRbdState_;
    measuredRbdState_ = getRobotState();
    const auto t2 = Clock::now();
    vector_t optimizedState_mrt, optimizedInput_mrt;
    auto &info = HumanoidInterface_->getCentroidalModelInfo();
    optimizedInput_mrt_ = intail_input_;
    optimizedState_mrt_ = initial_status_;
    if (rl_spawn_settling_)
    {
      optimizedState_mrt_ = spawn_centroidal_status_;
    }
    else if (is_rl_controller_ && !is_real_ && sim_keep_wbc_with_rl_)
    {
      Eigen::VectorXd local_action(jointNum_ + jointArmNum_);
      {
        std::lock_guard<std::mutex> lock(action_mtx_);
        local_action = actions_;
      }
      const int n = static_cast<int>(jointNum_ + jointArmNum_);
      if (sim_wbc_ref_filtered_.size() != n)
      {
        sim_wbc_ref_filtered_.resize(n);
        for (int i = 0; i < n; ++i)
        {
          sim_wbc_ref_filtered_[i] = initial_status_(12 + i);
        }
      }
      const double alpha = std::min(1.0, period.toSec() / std::max(sim_wbc_ref_filter_tau_, 1e-4));
      double active_blend = sim_wbc_rl_action_blend_;
      if (sim_wbc_rl_blend_ramp_sec_ > 0.0)
      {
        const double ramp_steps = sim_wbc_rl_blend_ramp_sec_ * inferenceFrequency_;
        const double ramp_t = std::min(1.0, static_cast<double>(rl_takeover_step_) / std::max(ramp_steps, 1.0));
        active_blend =
            sim_wbc_rl_blend_start_ + ramp_t * (sim_wbc_rl_action_blend_ - sim_wbc_rl_blend_start_);
      }
      for (int i = 0; i < n; ++i)
      {
        const double rl_joint =
            local_action[i] * actionScale_ * actionScaleTest_[i] + rlPdTargetJoint(i);
        const double stand_joint = initial_status_(12 + i);
        double delta = rl_joint - stand_joint;
        if (sim_wbc_max_rl_joint_offset_rad_ > 0.0)
        {
          delta = std::clamp(delta, -sim_wbc_max_rl_joint_offset_rad_, sim_wbc_max_rl_joint_offset_rad_);
        }
        const double blended = stand_joint + active_blend * delta;
        sim_wbc_ref_filtered_[i] += alpha * (blended - sim_wbc_ref_filtered_[i]);
        optimizedState_mrt_(12 + i) = sim_wbc_ref_filtered_[i];
      }
    }
    // optimized_mode_ = plannedMode_;
    currentObservation_.input = optimizedInput_mrt_;
    std::chrono::time_point<std::chrono::high_resolution_clock> t3;
    std::chrono::time_point<std::chrono::high_resolution_clock> t4;
    std::chrono::time_point<std::chrono::high_resolution_clock> t5;
    kuavo_msgs::jointCmd jointCmdMsg;
    const bool use_wbc_torque =
        !is_rl_controller_ || rl_wbc_torque_handoff_ || (!is_real_ && sim_keep_wbc_with_rl_ && is_rl_controller_);
    if (use_wbc_torque)
    {
      bool rl_spawn_ready = false;
      if (rl_spawn_settling_)
      {
        ++rl_spawn_settle_step_;
        const SensorData sd = getRobotSensorData();
        const bool upright = isGravityUpright(sd);
        if (upright)
        {
          ++rl_spawn_upright_count_;
        }
        else
        {
          rl_spawn_upright_count_ = 0;
        }
        if (rl_spawn_settle_step_ >= kRlSpawnSettleSteps &&
            rl_spawn_upright_count_ >= kRlSpawnUprightFrames)
        {
          rl_spawn_ready = true;
        }
        else if (rl_spawn_settle_step_ >= kRlSpawnSettleSteps * 3)
        {
          const Eigen::Vector3d g = gravityBodyFromSensor(sd);
          ROS_ERROR("[RL mode] settle timeout (upright_frames=%d), gravity_body=[%.3f,%.3f,%.3f] — NOT starting RL (use WBC stand + R instead)",
                    rl_spawn_upright_count_, g.x(), g.y(), g.z());
        }
      }
      if (currentObservation_.mode == ModeNumber::SS)
      {
        wbc_->setStanceMode(true);
      }
      else
      {
        wbc_->setStanceMode(false);
      }
      t3 = Clock::now();
      ros_logger_->publishVector("/humanoid_controller/optimizedInput_mrt_", optimizedInput_mrt_);
      ros_logger_->publishVector("/humanoid_controller/optimizedState_mrt_", optimizedState_mrt_);
      ros_logger_->publishVector("/humanoid_controller/optimizedState_mrt_/linear_vel_xyz", optimizedState_mrt_.head<3>());
      ros_logger_->publishVector("/humanoid_controller/optimizedState_mrt_/angular_vel_xyz", optimizedState_mrt_.segment<3>(3));
      ros_logger_->publishVector("/humanoid_controller/optimizedState_mrt_/pos_xyz", optimizedState_mrt_.segment<3>(6));
      ros_logger_->publishVector("/humanoid_controller/optimizedState_mrt_/angular_zyx", optimizedState_mrt_.segment<3>(9));
      ros_logger_->publishVector("/humanoid_controller/optimizedState_mrt_/joint_pos", optimizedState_mrt_.segment(12, info.actuatedDofNum));
      // ros_logger_->publishValue("/humanoid_controller/optimized_mode_", static_cast<double>(optimized_mode_));
      vector_t x = wbc_->update(optimizedState_mrt_, optimizedInput_mrt_, measuredRbdState_, plannedMode_, period.toSec());
      t4 = Clock::now();
      // 决策变量, 6*body_acc + 12*joint_acc + 3x4*contact_force + 12*torque = 42
      const vector_t &torque = x.tail(jointNum_ + jointArmNum_);
      const vector_t &wbc_planned_joint_acc = x.segment(6, jointNum_ + jointArmNum_);
      const vector_t &wbc_planned_body_acc = x.head(6);
      const vector_t &wbc_planned_contact_force = x.segment(6 + jointNum_ + jointArmNum_, wbc_->getContactForceSize());
      ros_logger_->publishVector("/humanoid_controller/torque", torque);
      ros_logger_->publishVector("/humanoid_controller/wbc_planned_joint_acc", wbc_planned_joint_acc);
      ros_logger_->publishVector("/humanoid_controller/wbc_planned_body_acc/linear", wbc_planned_body_acc.head<3>());
      ros_logger_->publishVector("/humanoid_controller/wbc_planned_body_acc/angular", wbc_planned_body_acc.tail<3>());
      ros_logger_->publishVector("/humanoid_controller/wbc_planned_contact_force/left_foot", wbc_planned_contact_force.head<12>());
      ros_logger_->publishVector("/humanoid_controller/wbc_planned_contact_force/right_foot", wbc_planned_contact_force.tail<12>());
      vector_t posDes = centroidal_model::getJointAngles(optimizedState_mrt_, HumanoidInterface_->getCentroidalModelInfo());
      vector_t velDes = centroidal_model::getJointVelocities(optimizedInput_mrt_, HumanoidInterface_->getCentroidalModelInfo());
      scalar_t dt = period.toSec();
      posDes = posDes + 0.5 * wbc_planned_joint_acc * dt * dt;
      velDes = velDes + wbc_planned_joint_acc * dt;
      auto current_jointPos = measuredRbdState_.segment(6, info.actuatedDofNum);
      auto current_jointVel = measuredRbdState_.segment(6 + info.generalizedCoordinatesNum, info.actuatedDofNum);
      output_tau_ = torque;
      last_wbc_tau_ = torque;
      if (rl_spawn_ready)
      {
        rl_spawn_settling_ = false;
        is_rl_controller_ = true;
        activateRlControllerFromStart(sim_keep_wbc_with_rl_);
      }
      t5 = Clock::now();
      // TODO: send the controller command to hardware interface
      std_msgs::Float32MultiArray targetTorqueMsg;
      for (int i1 = 0; i1 < jointNum_ + jointArmNum_; ++i1)
      {
        targetTorqueMsg.data.push_back(output_tau_(i1));
      }
      // targetTorquePub_.publish(targetTorqueMsg);
      jointCmdMsg.header.stamp = time;
      for (int i1 = 0; i1 < jointNum_ + jointArmNum_; ++i1)
      {
        jointCmdMsg.joint_q.push_back(posDes(i1));
        jointCmdMsg.joint_v.push_back(velDes(i1));
        jointCmdMsg.tau.push_back(output_tau_(i1));
        jointCmdMsg.joint_kp.push_back(0);
        jointCmdMsg.joint_kd.push_back(0);
        jointCmdMsg.tau_ratio.push_back(1);
        jointCmdMsg.tau_max.push_back(kuavo_settings_.hardware_settings.max_current[i1]);
        jointCmdMsg.control_modes.push_back(control_mode_);
      }
      if (rl_wbc_torque_handoff_ && ros::Time::now() >= rl_wbc_handoff_end_time_)
      {
        rl_wbc_torque_handoff_ = false;
        finalizeRlWbcHandoff();
      }
    }
    else
    {
      auto current_jointPos = measuredRbdState_.segment(6, info.actuatedDofNum);
      auto current_jointVel = measuredRbdState_.segment(12 + jointNum_ + jointArmNum_, info.actuatedDofNum);
      t3 = Clock::now();
      Eigen::VectorXd actuation(jointNum_ + jointArmNum_);
      actuation.setZero();
      actuation = updateRLcmd(measuredRbdState_);
      t4 = Clock::now();
      jointCmdMsg.header.stamp = time;
      if (!is_real_)
      {
        Eigen::VectorXd local_action(jointNum_ + jointArmNum_);
        {
          std::lock_guard<std::mutex> lock(action_mtx_);
          local_action = actions_;
        }
        const double crossfade_alpha =
            rl_torque_crossfade_active_
                ? std::min(1.0, (ros::Time::now() - rl_torque_crossfade_start_time_).toSec() / kRlTorqueCrossfadeSec)
                : 0.0;
        for (int i1 = 0; i1 < jointNum_ + jointArmNum_; ++i1)
        {
          const double q_des =
              local_action[i1] * actionScale_ * actionScaleTest_[i1] + defalutJointPos_[i1];
          double ff_tau = 0.0;
          if (rl_torque_crossfade_active_ && i1 < wbc_tau_at_rl_takeover_.size())
          {
            ff_tau = (1.0 - crossfade_alpha) * wbc_tau_at_rl_takeover_[i1];
          }
          jointCmdMsg.joint_q.push_back(q_des);
          jointCmdMsg.joint_v.push_back(0.0);
          jointCmdMsg.joint_kp.push_back(jointKp_[i1]);
          jointCmdMsg.joint_kd.push_back(jointKd_[i1]);
          jointCmdMsg.tau.push_back(ff_tau);
          jointCmdMsg.tau_ratio.push_back(1);
          jointCmdMsg.tau_max.push_back(torqueLimits_[i1]);
          jointCmdMsg.control_modes.push_back(JointControlMode_(i1));
        }
        if (rl_torque_crossfade_active_ && crossfade_alpha >= 1.0)
        {
          rl_torque_crossfade_active_ = false;
        }
      }
      else
      {
        for (int i1 = 0; i1 < jointNum_ + jointArmNum_; ++i1)
        {
          if (JointControlMode_(i1) == 0)
          {
            if (JointPDMode_(i1) == 0)
            {
              jointCmdMsg.joint_q.push_back(0.0);
              jointCmdMsg.joint_v.push_back(0.0);
              jointCmdMsg.joint_kp.push_back(0);
              jointCmdMsg.joint_kd.push_back(0);
              jointCmdMsg.tau.push_back(actuation(i1));
              jointCmdMsg.tau_ratio.push_back(1);
              jointCmdMsg.tau_max.push_back(torqueLimits_[i1]);
              jointCmdMsg.control_modes.push_back(JointControlMode_(i1));
            }
            else
            {
              jointCmdMsg.joint_q.push_back(actuation(i1));
              jointCmdMsg.joint_v.push_back(0.0);
              jointCmdMsg.joint_kp.push_back(jointKp_[i1]);
              jointCmdMsg.joint_kd.push_back(jointKd_[i1]);
              jointCmdMsg.tau.push_back(0.0);
              jointCmdMsg.tau_ratio.push_back(1);
              jointCmdMsg.tau_max.push_back(torqueLimits_[i1]);
              jointCmdMsg.control_modes.push_back(JointControlMode_(i1));
            }
          }
          else
          {
            jointCmdMsg.joint_q.push_back(current_jointPos(i1));
            jointCmdMsg.joint_v.push_back(0.0);
            jointCmdMsg.joint_kp.push_back(jointKp_[i1]);
            jointCmdMsg.joint_kd.push_back(jointKd_[i1]);
            jointCmdMsg.tau.push_back(actuation(i1));
            jointCmdMsg.tau_ratio.push_back(1);
            jointCmdMsg.tau_max.push_back(torqueLimits_[i1]);
            jointCmdMsg.control_modes.push_back(JointControlMode_(i1));
          }
        }
      }
      t5 = Clock::now();
    }

    // 补充头部维度
    // 计算头部反馈力
    if (headNum_ > 0)
    {
      vector_t get_head_pos = vector_t::Zero(headNum_);
      head_mtx.lock();
      get_head_pos = desire_head_pos_;
      head_mtx.unlock();
      auto &hardware_settings = kuavo_settings_.hardware_settings;
      vector_t head_feedback_tau = vector_t::Zero(headNum_);
      vector_t head_feedback_vel = vector_t::Zero(headNum_);
      if (!is_real_) // 实物不需要头部反馈力，来自kuavo仓库的移植
        head_feedback_tau = head_kp_.cwiseProduct(get_head_pos - sensor_data_head_.jointPos_) + head_kd_.cwiseProduct(-sensor_data_head_.jointVel_);
      for (int i3 = 0; i3 < headNum_; ++i3)
      {
        auto cur_head_pos = sensor_data_head_.jointPos_ * TO_DEGREE;
        auto vel = (get_head_pos[i3] - sensor_data_head_.jointPos_[i3]) * TO_DEGREE / dt_ * ruiwo_motor_velocities_factor_;
        double head_limit_vel = hardware_settings.joint_velocity_limits[jointNum_ + armNumReal_ + i3];

        vel = std::clamp(vel, -head_limit_vel, head_limit_vel) * TO_RADIAN;
        jointCmdMsg.joint_q.push_back(get_head_pos(i3));
        jointCmdMsg.joint_v.push_back(0);
        jointCmdMsg.tau.push_back(head_feedback_tau(i3));
        jointCmdMsg.tau_ratio.push_back(1);
        jointCmdMsg.tau_max.push_back(10);
        jointCmdMsg.control_modes.push_back(2);
      }
    }

    jointCmdPub_.publish(jointCmdMsg);
    {
      std::lock_guard<std::mutex> lock(joint_cmd_mutex_);
      for (int i1 = 0; i1 < jointNum_ + jointArmNum_; ++i1)
      {
        jointTorqueCmd_(i1) = jointCmdMsg.tau[i1];
      }
    }
    const auto t6 = Clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t1).count() > 1000)
    {
      std::cout << "t1-t2: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
      std::cout << "t2-t3: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << std::endl;
      std::cout << "t3-t4: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << std::endl;
      std::cout << "t4-t5: " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << " ms" << std::endl;
      std::cout << "t5-t6: " << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << " ms" << std::endl;
    }
    // publish time cost
    std_msgs::Float64 msg;
    wbcFrequencyPub_.publish(msg);
    wbcTimeCostPub_.publish(msg);
    static double last_ros_time = ros::Time::now().toSec();
    ros_logger_->publishValue("/monitor/time_cost/controller_loop_time", (ros::Time::now().toSec() - last_ros_time) * 1000);
    last_ros_time = ros::Time::now().toSec();
  }
  void humanoidController::applySensorData(const SensorData &data)
  {
    SensorData sensor_data_copy = data;
    jointPos_ = data.jointPos_;
    jointVel_ = data.jointVel_;
    jointAcc_ = data.jointAcc_;
    jointCurrent_ = data.jointCurrent_;
    quat_ = data.quat_;
    angularVel_ = data.angularVel_;
    linearAccel_ = data.linearAccel_;
    freeLinearAccel_ = data.freeLinearAccel_;
    orientationCovariance_ = data.orientationCovariance_;
    angularVelCovariance_ = data.angularVelCovariance_;
    linearAccelCovariance_ = data.linearAccelCovariance_;
    current_time_ = data.timeStamp_;
    stateEstimate_->updateImu(quat_, angularVel_, linearAccel_, orientationCovariance_, angularVelCovariance_, linearAccelCovariance_);
    sensor_data_copy.jointPos_ = jointPos_;
    sensor_data_copy.jointVel_ = jointVel_;
    sensor_data_copy.jointAcc_ = jointAcc_;
    sensor_data_copy.jointCurrent_ = jointCurrent_;
    sensor_data_copy.quat_ = quat_;
    sensor_data_copy.angularVel_ = angularVel_;
    sensor_data_copy.linearAccel_ = linearAccel_;
    sensor_data_copy.freeLinearAccel_ = freeLinearAccel_;
    sensor_data_copy.quat_offset_ = stateEstimate_->getImuOrientation();
    setRobotSensorData(sensor_data_copy);
  }
  void humanoidController::updateStateEstimation(const ros::Time &time, bool is_init)
  {
    // vector_t jointPos(jointNum_), jointVel(jointNum_), jointCurrent(jointNum_);
    contact_flag_t contacts;
    Eigen::Quaternion<scalar_t> quat;
    // contact_flag_t contactFlag;
    vector3_t angularVel, linearAccel;
    matrix3_t orientationCovariance, angularVelCovariance, linearAccelCovariance;
    SensorData sensors_data;
    vector_t measuredRbdState_;
    measuredRbdState_ = getRobotState();
    if (is_init)
      sensors_data = sensors_data_buffer_ptr_->getLastData();
    else
      sensors_data = sensors_data_buffer_ptr_->getLastData();
    applySensorData(sensors_data);
    if (is_init)
    {
      last_time_ = current_time_ - ros::Duration(0.001);
      stateEstimate_->updateJointStates(jointPos_, jointVel_);
      stateEstimate_->updateIntialEulerAngles(quat_);
      applySensorData(sensors_data);
      stateEstimate_->set_intial_state(currentObservation_.state);
      measuredRbdState_ = stateEstimate_->getRbdState();
      // std::cout << "initial measuredRbdState_:" << measuredRbdState_.transpose() << std::endl;
    }
    double diff_time = (current_time_ - last_time_).toSec();
    // auto est_mode = stateEstimate_->ContactDetection(plannedMode_, jointVel_, jointCurrent_, diff_time);
    if (is_rl_controller_)
      plannedMode_ = rl_plannedMode_;
    last_time_ = current_time_;
    ros::Duration period = ros::Duration(diff_time);
    vector_t activeTorque_ = jointCurrent_;
    stateEstimate_->setCmdTorque(activeTorque_);
    stateEstimate_->estContactForce(period);
    auto est_contact_force = stateEstimate_->getEstContactForce();
    ros_logger_->publishVector("/state_estimate/contactForce", est_contact_force);
    stateEstimate_->updateMode(plannedMode_);
    ros_logger_->publishValue("/rl_controller/rl_optimized_mode_", static_cast<double>(plannedMode_));
    if (diff_time > 0.00005 || is_init)
    {
      Eigen::VectorXd updated_joint_pos = jointPos_;
      Eigen::VectorXd updated_joint_vel = jointVel_;
      Eigen::VectorXd updated_joint_current = jointCurrent_;
#ifdef KUAVO_CONTROL_LIB_FOUND
      if (use_joint_filter_)
      {
        joint_filter_ptr_->update(measuredRbdState_, updated_joint_pos, updated_joint_vel, updated_joint_current, output_tau_, plannedMode_);
      }
#endif
      stateEstimate_->updateJointStates(updated_joint_pos, updated_joint_vel); // 使用关节滤波之后的jointPos和jointVel更新状态估计器
      contactTrotgait_ = stateEstimate_->updateKinematics(current_time_, period);
      measuredRbdState_ = stateEstimate_->update(time, period); // angle(zyx),pos(xyz),jointPos[info_.actuatedDofNum],angularVel(zyx),linervel(xyz),jointVel[info_.actuatedDofNum]
      currentObservation_.time += period.toSec();
    }
    setRobotState(measuredRbdState_);
    ros_logger_->publishVector("/state_estimate/measuredRbdState", measuredRbdState_);
    ros_logger_->publishVector("/state_estimate/measuredRbdState/angle_zyx", measuredRbdState_.segment(0, 3));
    ros_logger_->publishVector("/state_estimate/measuredRbdState/pos_xyz", measuredRbdState_.segment(3, 3));
    auto &info = HumanoidInterface_->getCentroidalModelInfo();
    ros_logger_->publishVector("/state_estimate/measuredRbdState/joint_pos", measuredRbdState_.segment(6, info.actuatedDofNum));
    ros_logger_->publishVector("/state_estimate/measuredRbdState/angular_vel_xyz", measuredRbdState_.segment(6 + info.actuatedDofNum, 3));
    ros_logger_->publishVector("/state_estimate/measuredRbdState/linear_vel_xyz", measuredRbdState_.segment(9 + info.actuatedDofNum, 3));
    ros_logger_->publishVector("/state_estimate/measuredRbdState/joint_vel", measuredRbdState_.tail(info.actuatedDofNum));
    scalar_t yawLast = currentObservation_.state(9);
    currentObservation_.state = rbdConversions_->computeCentroidalStateFromRbdModel(measuredRbdState_);
    currentObservation_.state(9) = yawLast + angles::shortest_angular_distance(yawLast, currentObservation_.state(9));
    std_msgs::Float32MultiArray state;
    for (int i1 = 0; i1 < currentObservation_.state.rows(); ++i1)
    {
      state.data.push_back(currentObservation_.state(i1));
    }
    currentObservation_.mode = plannedMode_;
  }
  void humanoidController::inference_thread_func()
  {
    ros::Rate inference_rate(inferenceFrequency_);
    while (ros::ok())
    {
      if (!is_rl_controller_ || rl_spawn_settling_)
      {
        inference_rate.sleep();
        continue;
      }
      const auto sensors_data = getRobotSensorData();
      const auto measuredRbdState_ = getRobotState();
      Eigen::VectorXd local_measure_state = measuredRbdState_;
      updateObservation(local_measure_state, sensors_data);
      inference();
      inference_rate.sleep();
    }
  }
  void humanoidController::inference()
  {
    infer_request_ = compiled_model_.create_infer_request();
    const auto input_port = compiled_model_.input();
    Eigen::VectorXf float_network_input = networkInputData_.cast<float>();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), float_network_input.data());
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.start_async();
    infer_request_.wait();
    const auto output_tensor = infer_request_.get_output_tensor();
    const size_t output_buf_length = output_tensor.get_size();
    const auto output_buf = output_tensor.data<float>();
    const size_t expected_output_length = withArm_ ? jointNum_ + jointArmNum_ : jointNum_;
    if (output_buf_length != expected_output_length)
    {
      std::cout << "神经网络输出维度错误！！维度等于： " << output_buf_length << std::endl;
      return;
    }
    Eigen::VectorXd actionTargetPos_(output_buf_length);
    {
      std::lock_guard<std::mutex> lock(action_mtx_);
      for (int i = 0; i < output_buf_length; ++i)
      {
        double action = output_buf[i];
        if (rl_wbc_torque_handoff_)
        {
          action = 0.0;
        }
        else if (!sim_keep_wbc_with_rl_ && rl_takeover_step_ < kRlActionRampSteps)
        {
          const double t =
              static_cast<double>(rl_takeover_step_ + 1) / static_cast<double>(kRlActionRampSteps);
          const double cap = kRlActionRampMax * t;
          action = std::clamp(static_cast<double>(output_buf[i]) * t, -cap, cap);
        }
        else
        {
          action = output_buf[i];
        }
        if (sim_rl_debug_zero_action_ && !is_real_)
        {
          action = 0.0;
        }
        actions_[i] = action;
        actionTargetPos_[i] = action * actionScale_ + rlPdTargetJoint(i);
      }
      clip(actions_, jointNum_ + jointArmNum_, clipActions_);
      ros_logger_->publishValue("/rl_controller/rl_takeover_step", static_cast<double>(rl_takeover_step_));
      if (rl_takeover_step_ == 0 && !rl_wbc_torque_handoff_ && !rl_action_ramp_logged_)
      {
        rl_action_ramp_logged_ = true;
        if (!sim_keep_wbc_with_rl_)
        {
          ROS_INFO("[RL mode] First RL inference step: max|action|=%.4f (ramp %d steps, crossfade %.0fms)",
                   actions_.cwiseAbs().maxCoeff(), kRlActionRampSteps, kRlTorqueCrossfadeSec * 1000.0);
        }
        else
        {
          ROS_INFO("[RL mode] First RL inference step: max|action|=%.4f (no ramp, WBC blend=%.2f)",
                   actions_.cwiseAbs().maxCoeff(), sim_wbc_rl_action_blend_);
        }
      }
      ros_logger_->publishVector("/rl_controller/actions", actions_);
      ros_logger_->publishVector("/rl_controller/actionTargetPos", actionTargetPos_);
    }
    // One policy step per inference tick (50Hz), matching Isaac Lab decimation.
    if (is_rl_controller_ && !rl_wbc_torque_handoff_)
    {
      episodeLength_++;
      rl_takeover_step_++;
    }
  }
  void humanoidController::updatePhase(const CommandData &CommandData_)
  {
    if (is_rl_controller_)
    {
      // Match training action_phase: one policy step == one frame index.
      int num_frames = std::max(1, static_cast<int>(std::lround(cycleTime_ * inferenceFrequency_)));
      if (hasDanceReference_ && !danceReferenceTrajectoryGym_.empty())
      {
        num_frames = static_cast<int>(danceReferenceTrajectoryGym_.size());
      }
      phase_ = static_cast<double>(episodeLength_ % num_frames) / static_cast<double>(num_frames);
      // In-place dance: keep double support like Isaac Lab training (no alternating trot).
      rl_plannedMode_ = ModeNumber::SS;
    }
    else
    {
      const double inference_dt = 1.0 / inferenceFrequency_;
      phase_ = CommandData_.cmdStance_ == 1 ? 0 : episodeLength_ * inference_dt / cycleTime_;
      commandPhase_(0) = sin(2 * M_PI * phase_);
      commandPhase_(1) = cos(2 * M_PI * phase_);
      rl_plannedMode_ = (commandPhase_(0) > 0) ? ModeNumber::SF : (commandPhase_(0) < 0) ? ModeNumber::FS
                                                                                         : ModeNumber::SS;
      return;
    }
    commandPhase_(0) = sin(2 * M_PI * phase_);
    commandPhase_(1) = cos(2 * M_PI * phase_);
  }
  void humanoidController::updateObservation(const Eigen::VectorXd &state_est, const SensorData &sensor_data)
  {
    CommandData CommandData_;
    CommandData_ = getCommandData();
    updatePhase(CommandData_);
    // Extract state data
    const Eigen::Vector3d baseEuler(state_est(2), state_est(1), state_est(0));
    const Eigen::Vector3d baseAngVel(state_est(6 + jointNum_ + jointArmNum_),
                                     state_est(6 + jointNum_ + jointArmNum_ + 1),
                                     state_est(6 + jointNum_ + jointArmNum_ + 2));
    const Eigen::Vector3d baseLineVel = state_est.segment(9 + jointNum_ + jointArmNum_, 3);
    const Eigen::Vector3d basePos = state_est.segment(3, 3);
    // Extract and process sensor data
    Eigen::VectorXd jointPos = sensor_data.jointPos_ - defalutJointPos_;
    if (!is_real_ && sim_handoff_obs_offset_)
    {
      jointPos -= sim_joint_obs_offset_;
    }
    const Eigen::VectorXd &jointVel = sensor_data.jointVel_;
    Eigen::VectorXd jointTorque = sensor_data.jointCurrent_;
    const Eigen::Vector3d &bodyAngVel = sensor_data.angularVel_;
    const Eigen::Vector3d &bodyLineAcc = sensor_data.linearAccel_;
    const Eigen::Vector3d &bodyLineFreeAcc = sensor_data.freeLinearAccel_;
    // Normalize joint torques
    for (int i = 0; i < jointNum_ + jointArmNum_; ++i)
    {
      jointTorque[i] /= torqueLimits_[i];
    }
    // Transform base linear velocity
    const Eigen::Matrix3d R = sensor_data.quat_offset_.matrix();
    const Eigen::Vector3d bodyLineVel = R.transpose() * baseLineVel;
    const Eigen::Vector3d gravity_base(0.0, 0.0, -1.0);
    const Eigen::Vector3d gravity_body = R.transpose() * gravity_base;
    // Get local action
    Eigen::VectorXd local_action;
    {
      std::lock_guard<std::mutex> lock(action_mtx_);
      local_action = actions_;
      if (!is_rl_controller_)
      {
        local_action.setZero();
      }
    }
    // In-place dance: velocity command must stay zero (matches training).
    Eigen::VectorXd tempCommand_(3);
    tempCommand_.setZero();
    if (!(hasDanceReference_ && is_rl_controller_))
    {
      CommandData_.scale();
      tempCommand_ = CommandData_.getCommand().head(3);
    }
    referenceJointPos_.setZero();
    if (hasDanceReference_ && !danceReferenceTrajectoryGym_.empty())
    {
      const int frame_idx = episodeLength_ % static_cast<int>(danceReferenceTrajectoryGym_.size());
      referenceJointPos_ = danceReferenceTrajectoryGym_[frame_idx];
    }
    // Populate singleInputDataMap_
    const std::map<std::string, Eigen::VectorXd> singleInputDataMap_ = {
        {"gravity_body", gravity_body},
        {"baseEuler", baseEuler},
        {"baseAngVel", baseAngVel},
        {"baseLineVel", baseLineVel},
        {"basePos", basePos},
        {"jointPos", jointPos},
        {"jointVel", jointVel},
        {"jointTorque", jointTorque},
        {"bodyAngVel", bodyAngVel},
        {"bodyLineAcc", bodyLineAcc},
        {"bodyLineFreeAcc", bodyLineFreeAcc},
        {"bodyLineVel", bodyLineVel},
        {"commandPhase", commandPhase_},
        {"command", tempCommand_},
        {"referenceJointPos", referenceJointPos_},
        {"action", local_action}};
    // Fill singleInputData
    int index = 0;
    for (const auto &key : singleInputDataKeys)
    {
      const auto &value = singleInputDataID_[key];
      singleInputData.segment(index, value[1]) = singleInputDataMap_.at(key).segment(value[0], value[1]) * value[2];
      index += value[1];
    }
    // Clip and update input_deque
    clip(singleInputData, numSingleObs_, clipObservations_);
    if (rl_obs_reset_pending_.exchange(false))
    {
      input_deque.clear();
      for (int i = 0; i < frameStack_; ++i)
      {
        input_deque.push_back(singleInputData);
      }
      ROS_INFO("[RL mode] Observation buffer reset from current measured state");
    }
    else
    {
      input_deque.push_back(singleInputData);
      input_deque.pop_front();
    }
    // Update networkInputData_
    for (int i = 0; i < frameStack_; ++i)
    {
      networkInputData_.segment(i * numSingleObs_, numSingleObs_) = input_deque[i];
    }
    // Publish data
    ros_logger_->publishVector("/rl_controller/singleInputData", singleInputData);
  }
  Eigen::VectorXd humanoidController::updateRLcmd(const vector_t &state)
  {
    Eigen::VectorXd actuation(jointNum_ + jointArmNum_);
    Eigen::VectorXd cmd_out(jointNum_ + jointArmNum_);
    Eigen::VectorXd cmd(jointNum_ + jointArmNum_);
    Eigen::VectorXd cmd_filter(jointNum_ + jointArmNum_);
    Eigen::VectorXd torque(jointNum_ + jointArmNum_);
    const Eigen::VectorXd jointPos_ = state.segment(6, jointNum_ + jointArmNum_);
    const Eigen::VectorXd jointVel_ = state.tail(jointNum_ + jointArmNum_);
    Eigen::VectorXd motorPos_ = jointPos_;
    Eigen::VectorXd motorVel_ = jointVel_;
    Eigen::VectorXd local_action;
    {
      std::lock_guard<std::mutex> lock(action_mtx_);
      local_action = actions_;
    }
    if (!withArm_)
    {
      local_action.tail(jointArmNum_).setZero();
    }
    Eigen::VectorXd jointTor_(jointNum_ + jointArmNum_);
    if (is_real_)
    {
      motorPos_.head(jointNum_) = ankleSolver.joint_to_motor_position(jointPos_.head(jointNum_));
      motorVel_.head(jointNum_) = ankleSolver.joint_to_motor_velocity(jointPos_.head(jointNum_), motorPos_, jointVel_.head(jointNum_));
      jointTor_ = -(jointKd_.cwiseProduct(motorVel_));
      jointTor_.head(jointNum_) = ankleSolver.motor_to_joint_torque(jointPos_.head(jointNum_), motorPos_, jointTor_.head(jointNum_));
      for (int i = 0; i < jointNum_ + jointArmNum_; i++)
      {
        jointTor_(i) = jointTor_(i) + jointKp_(i) * (local_action[i] * actionScale_ * actionScaleTest_[i] - jointPos_[i] + defalutJointPos_[i]);
      }
    }
    else
    {
      // Match Isaac Lab DelayedPDActuator: joint-space PD, no ankle motor mapping in sim.
      for (int i = 0; i < jointNum_ + jointArmNum_; i++)
      {
        jointTor_(i) = jointKp_(i) * (local_action[i] * actionScale_ * actionScaleTest_[i] - jointPos_[i] +
                                      rlPdTargetJoint(i)) -
                       jointKd_(i) * jointVel_[i];
      }
    }
    if (is_real_)
    {
      for (int i = 0; i < jointNum_ + jointArmNum_; i++)
      {
        if (JointControlMode_(i) == 0)
        {
          if (JointPDMode_(i) == 0)
          {
            cmd[i] = jointKp_[i] * (local_action[i] * actionScale_ * actionScaleTest_[i] - jointPos_[i] + defalutJointPos_[i]) - jointKd_[i] * jointVel_[i];
            cmd[i] = std::clamp(cmd[i], -torqueLimits_[i], torqueLimits_[i]);
            torque[i] = cmd[i];
          }
          else
          {
            cmd[i] = (local_action[i] * actionScale_ * actionScaleTest_[i] + defalutJointPos_[i]);
            torque[i] = jointKp_[i] * (local_action[i] * actionScale_ * actionScaleTest_[i] - jointPos_[i] + defalutJointPos_[i]) - jointKd_[i] * jointVel_[i];
          }
        }
        else if (JointControlMode_(i) == 2)
        {
          cmd[i] = jointKp_[i] * (local_action[i] * actionScale_ * actionScaleTest_[i] - jointPos_[i] + defalutJointPos_[i]);
          torque[i] = jointTor_[i];
        }
      }
    }
    else
    {
      for (int i = 0; i < jointNum_ + jointArmNum_; i++)
      {
        if (JointControlMode_(i) == 0)
        {
          cmd[i] = jointKp_[i] * (local_action[i] * actionScale_ * actionScaleTest_[i] - jointPos_[i] + defalutJointPos_[i]) - jointKd_[i] * jointVel_[i];
        }
        else if (JointControlMode_(i) == 2)
        {
          cmd[i] = jointTor_[i];
        }
        cmd[i] = std::clamp(cmd[i], -torqueLimits_[i], torqueLimits_[i]);
        torque[i] = cmd[i];
        if (std::isnan(cmd[i]) || std::isnan(torque[i])) {
        cmd[i] = 0.0;
        torque[i] = 0.0;
        }
      }

    }


    cmd_filter = jointCmdFilter_.update(cmd);
    cmd_out = cmd_filter.cwiseProduct(jointCmdFilterState_) + cmd.cwiseProduct(Eigen::VectorXd::Ones(jointNum_ + jointArmNum_) - jointCmdFilterState_);
    actuation = cmd_out;
    ros_logger_->publishVector("/rl_controller/cmd_out", cmd_out);
    ros_logger_->publishVector("/rl_controller/torque", torque);
    ros_logger_->publishVector("/rl_controller/cmd_filter", cmd_filter);
    ros_logger_->publishVector("/rl_controller/cmd", cmd);
    return actuation;
  }

  void humanoidController::clip(Eigen::VectorXd &a, int num, double limit)
  {
    a = a.cwiseMax(-limit).cwiseMin(limit);
  }

  humanoidController::~humanoidController()
  {
    controllerRunning_ = false;
  }

  void humanoidController::setupHumanoidInterface(const std::string &taskFile, const std::string &urdfFile, const std::string &referenceFile, const std::string &gaitCommandFile,
                                                  bool verbose, int robot_version_int)
  {
    HumanoidInterface_ = std::make_shared<HumanoidInterface>(taskFile, urdfFile, referenceFile, gaitCommandFile, robot_version_int);
    rbdConversions_ = std::make_shared<CentroidalModelRbdConversions>(HumanoidInterface_->getPinocchioInterface(),
                                                                      HumanoidInterface_->getCentroidalModelInfo());
  }

  void humanoidController::setupStateEstimate(const std::string &taskFile, bool verbose)
  {
    stateEstimate_ = std::make_shared<KalmanFilterEstimate>(HumanoidInterface_->getPinocchioInterface(),
                                                            HumanoidInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_);
    dynamic_cast<KalmanFilterEstimate &>(*stateEstimate_).loadSettings(taskFile, verbose);
    currentObservation_.time = 0;
  }

  void humanoidCheaterController::setupStateEstimate(const std::string & /*taskFile*/, bool /*verbose*/)
  {
    stateEstimate_ = std::make_shared<FromTopicStateEstimate>(HumanoidInterface_->getPinocchioInterface(),
                                                              HumanoidInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_, controllerNh_);
  }

  void humanoidKuavoController::setupStateEstimate(const std::string &taskFile, bool verbose)
  {
#ifdef KUAVO_CONTROL_LIB_FOUND
    // auto [plant, context] = drake_interface_->getPlantAndContext();
    stateEstimate_ = std::make_shared<InEkfBaseFilter>(HumanoidInterface_->getPinocchioInterface(),
                                                       HumanoidInterface_->getCentroidalModelInfo(),
                                                       *eeKinematicsPtr_,
                                                       drake_interface_,
                                                       dt_,
                                                       ros_logger_);
    std::cout << "InEkfBaseFilter stateEstimate_ initialized" << std::endl;
#endif
  }

} // namespace humanoid_controller
// PLUGINLIB_EXPORT_CLASS(humanoid_controller::humanoidController)
// PLUGINLIB_EXPORT_CLASS(humanoid_controller::humanoidCheaterController)
