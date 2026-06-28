#pragma once
#include <controller_interface/controller.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <humanoid_common/hardware_interface/ContactSensorInterface.h>
#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
#include <humanoid_estimation/StateEstimateBase.h>
#include <humanoid_interface/HumanoidInterface.h>
#include <humanoid_wbc/WbcBase.h>
#include "kuavo_msgs/gaitTimeName.h"
#include "std_srvs/Trigger.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Bool.h"
#include "sensor_msgs/Joy.h"
#include "sensor_msgs/Imu.h"
#include "kuavo_msgs/sensorsData.h"
#include "kuavo_msgs/jointCmd.h"
#include "humanoid_interface/common/TopicLogger.h"
#include "humanoid_interface_drake/humanoid_interface_drake.h"
#ifdef KUAVO_CONTROL_LIB_FOUND
#include "kuavo_estimation/joint_filter/joint_filter.h"
#endif
#include "humanoid_common/hardware_interface/hardware_interface_ros.h"
#include <atomic>
#include <queue>
#include <mutex>
#include "humanoid_controllers/LowPassFilter.h"
#include "humanoid_controllers/LowPassFilter5thOrder.h"
#include "kuavo_solver/ankle_solver.h"
#include "kuavo_msgs/robotHeadMotionData.h"
#include <openvino/openvino.hpp>
namespace humanoid_controller
{
  using namespace ocs2;
  using namespace humanoid;
  struct CommandData
  {
    double cmdVelLineX_;
    double cmdVelLineY_;
    double cmdVelLineZ_;
    double cmdVelAngularX_;
    double cmdVelAngularY_;
    double cmdVelAngularZ_;
    double cmdStance_;
    double cmdVelScaleLineX_;
    double cmdVelScaleLineY_;
    double cmdVelScaleLineZ_;
    double cmdVelScaleAngularX_;
    double cmdVelScaleAngularY_;
    double cmdVelScaleAngularZ_;
    double cmdScaleStance_;
    void setzero()
    {
      cmdVelLineX_ = 0.0;
      cmdVelLineY_ = 0.0;
      cmdVelLineZ_ = 0.0;
      cmdVelAngularX_ = 0.0;
      cmdVelAngularY_ = 0.0;
      cmdVelAngularZ_ = 0.0;
      cmdStance_ = 0.0;
    }
    void scale()
    {
      cmdVelLineX_ *= cmdVelScaleLineX_;
      cmdVelLineY_ *= cmdVelScaleLineY_;
      cmdVelLineZ_ *= cmdVelScaleLineZ_;
      cmdVelAngularX_ *= cmdVelScaleAngularX_;
      cmdVelAngularY_ *= cmdVelScaleAngularY_;
      cmdVelAngularZ_ *= cmdVelScaleAngularZ_;
      cmdStance_ *= cmdScaleStance_;
    }
    Eigen::VectorXd getCommand()
    {
      Eigen::VectorXd dynamicVector(4);
      dynamicVector << cmdVelLineX_, cmdVelLineY_, cmdVelAngularZ_, cmdStance_;
      return dynamicVector;
    }
  };
  struct gaitTimeName
  {
    std::string name;
    double startTime;
  };
  struct SensorData
  {
    ros::Time timeStamp_;
    vector_t jointPos_;
    vector_t jointVel_;
    vector_t jointAcc_;
    vector_t jointCurrent_;
    vector3_t angularVel_;
    vector3_t linearAccel_;
    vector3_t freeLinearAccel_;
    Eigen::Quaternion<scalar_t> quat_;
    matrix3_t orientationCovariance_;
    matrix3_t angularVelCovariance_;
    matrix3_t linearAccelCovariance_;
    Eigen::Quaternion<scalar_t> quat_offset_;
    void resize_joint(size_t num)
    {
      jointPos_.resize(num);
      jointVel_.resize(num);
      jointAcc_.resize(num);
      jointCurrent_.resize(num);
    }
  };

  struct SystemObservation
  {
    size_t mode = 0;
    scalar_t time = 0.0;
    vector_t state;
    vector_t input;

    friend void swap(SystemObservation &a, SystemObservation &b) noexcept;
  };
  class humanoidController
  {
  public:
    humanoidController() = default;
    ~humanoidController();
    void keyboard_thread_func();
    bool init(HybridJointInterface *robot_hw, ros::NodeHandle &controller_nh, bool is_nodelet_node = false);
    void preUpdate(const ros::Time &time);
    bool preUpdateComplete() {return isPreUpdateComplete;}
    void update(const ros::Time &time, const ros::Duration &period);
    void starting(const ros::Time &time);
    void applySensorData(const SensorData &data);
    void inference_thread_func();
    void inference();
    void updatePhase(const CommandData &CommandData_);
    void updateObservation(const Eigen::VectorXd &state_est, const SensorData &sensor_data);
    Eigen::VectorXd updateRLcmd(const vector_t &state);
    void headCmdCallback(const kuavo_msgs::robotHeadMotionData::ConstPtr &msg);
    void joyCallback(const sensor_msgs::Joy::ConstPtr &joy_msg);
    std::vector<bool> commandLineToTargetTrajectories(const vector_t &joystick_origin_axis, geometry_msgs::Twist &cmdVel);
    void checkAndPublishCommandLine(const vector_t &joystick_origin_axis);
    void loadJoyJsonConfig(const std::string &config_file);
    Eigen::VectorXd get_obs() { return networkInputData_; };
    void clip(Eigen::VectorXd &a, int num, double limit);
    void printRLparam();
    void activateRlController();
    void finalizeRlWbcHandoff();
    void captureSimObsOffsetFromState(const vector_t &rbd_state);
    void beginPureRlTorqueCrossfade();
    void beginRlSpawnSettle();
    void activateRlControllerFromStart(bool blend_from_wbc = false);
    std::map<std::string, int> joyButtonMap = {
        {"BUTTON_STANCE", 0},
        {"BUTTON_TROT", 1},
        {"BUTTON_RL", 2},
        {"BUTTON_WALK", 3},
        {"BUTTON_LB", 4},
        {"BUTTON_RB", 5},
        {"BUTTON_BACK", 6},
        {"BUTTON_START", 7}};
    std::map<std::string, int> joyAxisMap = {
        {"AXIS_LEFT_STICK_Y", 0},
        {"AXIS_LEFT_STICK_X", 1},
        {"AXIS_LEFT_LT", 2},
        {"AXIS_RIGHT_STICK_YAW", 3},
        {"AXIS_RIGHT_STICK_Z", 4},
        {"AXIS_RIGHT_RT", 5},
        {"AXIS_LEFT_RIGHT_TRIGGER", 6},
        {"AXIS_FORWARD_BACK_TRIGGER", 7}};
    sensor_msgs::Joy oldJoyMsg_;
    vector_t joystickOriginAxis_ = vector_t::Zero(6);
    vector_t joystickOriginAxisPre_ = vector_t::Zero(6);
    vector_t commadLineTarget_ = vector_t::Zero(6);
    double joystickSensitivity = 100;
    // param for RL
    Eigen::VectorXd defalutJointPos_;
    Eigen::VectorXd sim_pd_equilibrium_joint_;
    Eigen::VectorXd sim_joint_obs_offset_;
    bool sim_mujoco_equilibrium_{false};
    bool sim_handoff_obs_offset_{false};
    bool rl_wbc_torque_handoff_{false};
    bool rl_torque_crossfade_active_{false};
    ros::Time rl_wbc_handoff_end_time_;
    ros::Time rl_torque_crossfade_start_time_;
    double rlPdTargetJoint(int i) const
    {
      return defalutJointPos_[i];
    }
    Eigen::VectorXd JointControlMode_;
    Eigen::VectorXd JointPDMode_;
    Eigen::VectorXd initialState_;
    Eigen::VectorXd spawn_centroidal_status_;
    Eigen::VectorXd jointCmdFilterState_;
    Eigen::Vector3d accFilterState_;
    Eigen::Vector3d freeAccFilterState_;
    Eigen::Vector3d gyroFilterState_;
    Eigen::Vector4d command_;
    Eigen::Vector4d initalCommand_;
    Eigen::Vector4d commandScale_;
    Eigen::VectorXd jointTorqueCmd_;
    Eigen::VectorXd jointKp_;
    Eigen::VectorXd jointKd_;
    Eigen::VectorXd torqueLimits_;
    Eigen::VectorXd actionScaleTest_;
    Eigen::Vector4d velocityLimits_;
    // int ankleSolverType_ = 0;
    double actionScale_ = 0.25;
    int frameStack_ = 15;            // 多长时间步的obs
    int numSingleObs_ = 46;          // 单一时刻的obs的维度
    float cycleTime_ = 0.7;          // 周期时间为0.64s
    float phase_ = 0;                // 初始相位设置为0
    int episodeLength_ = 0;          // 用于计算相位，每走一个step就加1
    double clipObservations_ = 18.0; // 用于对输入进行clip
    double clipActions_ = 18.0;      // 用于对actions进行clip
    bool withArm_ = true;            // false: no arm, true: with arm
    double ruiwo_motor_velocities_factor_{0.0};
    std::string networkModelPath_;
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    std::unordered_map<std::string, double> scales_;                 // 存储obs的scale系数
    std::map<std::string, std::array<double, 3>> singleInputDataID_; // 存储singleInputData的id、min、max、scale
    std::vector<std::string> singleInputDataKeys;
    std::deque<Eigen::VectorXd> input_deque; // 用于多个历史的存储输入
    Eigen::VectorXd singleInputData;   // 单一时间的输入向量
    Eigen::VectorXd networkInputData_; // 将所有历史长度的数据接成一个向量
    Eigen::VectorXd commandPhase_;     // sin(phase), cos(phase)
    Eigen::VectorXd referenceJointPos_; // CSV reference joint pos relative to default (gym order)
    Eigen::VectorXd actions_;
    Eigen::VectorXd rl_takeover_seed_actions_;
    Eigen::VectorXd last_wbc_tau_;
    Eigen::VectorXd wbc_tau_at_rl_takeover_;
    double inferenceFrequency_;
    std::string danceCsvFile_;
    std::vector<Eigen::VectorXd> danceReferenceTrajectoryGym_;
    bool hasDanceReference_{false};
    std::mutex action_mtx_;
    std::mutex state_mtx_;
    std::mutex joint_cmd_mutex_;
    std::mutex joy_mutex_;
    std::mutex cmdvel_mutex_;
    std::thread inferenceThread_;
    Eigen::VectorXd joint_pos_limit; // 关节位置的限制
    Eigen::VectorXd joint_vel_limit; // 关节速度的限制
  protected:
    virtual void updateStateEstimation(const ros::Time &time, bool is_init = false);
    virtual void setupHumanoidInterface(const std::string &taskFile, const std::string &urdfFile, const std::string &referenceFile, const std::string &gaitFile,
                                        bool verbose, int robot_version_int);
    virtual void setupStateEstimate(const std::string &taskFile, bool verbose);
    void sensorsDataCallback(const kuavo_msgs::sensorsData::ConstPtr &msg);
    void real_init_wait();
    void loadSettings(const std::string &rlParamFile, bool verbose, double dt);
    void setRobotState(const vector_t &state)
    {
      std::lock_guard<std::mutex> lock(state_mtx_);
      robotState_ = state;
    };
    vector_t getRobotState()
    {
      std::lock_guard<std::mutex> lock(state_mtx_);
      return robotState_;
    };
    void setRobotSensorData(const SensorData &sensor_data)
    {
      std::lock_guard<std::mutex> lock(sensor_data_mutex_);
      robotSensorsData_ = sensor_data;
    };
    SensorData getRobotSensorData()
    {
      std::lock_guard<std::mutex> lock(sensor_data_mutex_);
      return robotSensorsData_;
    };
    void setJoyCmdState(const vector_t &joyCmdState)
    {
      std::lock_guard<std::mutex> lock(joy_mutex_);
      joystickOriginAxis_ = joyCmdState;
    };
    vector_t getJoyCmdState()
    {
      std::lock_guard<std::mutex> lock(joy_mutex_);
      return joystickOriginAxis_;
    };
    void setCommandData(const CommandData &CommandData)
    {
      std::lock_guard<std::mutex> lock(cmdvel_mutex_);
      CommandData_ = CommandData;
    };
    CommandData getCommandData()
    {
      std::lock_guard<std::mutex> lock(cmdvel_mutex_);
      return CommandData_;
    };
    ros::Time last_time_;
    ros::Time current_time_;
    std::queue<SensorData> sensorDataQueue;
    std::mutex sensor_data_mutex_;
    std::thread keyboardThread_;
    // Interface
    std::shared_ptr<HumanoidInterface> HumanoidInterface_;
    std::shared_ptr<HumanoidInterface> HumanoidInterface_mpc;
    std::shared_ptr<PinocchioEndEffectorKinematics> eeKinematicsPtr_;
    // State Estimation
    SensorData robotSensorsData_;
    SystemObservation currentObservation_, lastObservation_;
    vector_t robotState_;
    CommandData initialCommandData_;
    CommandData CommandData_;
    Eigen::VectorXd state_est_;
    std::shared_ptr<StateEstimateBase> stateEstimate_;
    std::shared_ptr<CentroidalModelRbdConversions> rbdConversions_;
    bool is_initialized_ = false;
    bool wbc_only_{false};
    bool rl_from_start_{false};
    bool sim_keep_wbc_with_rl_{true};
    bool sim_rl_debug_zero_action_{false};
    double sim_wbc_rl_action_blend_{0.45};
    double sim_wbc_rl_blend_start_{0.45};
    double sim_wbc_rl_blend_ramp_sec_{0.0};
    double sim_wbc_ref_filter_tau_{0.02};
    double sim_wbc_max_rl_joint_offset_rad_{0.35};
    Eigen::VectorXd sim_wbc_ref_filtered_;
    bool head_desire_initialized_{false};
    bool is_rl_controller_ = false;
    std::atomic<bool> rl_spawn_settling_{false};
    int rl_spawn_settle_step_{0};
    int rl_spawn_upright_count_{0};
    std::atomic<bool> rl_obs_reset_pending_{false};
    int rl_takeover_step_{0};
    int rl_torque_blend_step_{0};
    bool rl_action_ramp_logged_{false};
    bool Walkenable_ = false;
    bool contactTrotgait_ = false;
    bool cmdTrotgait_ = false;
    int hardware_status_ = 0;

    // preUpdate, 介于蹲姿启动和进MPC之间的状态处理
    bool isPreUpdateComplete{false};
    std::shared_ptr<WbcBase> standUpWbc_;

    // Whole Body Control
    std::shared_ptr<WbcBase> wbc_;
    // Visualization
    ros::Publisher observationPublisher_;
    // Controller Interface
    ros::Publisher targetTorquePub_;
    ros::Publisher jointCmdPub_;
    ros::Publisher targetPosPub_;
    ros::Publisher targetVelPub_;
    ros::Publisher targetKpPub_;
    ros::Publisher targetKdPub_;
    ros::Publisher RbdStatePub_;
    ros::Publisher wbcFrequencyPub_;
    ros::Publisher wbcTimeCostPub_;
    ros::Publisher feettargetTrajectoriesPublisher_;
    ros::Subscriber jointPosVelSub_;
    ros::Subscriber sensorsDataSub_;
    ros::Subscriber jointAccSub_;
    ros::Subscriber imuSub_;
    ros::Subscriber mpcStartSub_;
    ros::Subscriber gait_scheduler_sub_;
    ros::Subscriber joy_sub_;
    ros::Publisher mpcPolicyPublisher_;
    ros::Publisher stop_pub;
    PinocchioInterface *pinocchioInterface_ptr_;
    CentroidalModelInfo centroidalModelInfo_;
    CentroidalModelInfo centroidalModelInfoWBC_;

    // Node Handle
    ros::NodeHandle controllerNh_;
    HighlyDynamic::HumanoidInterfaceDrake *drake_interface_{nullptr};
    AnkleSolver ankleSolver;
#ifdef KUAVO_CONTROL_LIB_FOUND
    HighlyDynamic::JointFilter *joint_filter_ptr_{nullptr};
#endif
    HighlyDynamic::KuavoSettings kuavo_settings_;
    KuavoHardwareInterface *hardware_interface_ptr_{nullptr};
    bool is_nodelet_node_{false};
    ros::ServiceServer real_initial_start_service_;
    KuavoDataBuffer<SensorData> *sensors_data_buffer_ptr_;
    bool is_real_{false};
    bool is_mix_{false};
    bool is_cali_{false};
    char intial_input_cmd_ = '\0';
    double dt_ = 0.001;
    std::thread mpcThread_;
    std::atomic_bool controllerRunning_{};
    size_t jointNum_ = 12;
    size_t armNum_ = 0;
    size_t headNum_ = 2;
    size_t jointNumReal_ = 12;
    size_t armNumReal_ = 0;
    size_t actuatedDofNumReal_ = 12;
    size_t jointArmNum_ = 0;
    vector_t desire_arm_q_prev_;
    vector_t jointPos_, jointVel_;
    vector_t jointAcc_;
    vector_t jointCurrent_;
    bool init_input_ = false;
    Eigen::Quaternion<scalar_t> quat_;
    Eigen::Quaternion<scalar_t> quat_yaw_offset;
    vector_t desire_head_pos_ = vector_t::Zero(2);
    SensorData sensor_data_head_;
    contact_flag_t contactFlag_;
    vector3_t angularVel_, linearAccel_, freeLinearAccel_;
    matrix3_t orientationCovariance_, angularVelCovariance_, linearAccelCovariance_;
    size_t plannedMode_ = ModeNumber::SS;
    size_t rl_plannedMode_ = ModeNumber::SS;
    // vector_t defalutJointPos_;
    vector_t initial_status_;
    vector_t intail_input_;
    vector_t head_kp_, head_kd_;
    vector_t output_tau_, output_pos_, output_vel_;
    const std::string robotName_ = "humanoid";
    bool use_external_mpc_{true};
    bool use_joint_filter_{false};
    TopicLogger *ros_logger_{nullptr};
    vector_t optimizedState_mrt_, optimizedInput_mrt_;
    bool is_play_back_mode_ = false;
    int control_mode_ = 2; // 0：CST, 1: CSV, 2:CSP
    LowPassFilter2ndOrder accFilter_;
    LowPassFilter2ndOrder freeAccFilter_;
    LowPassFilter2ndOrder gyroFilter_;
    LowPassFilter2ndOrder jointCmdFilter_;
    LowPassFilter5thOrder joystickFilter_;
    // LowPassFilter2ndOrder joint_pos_filter_;
    // LowPassFilter2ndOrder joint_vel_filter_;
    // LowPassFilter2ndOrder joint_current_filter_;
    // LowPassFilter5thOrder joint_cmd_filter_5th_;
    bool is_swing_arm_ = false;
    double swing_arm_gain_{0.0}, swing_elbow_scale_{0.0};
    gaitTimeName current_gait_{"stance", 0.0}, last_gait_{"stance", 0.0};
    vector_t desire_arm_q, desire_arm_v;
  };
  class humanoidCheaterController : public humanoidController
  {
  protected:
    void setupStateEstimate(const std::string &taskFile, bool verbose) override;
  };
  class humanoidKuavoController : public humanoidController
  {
  protected:
    void setupStateEstimate(const std::string &taskFile, bool verbose) override;
  };
} // namespace humanoid_controller
