#include <pinocchio/fwd.hpp>

#include <ros/ros.h>
// 【修改】引入官方自定义的传感器数据结构
#include <kuavo_msgs/sensorsData.h>
#include <std_msgs/Float64MultiArray.h>

#include <iostream>
#include <string>
#include <map>
#include <vector>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <Eigen/Core>

class GravityCompensationNode {
private:
    ros::NodeHandle nh_;
    ros::Subscriber sensor_data_sub_;
    ros::Publisher effort_pub_;

    pinocchio::Model model_;
    pinocchio::Data data_;
    
    std::map<std::string, int> joint_name_to_pinocchio_idx_;
    std::vector<std::string> target_arm_joints_;
    
    // 【新增】建立底层硬件数组索引(12~25) 到 URDF 关节名字的绝对映射
    std::vector<std::pair<int, std::string>> hardware_idx_to_joint_name_;

    Eigen::VectorXd q_; 

public:
    GravityCompensationNode() {
        // 模型路径（Docker内的绝对路径）
        std::string urdf_filename = "/root/kuavo_ws/src/kuavo_assets/models/biped_s49/urdf/biped_s49_gazebo.urdf";
        pinocchio::urdf::buildModel(urdf_filename, model_);
        data_ = pinocchio::Data(model_);

        for (pinocchio::JointIndex joint_id = 1; joint_id < (pinocchio::JointIndex)model_.njoints; ++joint_id) {
            std::string joint_name = model_.names[joint_id];
            int idx_q = model_.joints[joint_id].idx_q();
            joint_name_to_pinocchio_idx_[joint_name] = idx_q;
        }

        // 下发的力矩顺序（14个关节）
        target_arm_joints_ = {
            "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
            "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint"
        };

        // 【新增】根据官方文档，左臂 12~18，右臂 19~25
        hardware_idx_to_joint_name_ = {
            {12, "zarm_l1_joint"}, {13, "zarm_l2_joint"}, {14, "zarm_l3_joint"}, {15, "zarm_l4_joint"}, {16, "zarm_l5_joint"}, {17, "zarm_l6_joint"}, {18, "zarm_l7_joint"},
            {19, "zarm_r1_joint"}, {20, "zarm_r2_joint"}, {21, "zarm_r3_joint"}, {22, "zarm_r4_joint"}, {23, "zarm_r5_joint"}, {24, "zarm_r6_joint"}, {25, "zarm_r7_joint"}
        };

        q_ = Eigen::VectorXd::Zero(model_.nq);

        // 【修改】订阅目标变更为底层高速通道
        sensor_data_sub_ = nh_.subscribe("/sensors_data_raw", 10, &GravityCompensationNode::sensorsDataCallback, this);
        effort_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/kuavo/arm_effort_command", 10);

        ROS_INFO("====================================================================");
        ROS_INFO("Kuavo Arm Gravity Compensation Node: FULL LOOP CLOSED & LISTENING TO SENSORS.");
        ROS_INFO("====================================================================");
    }

    // 【修改】回调函数接收的类型变为 kuavo_msgs::sensorsData
    void sensorsDataCallback(const kuavo_msgs::sensorsData::ConstPtr& msg) {
        // 安全拦截：防止数组越界导致节点崩溃
        if (msg->joint_data.joint_q.size() < 26) {
            ROS_WARN_THROTTLE(1.0, "[Warning] joint_q array is too small: %zu", msg->joint_data.joint_q.size());
            return;
        }

        // 心跳信号，证明彻底通了
        ROS_INFO_THROTTLE(1.0, "💖 [Heartbeat] Receiving /sensors_data_raw! L1 Angle: %.3f rad", msg->joint_data.joint_q[12]);

        // 直接通过硬件数组下标（12~25），把弧度值拍进 Pinocchio 的解算坐标 q_ 里
        for (const auto& pair : hardware_idx_to_joint_name_) {
            int hw_idx = pair.first;
            std::string j_name = pair.second;
            
            if (joint_name_to_pinocchio_idx_.find(j_name) != joint_name_to_pinocchio_idx_.end()) {
                int pinocchio_q_idx = joint_name_to_pinocchio_idx_[j_name];
                q_[pinocchio_q_idx] = msg->joint_data.joint_q[hw_idx];
            }
        }

        // 动力学解算 G(q)
        pinocchio::computeGeneralizedGravity(model_, data_, q_);

        // 打包 14 个关节的重力矩发布给 MuJoCo
        std_msgs::Float64MultiArray effort_msg;
        effort_msg.data.resize(target_arm_joints_.size());

        for (size_t i = 0; i < target_arm_joints_.size(); ++i) {
            std::string name = target_arm_joints_[i];
            int q_idx = joint_name_to_pinocchio_idx_[name];
            effort_msg.data[i] = data_.g[q_idx];
        }

        effort_pub_.publish(effort_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gravity_compensation_node");
    GravityCompensationNode node;
    ros::spin();
    return 0;
}