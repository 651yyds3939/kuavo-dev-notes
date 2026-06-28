/*
 * Maximum Force Threshold Testing System with Checkpoint Support
 * FIXED: Proper fall detection and NaN detection
 *
 * UPDATE 2025-06-09 (异步重构，不改动任何功能) :
 *   1) 改用 ros::AsyncSpinner(2) —— 回调线程永不被阻塞
 *   2) reset / 间隔等待改为 ros::WallDuration / WallTimer
 *   3) debug 打印改用 ros::WallTime，避免 /clock 归零后日志停滞
 *   4) resetRobot() 内复位 nan_detected_ / fall_detected_
 *   5) 订阅队列扩大到 100
 */

 #include <ros/ros.h>
 #include <geometry_msgs/Wrench.h>
 #include <nav_msgs/Odometry.h>
 #include <std_srvs/Empty.h>
 #include <kuavo_msgs/jointCmd.h>
 #include <eigen3/Eigen/Dense>
 #include <vector>
 #include <string>
 #include <fstream>
 #include <cmath>
 #include <algorithm>
 #include <csignal>
 #include <atomic>
 #include <json/json.h>
 
 // Global shutdown flag
 std::atomic<bool> g_shutdown_requested(false);
 
 void signalHandler(int sig)
 {
     ROS_WARN("Received signal %d, requesting shutdown...", sig);
     g_shutdown_requested = true;
    //  ros::shutdown();
 }
 
 class MaxForceThresholdTest
 {
 private:
     ros::NodeHandle   nh_;
     ros::Publisher    force_pub_;
     ros::Subscriber   odom_sub_;
     ros::Subscriber   joint_cmd_sub_;
     ros::ServiceClient reset_client_;
     ros::WallTimer    auto_start_timer_;           // ← 墙钟计时器
 
     struct TestDirection
     {
         std::string    name;
         Eigen::Vector3d vector;
         TestDirection(const std::string& n,double x,double y,double z):
             name(n),vector(x,y,z){vector.normalize();}
     };
 
     struct ForceType
     {
         std::string name;
         double      duration;
         std::string description;
         ForceType(const std::string& n,double d,const std::string& desc):
             name(n),duration(d),description(desc){}
     };
 
     struct ThresholdResult
     {
         std::string direction_name;
         std::string force_type;
         double      max_force;
         double      test_force;
         bool        robot_fell;
         double      test_duration;
         ThresholdResult():max_force(0),test_force(0),
                           robot_fell(false),test_duration(0){}
     };
 
     // ---------- Test parameters ----------
     std::vector<TestDirection>    directions_;
     std::vector<ForceType>        force_types_;
     std::vector<ThresholdResult>  results_;
 
     // ---------- Test state ----------
     bool              is_testing_          = false;
     bool              fall_detected_       = false;
     bool              nan_detected_        = false;
     ros::Time         test_start_time_;
     Eigen::Vector3d   baseline_position_;
     Eigen::Quaterniond baseline_orientation_;
     Eigen::Vector3d   current_position_;
     Eigen::Quaterniond current_orientation_;
     bool              baseline_established_= false;
 
     // ---------- Checkpoint ----------
     std::string checkpoint_file_            = "/tmp/force_test_checkpoint.json";
     int         current_direction_index_    = 0;
     int         current_force_type_index_   = 0;
     double      current_force_level_        = 0;
     bool        load_from_checkpoint_       = false;
 
     // ---------- Config (ROS params) ----------
     double force_increment_ = 10.0;
     double min_force_       = 50.0;
     double max_force_       = 3000.0;
     double fall_threshold_  = 0.4;   // m
     double tilt_threshold_  = 45.0;  // deg
 
 public:
     MaxForceThresholdTest()
     {
         ROS_INFO("=== Maximum Force Threshold Testing System with Checkpoint ===");
 
         // publishers / subscribers
         force_pub_ = nh_.advertise<geometry_msgs::Wrench>("/external_wrench",10);
         odom_sub_  = nh_.subscribe("/ground_truth/state",100,
                                    &MaxForceThresholdTest::odomCallback,this);
         joint_cmd_sub_=nh_.subscribe("/joint_cmd",100,
                                    &MaxForceThresholdTest::jointCmdCallback,this);
 
         reset_client_=nh_.serviceClient<std_srvs::Empty>("reset_simulation");
 
         initializeTestParameters();
         initializeDirections();
         initializeForceTypes();
 
         if(loadCheckpoint()){
             ROS_INFO("Loaded checkpoint – resuming");
             load_from_checkpoint_=true;
         }
 
         if(reset_client_.waitForExistence(ros::Duration(5.0)))
             ROS_INFO("Reset service available");
         else
             ROS_WARN("Reset service not available – tests may fail");
 
         establishBaseline();
         printTestMatrix();
 
         auto_start_timer_=nh_.createWallTimer(
             ros::WallDuration(3.0),
             [this](const ros::WallTimerEvent&)
             {
                 if(!is_testing_ && ros::ok() && !g_shutdown_requested)
                 {
                     if(load_from_checkpoint_){
                         ROS_INFO("\nRESUMING TESTS FROM CHECKPOINT");
                         resumeFromCheckpoint();
                     }else{
                         ROS_INFO("\nSTARTING FRESH TESTS");
                         runAllThresholdTests();
                     }
                 }
             },
             true /* oneshot */);
     }
 
     // ==========================  CALLBACKS  ==========================
     void jointCmdCallback(const kuavo_msgs::jointCmd::ConstPtr& msg)
     {
         if (nan_detected_ || g_shutdown_requested) return;
     
         for (size_t i = 0; i < msg->tau.size(); ++i) {
             double val = msg->tau[i];
             if (std::isnan(val) || std::isinf(val)) {
                 ROS_ERROR("CRITICAL: NaN/Inf detected in joint_cmd[%zu]=%f", i, val);
                 nan_detected_ = true;
                 publishForce(Eigen::Vector3d::Zero());
     
                 ThresholdResult r;
                 r.direction_name = directions_[current_direction_index_].name;
                 r.force_type     = force_types_[current_force_type_index_].name;
                 r.robot_fell     = true;
                 r.test_force     = current_force_level_;
                 r.max_force      = std::max(0.0, current_force_level_ - force_increment_);
                 r.test_duration  = force_types_[current_force_type_index_].duration;
     
                //  results_.push_back(r);
     
                 current_force_level_ = 0.0; 
     
                 saveCheckpoint();
     
                 ROS_ERROR("Checkpoint saved after NaN – this test will be skipped next run.");
                 g_shutdown_requested = true;
                 return;
             }
         }
     }
 
     void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
     {
         if(g_shutdown_requested) return;
 
         current_position_.x()=msg->pose.pose.position.x;
         current_position_.y()=msg->pose.pose.position.y;
         current_position_.z()=msg->pose.pose.position.z;
 
         current_orientation_.w()=msg->pose.pose.orientation.w;
         current_orientation_.x()=msg->pose.pose.orientation.x;
         current_orientation_.y()=msg->pose.pose.orientation.y;
         current_orientation_.z()=msg->pose.pose.orientation.z;
 
         if(is_testing_ && baseline_established_ && !fall_detected_)
         {
             static ros::WallTime last_dbg = ros::WallTime::now();
             if((ros::WallTime::now()-last_dbg).toSec()>1.0){
                 double drop=baseline_position_.z()-current_position_.z();
                 ROS_INFO("DEBUG: height %.3f, base %.3f, drop %.3f",
                          current_position_.z(),baseline_position_.z(),drop);
                 last_dbg=ros::WallTime::now();
             }
 
             double height_drop=baseline_position_.z()-current_position_.z();
             if(height_drop>fall_threshold_ || current_position_.z()<0.5){
                 ROS_WARN("Fall detected by height");
                 fall_detected_=true; return;
             }
             Eigen::Vector3d up=current_orientation_*Eigen::Vector3d(0,0,1);
             double tilt=acos(std::clamp(std::abs(up.z()),0.0,1.0))*180.0/M_PI;
             if(tilt>tilt_threshold_){
                 ROS_WARN("Fall detected by tilt %.1f°",tilt);
                 fall_detected_=true;
             }
         }
     }
 
     // ==========================  PARAM INIT  ==========================
     void initializeTestParameters()
     {
         nh_.param("force_increment",force_increment_,50.0);
         nh_.param("min_force",min_force_,50.0);
         nh_.param("max_force",max_force_,3000.0);
         nh_.param("fall_threshold",fall_threshold_,0.3);
         nh_.param("tilt_threshold",tilt_threshold_,45.0);
         ROS_INFO("Param: %.0f–%.0f N Δ%.0f | fall %.2f m / %.0f°",
                  min_force_,max_force_,force_increment_,fall_threshold_,tilt_threshold_);
     }
 
     void initializeDirections()
     {
         directions_={
             {"forward",     1,      0,      0},
             {"backward",   -1,      0,      0},
             {"left",        0,      1,      0},
             {"right",       0,     -1,      0},
             {"front_left",  0.707,  0.707,  0},
             {"front_right", 0.707, -0.707,  0},
             {"back_left",  -0.707,  0.707,  0},
             {"back_right", -0.707, -0.707,  0}
         };
     }
 
     void initializeForceTypes()
     {
         force_types_={
             {"impulse",0.2,"Short burst force"},
             {"step",   2.0,"Sustained constant force"},
             {"ramp",   3.0,"Gradually increasing force"}
         };
     }
 
     // ==========================  PRINT HELP  ==========================
     void printTestMatrix()
     {
         ROS_INFO("\n=== TEST MATRIX ===");
         ROS_INFO("Directions: %zu",directions_.size());
         for(size_t i=0;i<directions_.size();++i){
             const auto& d=directions_[i];
             std::string st=(i< current_direction_index_) ? " [DONE]" :
                             (i==current_direction_index_)?" [CURRENT]":" [PENDING]";
             ROS_INFO("  %zu. %s: [%.2f, %.2f, %.2f]%s",
                      i,d.name.c_str(),d.vector.x(),d.vector.y(),d.vector.z(),st.c_str());
         }
         ROS_INFO("Force types: %zu",force_types_.size());
         for(size_t i=0;i<force_types_.size();++i){
             const auto& f=force_types_[i];
             std::string st=(i< current_force_type_index_) ? " [DONE]" :
                             (i==current_force_type_index_)?" [CURRENT]":" [PENDING]";
             ROS_INFO("  %zu. %s (%.1fs): %s%s",
                      i,f.name.c_str(),f.duration,f.description.c_str(),st.c_str());
         }
     }
 
     // ==========================  BASELINE  ==========================
     void establishBaseline()
     {
         ROS_INFO("Establishing baseline pose");
         ros::Rate r(20);
         std::vector<double> heights;
         for(int i=0;i<50 && ros::ok() && !g_shutdown_requested;++i){
             r.sleep();
             if(current_position_.z()>0.1) heights.push_back(current_position_.z());
         }
         if(!heights.empty()){
             std::sort(heights.begin(),heights.end());
             baseline_position_=current_position_;
             baseline_position_.z()=heights[heights.size()/2];
             baseline_orientation_=current_orientation_;
         }else{
             baseline_position_=Eigen::Vector3d(0,0,0.87);
             baseline_orientation_.setIdentity();
             ROS_WARN("Using fallback baseline height 0.87 m");
         }
         baseline_established_=true;
     }
 
     // ==========================  CHECKPOINT IO  ==========================
     bool loadCheckpoint()
     {
         std::ifstream file(checkpoint_file_);
         if(!file.is_open()){
             ROS_INFO("No checkpoint file found – fresh start");
             return false;
         }
         Json::Value root; Json::Reader rd;
         if(!rd.parse(file,root)){
             ROS_ERROR("Failed to parse checkpoint");
             return false;
         }
         current_direction_index_=root.get("direction_index",0).asInt();
         current_force_type_index_=root.get("force_type_index",0).asInt();
         current_force_level_=root.get("force_level",0).asDouble();
 
         if(root.isMember("min_force")){
             min_force_=root["min_force"].asDouble();
             max_force_=root["max_force"].asDouble();
             force_increment_=root["force_increment"].asDouble();
             ROS_INFO("Restored params %.0f–%.0f N (step %.0f)",
                      min_force_,max_force_,force_increment_);
         }
 
         if(root.isMember("results")){
             const Json::Value& arr=root["results"];
             for(const auto& j: arr){
                 ThresholdResult r;
                 r.direction_name=j["direction_name"].asString();
                 r.force_type    =j["force_type"].asString();
                 r.max_force     =j["max_force"].asDouble();
                 r.test_force    =j["test_force"].asDouble();
                 r.robot_fell    =j["robot_fell"].asBool();
                 r.test_duration =j["test_duration"].asDouble();
                 results_.push_back(r);
             }
         }
 
         ROS_INFO("Checkpoint loaded: dir %d/%zu, type %d/%zu, force %.0f N",
                  current_direction_index_,directions_.size(),
                  current_force_type_index_,force_types_.size(),
                  current_force_level_);
         return true;
     }
 
     void saveCheckpoint()
     {
         Json::Value root;
         root["direction_index"]=current_direction_index_;
         root["force_type_index"]=current_force_type_index_;
         root["force_level"]=current_force_level_;
         root["timestamp"]=static_cast<int>(time(nullptr));
 
         root["min_force"]=min_force_;
         root["max_force"]=max_force_;
         root["force_increment"]=force_increment_;
 
         Json::Value arr(Json::arrayValue);
         for(const auto& r: results_){
            Json::Value j;
            j["direction_name"] = r.direction_name;
            j["force_type"]     = r.force_type;
            j["max_force"]      = (std::isfinite(r.max_force) ? r.max_force : 0.0);
            j["test_force"]     = (std::isfinite(r.test_force) ? r.test_force : 0.0);
            j["robot_fell"]     = r.robot_fell;
            j["test_duration"]  = r.test_duration;
            arr.append(j);
        }
         root["results"]=arr;
 
         std::ofstream f(checkpoint_file_);
         Json::StreamWriterBuilder b; std::unique_ptr<Json::StreamWriter> w(b.newStreamWriter());
         w->write(root,&f);
         ROS_INFO("Checkpoint saved to %s",checkpoint_file_.c_str());
     }
 
     // ==========================  MAIN TEST FLOW  ==========================
     void runAllThresholdTests()
     {
         results_.clear();
         current_direction_index_=0;
         current_force_type_index_=0;
         resumeFromCheckpoint();
     }
 
     void resumeFromCheckpoint()
     {
         for(size_t dir_idx=current_direction_index_;dir_idx<directions_.size();++dir_idx){
             size_t start_type=(dir_idx==current_direction_index_)?current_force_type_index_:0;
 
             for(size_t type_idx=start_type;type_idx<force_types_.size();++type_idx){
                 if(g_shutdown_requested||!ros::ok()) break;
 
                 current_direction_index_=dir_idx;
                 current_force_type_index_=type_idx;
 
                 const auto& direction=directions_[dir_idx];
                 const auto& ftype=force_types_[type_idx];
 
                 bool skip=false;
                 for(const auto& r: results_){
                     if(r.direction_name==direction.name && r.force_type==ftype.name &&
                        (r.max_force>0 || r.robot_fell)){
                         skip=true; break;
                     }
                 }
                 if(skip){ ROS_INFO("Skip completed test %s %s",direction.name.c_str(),ftype.name.c_str()); continue;}
 
                 ROS_INFO("\n=== TEST %zu/%zu ===",
                          dir_idx*force_types_.size()+type_idx+1,
                          directions_.size()*force_types_.size());
                 ROS_INFO("Direction: %s | Type: %s",
                          direction.name.c_str(), ftype.name.c_str());
 
                 ThresholdResult res=findMaxForce(direction,ftype);
                 results_.push_back(res);
                 saveCheckpoint();
 
                 if(res.robot_fell)
                     ROS_INFO("Max force %.0f N (fell at %.0f N)",res.max_force,res.test_force);
                 else
                     ROS_WARN("Reached max %.0f N without fall",res.max_force);
 
                 ros::WallDuration(3.0).sleep();
             }
         }
         analyzeResults();
         saveResults();
         ROS_INFO("All tests completed! CSV @ /tmp/max_force_thresholds.csv");
     }
 
     ThresholdResult findMaxForce(const TestDirection& direction,const ForceType& ftype)
     {
         ThresholdResult res;
         res.direction_name=direction.name;
         res.force_type=ftype.name;
         res.test_duration=ftype.duration;
 
         double start_force=min_force_;
         if(current_force_level_>0){
             start_force=std::max(min_force_,current_force_level_-force_increment_);
             ROS_INFO("Resuming from %.0f N (prev %.0f N)",start_force,current_force_level_);
         }
 
         double cur=start_force;
         bool survive=true;
 
         while(cur<=max_force_ && survive && ros::ok() && !g_shutdown_requested){
             current_force_level_=cur;
             ROS_INFO("Testing %.0f N %s ",cur,direction.name.c_str());
 
             if(!resetRobot()){ ROS_ERROR("Reset failed"); break; }
             if(nan_detected_){ ROS_ERROR("NaN before test, abort"); saveCheckpoint(); break; }
 
             bool fell=testSingleForce(direction.vector,cur,ftype);
             if(fell || nan_detected_){
                 res.robot_fell=true;
                 res.test_force=cur;
                 res.max_force=std::max(0.0,cur-force_increment_);
                 survive=false;
             }else{
                 res.max_force=cur;
                 ROS_INFO("Survived %.0f N",cur);
                 cur+=force_increment_;
             }
         }
         if(survive) ROS_WARN("Survived up to %.0f N",max_force_);
         current_force_level_=0;
         return res;
     }
 
     bool testSingleForce(const Eigen::Vector3d& dir,double magnitude,const ForceType& ftype)
     {
         is_testing_=true;
         fall_detected_=false;
         test_start_time_=ros::Time::now();
 
         Eigen::Vector3d force_vec=dir*magnitude;
         ros::Rate rate(50);
         double elapsed=0;
 
         while(elapsed<ftype.duration+1.0 && ros::ok() && !g_shutdown_requested && !nan_detected_){
             elapsed=(ros::Time::now()-test_start_time_).toSec();
             Eigen::Vector3d cur_force=calculateForce(elapsed,force_vec,ftype);
             publishForce(cur_force);
 
             if(fall_detected_){
                 publishForce(Eigen::Vector3d::Zero());
                 is_testing_=false;
                 return true;
             }
             ros::spinOnce(); rate.sleep();
         }
         publishForce(Eigen::Vector3d::Zero());
         ros::WallDuration(1.0).sleep();
 
         is_testing_=false;
         return fall_detected_||nan_detected_;
     }
 
     Eigen::Vector3d calculateForce(double t,const Eigen::Vector3d& maxF,const ForceType& ftype)
     {
         double factor=0;
         if(ftype.name=="impulse"){
             factor=(t>=0.1 && t<0.1+ftype.duration)?1.0:0.0;
         }else if(ftype.name=="step"){
             factor=(t>=0.1 && t<0.1+ftype.duration)?1.0:0.0;
         }else if(ftype.name=="ramp"){
             if(t>=0.1 && t<1.1) factor=(t-0.1)/1.0;
             else if(t<0.1+ftype.duration) factor=1.0;
         }
         return factor*maxF;
     }
 
     void publishForce(const Eigen::Vector3d& f)
     {
         geometry_msgs::Wrench w;
         w.force.x=f.x(); w.force.y=f.y(); w.force.z=f.z();
         force_pub_.publish(w);
     }
 
     bool resetRobot()
     {
         ROS_INFO("  Resetting robot");
         publishForce(Eigen::Vector3d::Zero());
 
         std_srvs::Empty srv;
         if(!reset_client_.call(srv)){
             ROS_ERROR("Reset service call failed");
             return false;
         }
         ros::WallDuration(2.0).sleep();
         nan_detected_=false;
         fall_detected_=false;
         establishBaseline();
         return true;
     }
 
     void analyzeResults()
     {
         ROS_INFO("\n=== THRESHOLD ANALYSIS ===");
         printf("\n%-15s %-10s %-12s %-8s %-10s\n",
                "Direction","Type","Max Force(N)","Fell?","Test Force");
         printf("---------------------------------------------------------------\n");
         for(const auto& r: results_){
             printf("%-15s %-10s %-12.0f %-8s %-10.0f\n",
                    r.direction_name.c_str(),
                    r.force_type.c_str(),
                    r.max_force,
                    r.robot_fell?"YES":"NO",
                    r.test_force);
         }
         if(!results_.empty()){
             auto hi=std::max_element(results_.begin(),results_.end(),
                 [](auto& a,auto& b){return a.max_force<b.max_force;});
             auto lo=std::min_element(results_.begin(),results_.end(),
                 [](auto& a,auto& b){return a.max_force<b.max_force;});
             ROS_INFO("\nSTRONGEST: %s %s - %.0f N",
                      hi->direction_name.c_str(),hi->force_type.c_str(),hi->max_force);
             ROS_INFO("WEAKEST:  %s %s - %.0f N",
                      lo->direction_name.c_str(),lo->force_type.c_str(),lo->max_force);
         }
     }
 
     void saveResults()
     {
         std::ofstream f("/tmp/max_force_thresholds.csv");
         f<<"direction,force_type,max_force_N,robot_fell,test_force_N,test_duration_s\n";
         for(const auto& r: results_){
             f<<r.direction_name<<","<<r.force_type<<","<<r.max_force<<","
              <<(r.robot_fell?"true":"false")<<","<<r.test_force<<","<<r.test_duration<<"\n";
         }
         f.close();
     }
 };
 
 // ==========================  MAIN  ==========================
 int main(int argc,char** argv)
 {
     ros::init(argc,argv,"max_force_threshold_test");
 
     signal(SIGINT,signalHandler);
     signal(SIGTERM,signalHandler);
 
     ros::AsyncSpinner spinner(2);  // 1 for callbacks, 1 for timers / services
     spinner.start();
 
     try{
         MaxForceThresholdTest test;
         ros::waitForShutdown();
     }catch(const std::exception& e){
         ROS_ERROR("Exception: %s",e.what());
     }
     ROS_INFO("Threshold test system shutting down…");
     return 0;
 }
 