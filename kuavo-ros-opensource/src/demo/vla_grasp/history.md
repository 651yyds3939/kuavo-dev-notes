  935  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _contact_z_stop_above_m:=0.003   _left_cap_roll_extra_rad:=0.0
  936  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0
  937  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_y_m:=0.005   _left_tcp_extra_z_m:=-0.006   _contact_z_max_below_cap_m:=0.010   _contact_effort_threshold:=0.18
  938  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_y_m:=0.005   _left_tcp_extra_z_m:=-0.006   _contact_z_max_below_cap_m:=0.010 \
  939  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_y_m:=0.005   _left_tcp_extra_z_m:=-0.01   _contact_z_max_below_cap_m:=0.010   _contact_effort_threshold:=0.18
  940  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_y_m:=0.005   _left_tcp_extra_z_m:=-0.01   _contact_z_max_below_cap_m:=0.008   _contact_effort_threshold:=0.18
  941  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_y_m:=0.01   _left_tcp_extra_z_m:=-0.01   _contact_z_max_below_cap_m:=0.008   _contact_effort_threshold:=0.18
  942  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_tcp_extra_y_m:=0.005   _left_tcp_extra_z_m:=-0.006   _left_cap_roll_extra_rad:=0.0
  943  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0
  944  sudo su
  945  ssh leju_kuavo@192.168.26.12
  946  cd kuavo-ros-opensource/
  947  source devel/setup.bash 
  948  rosservice call /control_robot_leju_claw "data:
  949    name: ['left_claw', 'right_claw']
  950    position: [10.0, 10.0]
  951    velocity: [50.0, 50.0]
  952    effort: [0.3, 0.3]"
  953  python3 src/demo/vla_grasp/kuavo_state_publisher.py
  954  ssh leju_kuavo@192.168.26.12
  955  cd kuavo-ros-opensource && source devel/setup.bash
  956  roslaunch kuavo_arm_moveit_config move_group.launch
  957  cd kuavo-ros-opensource && source devel/setup.bash
  958  python3 src/demo/vla_grasp/look_down.py
  959  source devel/setup.bash 
  960  cd kuavo-ros-opensource && source devel/setup.bash
  961  python3 src/demo/vla_grasp/look_down.py
  962  source devel/setup.bash
  963  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_x_m:=-0.010   _left_tcp_extra_y_m:=0.0   _left_tcp_extra_z_m:=-0.050
  964  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_x_m:=-0.010   _left_tcp_extra_y_m:=0.0   _left_tcp_extra_z_m:=-0.04
  965  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_x_m:=-0.020   _left_tcp_extra_y_m:=0.005   _left_tcp_extra_z_m:=-0.030
  966  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_x_m:=-0.030   _left_tcp_extra_y_m:=0.000   _left_tcp_extra_z_m:=-0.030
  967  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=true   _left_claw_tip_enable:=false   _left_cap_roll_extra_rad:=0.0   _left_tcp_extra_x_m:=-0.040   _left_tcp_extra_y_m:=0.010   _left_tcp_extra_z_m:=-0.030
  968  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=truetouch 341.txt
  969  ssh leju_kuavo@192.168.26.12
  970  sudo su
  971  cd kuavo-ros-opensource
  972  source devel/setup.bash
  973  python3 src/demo/vla_grasp/look_down.py
  974  env | grep ANTHROPIC
  975  cd kuavo-ros-opensource && source devel/setup.bash
  976  source devel/setup.bash
  977  python3 src/demo/vla_grasp/look_down.py
  978  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false
  979  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.070   _left_tcp_extra_y_m:=0.025   _left_tcp_extra_z_m:=0.003
  980  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.070   _left_tcp_extra_y_m:=0.025   _left_tcp_extra_z_m:=0.003
  981  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.040   _left_tcp_extra_y_m:=0.025   _left_tcp_extra_z_m:=0.001
  982  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.035   _left_tcp_extra_y_m:=0.027   _left_tcp_extra_z_m:=0.001
  983  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.03   _left_tcp_extra_y_m:=0.027   _left_tcp_extra_z_m:=0.001
  984  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.025   _left_tcp_extra_y_m:=0.027   _left_tcp_extra_z_m:=0.001
  985  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.05   _left_tcp_extra_y_m:=0.027   _left_tcp_extra_z_m:=0.001
  986  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.05   _left_tcp_extra_y_m:=0.01   _left_tcp_extra_z_m:=0.005
  987  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.02   _left_tcp_extra_z_m:=0.003
  988  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.025   _left_tcp_extra_z_m:=0.0025
  989  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.022   _left_tcp_extra_z_m:=0.002
  990  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.02   _left_tcp_extra_z_m:=0.0023
  991  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.022   _left_tcp_extra_z_m:=0.0021
  992  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcp_extra_x_m:=0.0   _left_tcp_extra_y_m:=0.0   _left_tcp_extra_z_m:=0.0
  993  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcp_extra_x_m:=0.01   _left_tcp_extra_y_m:=0.0   _left_tcp_extra_z_m:=-0.005
  994  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcp_extra_x_m:=-0.05   _left_tcp_extra_y_m:=-0.03   _left_tcp_extra_z_m:=-0.005
  995  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcp_extra_x_m:=-0.05   _left_tcp_extra_y_m:=-0.06   _left_tcp_extra_z_m:=-0.001
  996  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcpcd kuavo-ros-opensource && source devel/setup.bash
  997  python3 src/demo/vla_grasp/look_down.py
  998  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcp_extra_x_m:=0.0   _left_tcp_extra_y_m:=0.0   _left_tcp_extra_z_m:=0.0   _left_claw_tip_ee_preclose_m:="[0.015, 0.0, 0.02]"   _left_claw_tip_ee_close_m:="[0.015, 0.0, 0.02]"
  999  source devel/setup.bash 
 1000  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcp_extra_x_m:=0.0   _left_tcp_extra_y_m:=0.0   _left_tcp_extra_z_m:=0.0   _left_claw_tip_ee_preclose_m:="[0.015, 0.0, 0.02]"   _left_claw_tip_ee_close_m:="[0.015, 0.0, 0.02]"
 1001  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcp_extra_x_m:=0.0   _left_tcp_extra_y_m:=0.0   _left_tcp_extra_z_m:=0.0   _left_claw_tip_ee_preclose_m:="[-0.015, 0.0, 0.02]"   _left_claw_tip_ee_close_m:="[-0.015, 0.0, 0.02]"
 1002  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.02   _left_claw_tip_world_z_close_m:=0.02   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.015   _left_tcp_extra_z_m:=0.0
 1003  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.02   _left_claw_tip_world_z_close_m:=0.02   _left_tcp_extra_x_m:=-0.063   _left_tcp_extra_y_m:=0.020   _left_tcp_extra_z_m:=0.0
 1004  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.02   _left_claw_tip_world_z_close_m:=0.02   _left_tcp_extra_x_m:=-0.068   _left_tcp_extra_y_m:=0.022   _left_tcp_extra_z_m:=0.0
 1005  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.02   _left_claw_tip_world_z_close_m:=0.02   _left_tcp_extra_x_m:=-0.072   _left_tcp_extra_y_m:=0.022   _left_tcp_extra_z_m:=0.0
 1006  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.02   _left_claw_tip_world_z_close_m:=0.02   _left_tcp_extra_x_m:=-0.072   _left_tcp_extra_y_m:=0.025   _left_tcp_extra_z_m:=0.0
 1007  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.02   _left_claw_tip_world_z_close_m:=0.02   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.020   _left_tcp_extra_z_m:=0.0
 1008  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.015   _left_claw_tip_world_z_close_m:=0.015   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.025   _left_tcp_extra_z_m:=0.0、
 1009  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.015   _left_claw_tip_world_z_close_m:=0.015   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.025   _left_tcp_extra_z_m:=0.0
 1010  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.01   _left_claw_tip_world_z_close_m:=0.015   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0
 1011  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.01   _left_claw_tip_world_z_close_m:=0.015   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0
 1012  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.01   _left_claw_tip_world_z_close_m:=0.015   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0
 1013  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.01   _left_claw_tip_world_z_close_m:=0.015   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0   _contact_z_max_below_cap_m:=0.010
 1014  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.005   _left_claw_tip_world_z_close_m:=0.010   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0   _contact_z_max_below_cap_m:=0.010
 1015  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.0065   _left_claw_tip_world_z_close_m:=0.010   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0   _contact_z_max_below_cap_m:=0.010
 1016  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.0055   _left_claw_tip_world_z_close_m:=0.010   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0   _contact_z_max_below_cap_m:=0.010
 1017  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=world_z   _left_claw_tip_world_z_m:=0.005   _left_claw_tip_world_z_close_m:=0.010   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0   _contact_z_max_below_cap_m:=0.010
 1018  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=false   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0   _cap_xy_refine_enable:=true   _cap_xy_refine_half_steps:=3   _cap_xy_refine_step_m:=0.004   _contact_z_max_below_cap_m:=0.010
 1019  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=false   _left_tcp_extra_x_m:=-0.06   _left_tcp_extra_y_m:=0.03   _left_tcp_extra_z_m:=0.0
 1020  python3 src/demo/vla_grasp/bimanual_unscrew.py   _left_cap_twist_cycles:=0   _right_hold_after_grasp:=false   _left_claw_tip_enable:=true   _left_claw_tip_mode:=ee   _left_tcp_extra_x_m:=0.0   _left_tcp_extra_y_m:=0.0   _left_tcp_extra_z_m:=0.0   _left_claw_tip_ee_preclose_m:="[0.0333, 0.0025, 0.0582]"   _left_claw_tip_ee_close_m:="[0.0333, 0.0025, 0.0582]"   _cap_xy_refine_enable:=false
