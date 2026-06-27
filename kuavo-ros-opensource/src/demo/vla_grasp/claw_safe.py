#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LejuClaw 安全控制：限幅 position/effort，闭合时监测堵转并自动中止。

API 约定（与 debug 一致）：0=张开，100=闭合。
"""

import time
from typing import List, Optional, Tuple

import rospy
from kuavo_msgs.msg import lejuClawState
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

# 默认保守参数（可在 GraspSkills 初始化后通过 ROS param 覆盖）
CLAW_OPEN = 10.0
CLAW_CLOSE_LEFT = 80.0
CLAW_CLOSE_RIGHT = 85.0
CLAW_VEL = [50, 50]
CLAW_EFFORT_OPEN = [0.5, 0.5]
CLAW_EFFORT_CLOSE_LEFT = 0.5
CLAW_EFFORT_CLOSE_RIGHT = 0.6

MAX_CLOSE_POSITION = 85.0
MAX_OPEN_EFFORT = 1.0
MAX_CLOSE_EFFORT = 0.6
STALL_EFFORT_THRESHOLD = 1.0
CLAW_STATE_WAIT_SEC = 1.5
STALL_POLL_INTERVAL_SEC = 0.15

SIDE_INDEX = {"left": 0, "right": 1}


def load_params():
    """在 rospy.init_node 之后调用，从私有/param 命名空间读取可调参数。"""
    global CLAW_OPEN, CLAW_CLOSE_LEFT, CLAW_CLOSE_RIGHT
    global CLAW_EFFORT_OPEN, CLAW_EFFORT_CLOSE_LEFT, CLAW_EFFORT_CLOSE_RIGHT
    global MAX_CLOSE_POSITION, MAX_CLOSE_EFFORT, STALL_EFFORT_THRESHOLD, CLAW_STATE_WAIT_SEC

    prefix = "~claw_"
    CLAW_OPEN = float(rospy.get_param(prefix + "open", CLAW_OPEN))
    CLAW_CLOSE_LEFT = float(rospy.get_param(prefix + "close_left", CLAW_CLOSE_LEFT))
    CLAW_CLOSE_RIGHT = float(rospy.get_param(prefix + "close_right", CLAW_CLOSE_RIGHT))
    MAX_CLOSE_POSITION = float(rospy.get_param(prefix + "max_close_pos", MAX_CLOSE_POSITION))
    MAX_CLOSE_EFFORT = float(rospy.get_param(prefix + "max_close_effort", MAX_CLOSE_EFFORT))
    STALL_EFFORT_THRESHOLD = float(rospy.get_param(prefix + "stall_effort", STALL_EFFORT_THRESHOLD))
    CLAW_STATE_WAIT_SEC = float(rospy.get_param(prefix + "wait_sec", CLAW_STATE_WAIT_SEC))
    open_eff = float(rospy.get_param(prefix + "effort_open", CLAW_EFFORT_OPEN[0]))
    CLAW_EFFORT_OPEN = [open_eff, open_eff]
    CLAW_EFFORT_CLOSE_LEFT = float(rospy.get_param(prefix + "effort_close_left", CLAW_EFFORT_CLOSE_LEFT))
    CLAW_EFFORT_CLOSE_RIGHT = float(rospy.get_param(prefix + "effort_close_right", CLAW_EFFORT_CLOSE_RIGHT))


def _is_close_cmd(pos: float) -> bool:
    return pos > 50.0


def sanitize_position(pos: float) -> float:
    pos = max(0.0, min(100.0, float(pos)))
    if _is_close_cmd(pos):
        return min(pos, MAX_CLOSE_POSITION)
    return pos


def sanitize_effort(pos: float, effort: float) -> float:
    if _is_close_cmd(pos):
        return min(float(effort), MAX_CLOSE_EFFORT)
    return min(float(effort), MAX_OPEN_EFFORT)


def sanitize_cmd(
    pos: List[float], vel: List[float], effort: List[float]
) -> Tuple[List[float], List[float], List[float]]:
    pos = [sanitize_position(p) for p in pos]
    effort = [sanitize_effort(p, e) for p, e in zip(pos, effort)]
    return pos, list(vel), effort


def build_open_cmd() -> Tuple[List[float], List[float], List[float]]:
    return [CLAW_OPEN, CLAW_OPEN], list(CLAW_VEL), list(CLAW_EFFORT_OPEN)


def build_close_cmd(is_left_arm: bool) -> Tuple[List[float], List[float], List[float]]:
    if is_left_arm:
        return (
            [CLAW_CLOSE_LEFT, CLAW_OPEN],
            list(CLAW_VEL),
            [CLAW_EFFORT_CLOSE_LEFT, CLAW_EFFORT_OPEN[1]],
        )
    return (
        [CLAW_OPEN, CLAW_CLOSE_RIGHT],
        list(CLAW_VEL),
        [CLAW_EFFORT_OPEN[0], CLAW_EFFORT_CLOSE_RIGHT],
    )


def format_claw_state(state_msg: Optional[lejuClawState]) -> str:
    if state_msg is None:
        return "state=unknown"
    return "state={} pos={} effort={}".format(
        list(state_msg.state),
        list(state_msg.data.position),
        list(state_msg.data.effort),
    )


class SafeClawController:
    """带限幅与堵转监测的夹爪控制。"""

    def __init__(self, load_ros_params: bool = True):
        if load_ros_params:
            load_params()
        self._last_state: Optional[lejuClawState] = None
        rospy.Subscriber("/leju_claw_state", lejuClawState, self._on_state, queue_size=1)
        time.sleep(0.3)

    def _on_state(self, msg: lejuClawState):
        self._last_state = msg

    @property
    def last_state(self) -> Optional[lejuClawState]:
        return self._last_state

    def _detect_stall(self, pos_cmd: List[float], start_state: Optional[lejuClawState]) -> Optional[int]:
        """若闭合指令导致某侧 effort 超阈值，返回 side index。"""
        if self._last_state is None:
            return None
        efforts = list(self._last_state.data.effort)
        for idx, cmd in enumerate(pos_cmd):
            if not _is_close_cmd(cmd):
                continue
            if idx >= len(efforts):
                continue
            if efforts[idx] >= STALL_EFFORT_THRESHOLD:
                return idx
        return None

    def _position_moved(
        self, side_idx: int, start_state: Optional[lejuClawState], min_delta: float = 0.5
    ) -> bool:
        if start_state is None or self._last_state is None:
            return True
        before = list(start_state.data.position)[side_idx]
        after = list(self._last_state.data.position)[side_idx]
        return abs(after - before) >= min_delta

    def _send_claw_service(
        self,
        pos: List[float],
        vel: List[float],
        effort: List[float],
        tag: str = "cmd",
        sanitize: bool = True,
    ) -> Tuple[bool, str]:
        """仅发 ROS 服务，不等待运动完成。返回 (success, error_message)。"""
        if sanitize:
            pos, vel, effort = sanitize_cmd(pos, vel, effort)
        else:
            pos = [max(0.0, min(100.0, float(p))) for p in pos]
            vel = list(vel)
            effort = [float(e) for e in effort]

        rospy.loginfo("claw before %s -> %s", tag, format_claw_state(self._last_state))
        try:
            req = controlLejuClawRequest()
            req.data.name = ["left_claw", "right_claw"]
            req.data.position = pos
            req.data.velocity = vel
            req.data.effort = effort
            rospy.wait_for_service("/control_robot_leju_claw", timeout=3.0)
            res = rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)(req)
            if not res.success:
                msg = res.message or "(empty rejection)"
                rospy.logwarn("claw %s rejected: %s", tag, msg)
                return False, msg
            return True, ""
        except Exception as exc:
            rospy.logwarn("claw %s service failed: %s", tag, exc)
            return False, str(exc)

    def _wait_side_settled(
        self,
        side_idx: int,
        timeout: float = 5.0,
        require_saw_moving: bool = False,
    ) -> bool:
        """等待该侧 state 变为 Reached(2)。require_saw_moving 防刚下发指令时读到旧 Reached。"""
        deadline = time.time() + timeout
        saw_moving = not require_saw_moving
        while time.time() < deadline and not rospy.is_shutdown():
            if self._last_state is None:
                time.sleep(STALL_POLL_INTERVAL_SEC)
                continue
            st = int(self._last_state.state[side_idx])
            if st == 1:
                saw_moving = True
            if st == -1:
                return False
            if st == 2 and saw_moving:
                return True
            time.sleep(STALL_POLL_INTERVAL_SEC)
        return False

    def _wait_side_target_pos(
        self,
        side_idx: int,
        timeout: float,
        target_pos: float,
        pos_tolerance: float = 8.0,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        等 motion 周期完成(先 Moving 再 Reached)，且 pos 接近 target。
        用于 open/close 探针，避免运动中误判。
        """
        deadline = time.time() + timeout
        saw_moving = False
        while time.time() < deadline and not rospy.is_shutdown():
            if self._last_state is None:
                time.sleep(STALL_POLL_INTERVAL_SEC)
                continue
            st = int(self._last_state.state[side_idx])
            pos, eff = self._side_metrics(side_idx)
            if st == 1:
                saw_moving = True
            if st == -1:
                return False, pos, eff
            if saw_moving and st == 2 and pos is not None:
                if abs(pos - target_pos) <= pos_tolerance:
                    return True, pos, eff
            time.sleep(STALL_POLL_INTERVAL_SEC)
        pos, eff = self._side_metrics(side_idx)
        if pos is not None and abs(pos - target_pos) <= pos_tolerance:
            return True, pos, eff
        return False, pos, eff

    def _probe_reopen(
        self,
        side_idx: int,
        open_pos: float,
        open_effort: float,
        pause: float,
        tag: str,
        open_success_pos: float,
        max_attempts: int = 3,
    ) -> Tuple[bool, bool, Optional[float], Optional[float], bool]:
        """
        闭合后 reopen，带 settle 等待与服务重试。
        返回 (reopen_ok, service_ever_ok, pos, effort, motion_settled)。
        """
        self._wait_side_settled(side_idx, timeout=pause + 3.0, require_saw_moving=True)
        time.sleep(0.3)

        pos_open = None
        eff_open = None
        service_ever_ok = False
        last_settled = False

        for attempt in range(max_attempts):
            pos_cmd, vel, eff_cmd = self._build_side_cmd(side_idx, open_pos, open_effort)
            attempt_tag = "%s#%d" % (tag, attempt + 1)
            ok, err = self._send_claw_service(pos_cmd, vel, eff_cmd, tag=attempt_tag, sanitize=False)
            if not ok:
                rospy.logwarn("reopen attempt %d/%d failed: %s", attempt + 1, max_attempts, err)
                time.sleep(1.0)
                continue

            service_ever_ok = True
            settled, pos_open, eff_open = self._wait_side_target_pos(
                side_idx,
                timeout=pause + 4.0,
                target_pos=open_pos,
                pos_tolerance=12.0,
            )
            last_settled = settled
            rospy.loginfo("claw after %s -> %s", attempt_tag, format_claw_state(self._last_state))

            if pos_open is not None and pos_open <= open_success_pos:
                return True, True, pos_open, eff_open, True

            if settled and pos_open is not None and pos_open > open_success_pos:
                return False, True, pos_open, eff_open, True

        return False, service_ever_ok, pos_open, eff_open, last_settled

    def call(
        self,
        pos: List[float],
        vel: List[float],
        effort: List[float],
        tag: str = "cmd",
        abort_on_stall: bool = True,
        auto_reopen_on_stall: bool = True,
        sanitize: bool = True,
        wait_sec: Optional[float] = None,
    ) -> bool:
        raw_pos = list(pos)
        raw_effort = list(effort)
        if sanitize:
            pos, vel, effort = sanitize_cmd(pos, vel, effort)
            if pos != raw_pos or effort != raw_effort:
                rospy.loginfo(
                    "claw %s sanitized pos %s -> %s effort -> %s",
                    tag,
                    raw_pos,
                    pos,
                    effort,
                )
        else:
            pos = [max(0.0, min(100.0, float(p))) for p in pos]
            vel = list(vel)
            effort = [float(e) for e in effort]

        start_state = self._last_state
        ok, err = self._send_claw_service(pos, vel, effort, tag=tag, sanitize=False)
        if not ok:
            return False

        settle = CLAW_STATE_WAIT_SEC if wait_sec is None else wait_sec
        deadline = time.time() + settle
        stalled_side = None
        while time.time() < deadline:
            time.sleep(STALL_POLL_INTERVAL_SEC)
            if not abort_on_stall:
                continue
            stalled_side = self._detect_stall(pos, start_state)
            if stalled_side is not None:
                side_name = "left" if stalled_side == 0 else "right"
                rospy.logwarn(
                    "claw %s STALL on %s (effort>=%.2f), aborting close",
                    tag,
                    side_name,
                    STALL_EFFORT_THRESHOLD,
                )
                break

        rospy.loginfo("claw after %s -> %s", tag, format_claw_state(self._last_state))

        if stalled_side is not None and auto_reopen_on_stall:
            reopen_pos = list(pos)
            reopen_effort = list(effort)
            reopen_pos[stalled_side] = CLAW_OPEN
            reopen_effort[stalled_side] = CLAW_EFFORT_OPEN[stalled_side]
            rospy.logwarn("claw auto-reopen side %s after stall", stalled_side)
            self.call(
                reopen_pos,
                vel,
                reopen_effort,
                tag="auto-reopen",
                abort_on_stall=False,
                auto_reopen_on_stall=False,
            )
            return False

        return stalled_side is None

    def _side_metrics(self, side_idx: int):
        if self._last_state is None:
            return None, None
        pos = list(self._last_state.data.position)[side_idx]
        eff = list(self._last_state.data.effort)[side_idx]
        return pos, eff

    def _build_side_cmd(self, side_idx: int, target_pos: float, target_effort: float):
        pos = [CLAW_OPEN, CLAW_OPEN]
        effort = list(CLAW_EFFORT_OPEN)
        pos[side_idx] = target_pos
        effort[side_idx] = target_effort
        return pos, list(CLAW_VEL), effort

    def run_stall_limit_probe(
        self,
        side: str,
        close_pos: float = 100.0,
        effort_start: float = 0.5,
        effort_step: float = 0.1,
        effort_max: float = 1.5,
        open_pos: float = 10.0,
        open_effort: float = 0.8,
        pause: float = 2.0,
        open_success_pos: float = 25.0,
    ):
        """
        危险诊断：完全闭合(close_pos)后 reopen，逐步增大闭合电流找自锁临界点。
        返回 dict: last_safe_effort, lock_effort, results[]
        """
        idx = SIDE_INDEX[side]
        close_pos = max(50.0, min(100.0, float(close_pos)))
        efforts = []
        e = effort_start
        while e <= effort_max + 1e-6:
            efforts.append(round(e, 3))
            e += effort_step

        print("\n===== STALL LIMIT PROBE (%s) — DANGEROUS =====" % side)
        print("close_pos=%.0f  effort %.2f→%.2f step %.2f  reopen pos=%.0f effort=%.2f"
              % (close_pos, effort_start, effort_max, effort_step, open_pos, open_effort))
        print("Ensure 退轴 + 0.5-1cm backlash before starting.\n")

        pos, vel, effort = self._build_side_cmd(idx, open_pos, open_effort)
        self.call(pos, vel, effort, tag="probe-init-open", sanitize=False, abort_on_stall=False)
        time.sleep(pause)
        print("  init open -> %s" % format_claw_state(self._last_state))

        last_safe = None
        lock_effort = None
        rows = []

        for close_eff in efforts:
            pos, vel, effort = self._build_side_cmd(idx, close_pos, close_eff)
            ok, err = self._send_claw_service(
                pos, vel, effort, tag="close-%.2f" % close_eff, sanitize=False
            )
            if not ok:
                print("  close@%.2fA SERVICE FAIL: %s — skip this step" % (close_eff, err))
                continue

            settled, pos_close, eff_close = self._wait_side_target_pos(
                idx,
                timeout=pause + 4.0,
                target_pos=close_pos,
                pos_tolerance=12.0,
            )
            if not settled:
                self._wait_side_settled(idx, timeout=2.0, require_saw_moving=True)
                pos_close, eff_close = self._side_metrics(idx)
            rospy.loginfo("claw after close-%.2f -> %s", close_eff, format_claw_state(self._last_state))

            reopen_ok, service_ok, pos_open, eff_open, reopen_settled = self._probe_reopen(
                idx,
                open_pos,
                open_effort,
                pause,
                tag="reopen-after-%.2f" % close_eff,
                open_success_pos=open_success_pos,
            )

            stall_on_close = eff_close is not None and eff_close >= STALL_EFFORT_THRESHOLD
            row = {
                "close_effort": close_eff,
                "pos_after_close": pos_close,
                "eff_after_close": eff_close,
                "pos_after_reopen": pos_open,
                "eff_after_reopen": eff_open,
                "reopen_ok": reopen_ok,
                "stall_on_close": stall_on_close,
                "reopen_service_ok": service_ok,
                "reopen_settled": reopen_settled,
            }
            rows.append(row)

            if reopen_ok and not stall_on_close:
                status = "OK"
            elif not service_ok:
                status = "SERVICE_FAIL"
            elif not reopen_settled:
                status = "TIMEOUT"
            else:
                status = "LOCK"

            print(
                "  close@%.2fA -> pos=%s eff=%s | reopen -> pos=%s | %s"
                % (
                    close_eff,
                    "%.1f" % pos_close if pos_close is not None else "?",
                    "%.3f" % eff_close if eff_close is not None else "?",
                    "%.1f" % pos_open if pos_open is not None else "?",
                    status,
                )
            )

            if reopen_ok and not stall_on_close:
                last_safe = close_eff
                continue

            if not service_ok:
                print("    -> reopen 服务未成功，跳过本档继续")
                continue

            if not reopen_settled:
                print("    -> reopen 超时(可能仍在运动中)，跳过本档继续")
                continue

            if not reopen_ok:
                lock_effort = close_eff
                print("\n*** MECHANICAL LOCK near close_effort=%.2fA ***" % close_eff)
                if stall_on_close:
                    print("    close phase: effort>=%.2f (stall)" % STALL_EFFORT_THRESHOLD)
                print("    reopen 服务成功但 pos=%.1f (need <=%.1f)" % (pos_open or -1, open_success_pos))
                break

        if lock_effort is None:
            safe_txt = "%.2fA" % last_safe if last_safe is not None else "N/A"
            print("\nProbe finished without mechanical lock up to %.2fA — last_safe=%s"
                  % (effort_max, safe_txt))
        else:
            safe_txt = "%.2fA" % last_safe if last_safe is not None else "none"
            print("\nRecommendation: grasp close_effort <= %s" % safe_txt)
            print("Attempting emergency reopen with effort=%.2f ..." % open_effort)
            self._probe_reopen(
                idx,
                open_pos,
                open_effort,
                pause,
                tag="emergency-reopen",
                open_success_pos=open_success_pos,
            )
            pos_final, _ = self._side_metrics(idx)
            if pos_final is not None and pos_final > open_success_pos:
                print("*** EMERGENCY REOPEN FAILED pos=%.1f — STOP, 退轴 manually ***" % pos_final)
            else:
                print("Emergency reopen -> %s" % format_claw_state(self._last_state))

        return {
            "last_safe_effort": last_safe,
            "lock_effort": lock_effort,
            "results": rows,
        }

    def verify_side_responsive(self, side: str, pause: float = 1.5) -> bool:
        """发一次 open，检查该侧 pos 是否有反馈变化。"""
        idx = SIDE_INDEX[side]
        before = None
        if self._last_state is not None:
            before = list(self._last_state.data.position)[idx]

        pos = [CLAW_OPEN, CLAW_OPEN]
        if side == "left":
            pos[1] = CLAW_OPEN
        else:
            pos[0] = CLAW_OPEN
        ok = self.call(pos, CLAW_VEL, CLAW_EFFORT_OPEN, tag="verify-%s" % side, abort_on_stall=False)
        time.sleep(pause)
        if self._last_state is None:
            return ok
        after = list(self._last_state.data.position)[idx]
        if before is not None and abs(after - before) < 0.5 and after > 35:
            rospy.logwarn("claw %s may be stuck: pos=%.1f (expected open after cmd %s)", side, after, CLAW_OPEN)
            return False
        return ok

    def run_safe_lock_test(self, side: str, pause: float = 2.0) -> bool:
        """
        渐进闭合测试：open -> 70 -> open -> 80 -> open。
        任一步堵转或 reopen 失败则返回 False。
        """
        idx = SIDE_INDEX[side]
        steps = [
            (CLAW_OPEN, "open"),
            (70.0, "close-70"),
            (CLAW_OPEN, "reopen-1"),
            (min(80.0, MAX_CLOSE_POSITION), "close-80"),
            (CLAW_OPEN, "reopen-2"),
        ]
        print("\n===== safe lock test (%s) =====" % side)
        all_ok = True
        for target, label in steps:
            pos = [CLAW_OPEN, CLAW_OPEN]
            pos[idx] = target
            effort = list(CLAW_EFFORT_OPEN)
            if _is_close_cmd(target):
                effort[idx] = CLAW_EFFORT_CLOSE_LEFT if idx == 0 else CLAW_EFFORT_CLOSE_RIGHT
            ok = self.call(pos, CLAW_VEL, effort, tag=label)
            time.sleep(pause)
            print("  %s ok=%s -> %s" % (label, ok, format_claw_state(self._last_state)))
            if not ok:
                all_ok = False
                print("*** safe lock test FAILED at %s — stop grasp, check 退轴/标定 ***" % label)
                break
        if all_ok:
            print("%s safe lock test PASSED (conservative close/reopen OK)" % side)
        return all_ok


def get_controller() -> SafeClawController:
    """进程内单例，避免重复订阅。"""
    if not hasattr(get_controller, "_instance"):
        get_controller._instance = SafeClawController(load_ros_params=True)
    return get_controller._instance


def safe_call_claw(pos, vel, effort, tag="cmd") -> bool:
    """模块级便捷接口（脚本内一次性调用）。"""
    return get_controller().call(pos, vel, effort, tag=tag)
