"""bt_runner (ROS 2 Humble, ros venv): the bt/tree.py tree, unchanged, ticked exactly once per /bt/state.

    python ros/bt_runner.py --mode bt_safe_abort [--config config/protocol.json]

The tree reaches the world only through Ctx.client and Ctx.arbiter:
  RosSkillClient  ros.lockstep.DeferredSkillClient over rclpy.action.ActionClient("vla_skill"), with the calls
                  py_trees_ros.action_clients.FromBlackboard makes [V ptr_ac.py]: send_goal_async(goal,
                  feedback_callback, goal_uuid=unique_identifier_msgs UUID), goal_handle.accepted,
                  get_result_async(), cancel_goal_async(), result.status vs action_msgs GoalStatus.STATUS_*.
                  py_trees_ros drops a cancel issued before the goal handle exists [V ptr_ac.py
                  send_cancel_request]; here such a cancel is sent as soon as the goal is accepted.
  RemoteArbiter   ros.lockstep: mirror reads, queued writes applied by skill_host at the same step.
Per tick: barrier (client view of every goal == skill_host's report) -> tick -> ack tick(t) -> send the tick's
goal / cancel requests -> wait for every response -> ack done(t). Protocol: ros/lockstep.py docstring.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402
from action_msgs.msg import GoalStatus as RosGoalStatus  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from unique_identifier_msgs.msg import UUID  # noqa: E402

from bt.tree import Ctx, build_tree  # noqa: E402
from bt_vla_interfaces.action import VlaSkill  # noqa: E402
from preempt.contracts import GoalStatus, Phase, ProtocolConfig, StopMode  # noqa: E402
from preempt.eventlog import _jsonable  # noqa: E402
from ros.lockstep import WIRE, DeferredSkillClient, Forward, RemoteArbiter  # noqa: E402
from sim.protocol import state_from_dict  # noqa: E402

QOS = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
RESULT = {RosGoalStatus.STATUS_SUCCEEDED: GoalStatus.SUCCEEDED, RosGoalStatus.STATUS_CANCELED: GoalStatus.CANCELED,
          RosGoalStatus.STATUS_ABORTED: GoalStatus.ABORTED}


class RosSkillClient(DeferredSkillClient):
    def __init__(self, node: Node, log):
        super().__init__(log)
        self.ac = ActionClient(node, VlaSkill, "vla_skill")
        self.fb_rx = 0
        self._lk = threading.Lock()
        self._outstanding = 0          # requests sent whose response is not in yet

    @property
    def outstanding(self) -> int:
        return self._outstanding

    def _count(self, d: int) -> None:
        with self._lk:
            self._outstanding += d

    def flush(self) -> None:
        if self.goal_out is not None:
            gid, instruction, mode = self.goal_out
            self.goal_out = None
            self._count(+1)
            fut = self.ac.send_goal_async(VlaSkill.Goal(instruction=instruction, stop_mode=mode),
                                          feedback_callback=self._on_feedback,
                                          goal_uuid=UUID(uuid=list(bytes.fromhex(gid))))
            fut.add_done_callback(lambda f, gid=gid: self._on_response(gid, f))
        for gid in self.cancel_out:
            self._send_cancel(gid)
        self.cancel_out = []

    def _send_cancel(self, gid) -> None:
        g = self.goals[gid]
        if g["handle"] is None:
            if g["accepted"] is None:
                g["cancel_on_accept"] = True       # goal response not in yet
            return                                 # rejected goal: nothing to cancel
        self._count(+1)
        g["handle"].cancel_goal_async().add_done_callback(lambda f: self._count(-1))

    # ---- executor thread ----
    def _on_response(self, gid, fut) -> None:
        g = self.goals[gid]
        h = fut.result()
        if h is None or not h.accepted:
            g["accepted"] = False
        else:
            g["handle"], g["accepted"] = h, True
            h.get_result_async().add_done_callback(lambda f, gid=gid: self._on_result(gid, f))
            if g["cancel_on_accept"]:
                self._send_cancel(gid)
        self._count(-1)

    def _on_result(self, gid, fut) -> None:
        self.goals[gid]["result"] = RESULT.get(fut.result().status, GoalStatus.ABORTED)

    def _on_feedback(self, msg) -> None:
        f = msg.feedback
        self.fb_rx += 1
        self.fb = {"step": f.step, "phase": f.phase, "chunk": f.chunk, "offset": f.offset}


def wait_until(cond, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not cond():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        time.sleep(0.001)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=[m.value for m in StopMode] + ["none"])
    p.add_argument("--config", default=str(ROOT / "config" / "protocol.json"))
    p.add_argument("--state-timeout", type=float, default=900.0, help="first inference may take minutes")
    p.add_argument("--sync-timeout", type=float, default=60.0)
    a = p.parse_args()
    cfg = ProtocolConfig.load(a.config)
    mode = StopMode.TRUNCATE_HOLD if a.mode == "none" else StopMode(a.mode)   # uninterrupted: never cancelled

    rclpy.init()
    node = Node("bt_runner", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    states: queue.Queue = queue.Queue()
    ack_pub = node.create_publisher(String, "/bt/ack", QOS)
    fwd, arb = Forward(), RemoteArbiter()
    client = RosSkillClient(node, fwd)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not client.ac.wait_for_server(timeout_sec=120.0):
        sys.exit("vla_skill action server not found")
    node.create_subscription(String, "/bt/state", lambda m: states.put(json.loads(m.data)), QOS)

    def ack(**kw):
        ack_pub.publish(String(data=json.dumps(dict(kw, v=WIRE), default=_jsonable)))

    ctx = root = task = None
    try:
        while True:
            m = states.get(timeout=a.state_timeout)
            if m.get("v") != WIRE:
                raise RuntimeError(f"/bt/state wire version {m.get('v')!r}")
            if m.get("end"):
                break
            t, s = m["t"], state_from_dict(m["state"])
            if ctx is None:   # home pose = eef pose at reset (retract target), as in bt/episode.py
                ctx = Ctx(cfg, client, arb, fwd, mode, m["lang"], s.eef_pos_base, s.eef_quat_base)
                root, task = build_tree(ctx)
            wait_until(lambda: client.caught_up(m["goals"]), a.sync_timeout, f"goal status sync at step {t}")
            ctx.t, ctx.state, ctx.phase, ctx.interrupt = t, s, Phase(m["phase"]), bool(m["interrupt"])
            arb.sync(m["owner"], m["last_gripper"])
            fwd.events = []
            root.tick_once()
            n = client.pending()
            tip = root.tip()
            ack(kind="tick", t=t, ops=arb.ops, events=fwd.events, requests=n, root=root.status.value,
                task=task.status.value, abort_failed=ctx.abort_failed, goals_sent=ctx.goals_sent,
                tip=tip.name if tip else None, fb_rx=client.fb_rx, cfg_hash=cfg.hash())
            if n:
                client.flush()
                wait_until(lambda: client.outstanding == 0, a.sync_timeout, f"action responses at step {t}")
                ack(kind="done", t=t)
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
