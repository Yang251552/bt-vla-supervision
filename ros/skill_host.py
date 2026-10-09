"""skill_host (ROS 2 Humble, ros venv): SimBridge (the only env stepper), the /vla_skill ActionServer around
preempt.skill.SkillController, the single command Arbiter, phase detection, interrupt injection and the
episode event log. One episode per run; scripts/run_episode.py calls run_episode(), bt_runner is the peer.

Lockstep, per control step t (same order as bt/episode.py, which the stub tests cover):
  1  s_t -> SUCCESS / phase / interrupt I,U (at phase entry + k, and + W) / standstill H
  2  skill.pre_step(t)                     inference blocks here and never consumes control steps
  3  finalize ROS goals                    terminal controller status -> goal_handle succeed/canceled/abort
  4  publish /clock (t seconds = step t) and /bt/state(t)
  5  wait /bt/ack tick(t): log the BT's events, apply its arbiter ops (recovery acquire/release/offer),
     then release the goal/cancel callbacks of this tick (they wait on `applied`, so R is logged before A
     and a recovery release is applied before a new goal is judged); if the tick sent requests, wait
     /bt/ack done(t) (bt_runner got every response, so those callbacks have run)
  6  stop if the BT finished; else skill.act(t) -> arbiter.resolve(t) -> feedback -> STEP -> sim.step
The BT is a separate process (bt_runner) that ticks exactly once per /bt/state message.

rclpy names (assumed, not seen in source): ActionServer(goal_callback gets only the Goal message, not its
id; handle_accepted_callback gets the ServerGoalHandle; cancel_callback -> CancelResponse), ServerGoalHandle
.goal_id/.request/.execute()/.succeed()/.canceled()/.abort()/.publish_feedback(), MultiThreadedExecutor,
Node(parameter_overrides=[use_sim_time]), rosgraph_msgs/Clock. One ActionServer has one callback group:
it is Reentrant, so a cancel request is handled while the goal's execute callback is blocked.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool, String

from bt_vla_interfaces.action import VlaSkill
from policy.client import ZmqPolicyClient
from preempt.arbiter import Arbiter
from preempt.contracts import BUSY, Ev, GoalStatus, Phase, ProtocolConfig, StopMode
from preempt.eventlog import EventLog, _jsonable
from preempt.phase import PhaseDetector
from preempt.recovery import still
from preempt.skill import SkillController
from ros.lockstep import WIRE, apply_ops, goal_reject_reason, report_statuses
from sim.protocol import SimClient

TERMINAL = (GoalStatus.SUCCEEDED, GoalStatus.CANCELED, GoalStatus.ABORTED, GoalStatus.REJECTED)
QOS = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
TIMEOUT_S = 60.0


def gid_of(goal_handle) -> str:
    return bytes(goal_handle.goal_id.uuid).hex()   # == uuid.UUID(...).hex on the client side


class LogTap:
    """Thread-safe EventLog front; remembers each goal's generation, terminal reason and fence step."""

    def __init__(self, log: EventLog):
        self.log, self._lk = log, threading.Lock()
        self.gen_of, self.q_step, self.reason = {}, {}, {}

    def __call__(self, ev, t, **f):
        with self._lk:
            if ev == Ev.GOAL_ACCEPTED:
                self.gen_of[f["goal_id"]] = f["gen"]
            elif ev == Ev.Q:
                self.q_step[f["gen"]] = t
            elif ev in (Ev.CANCELED, Ev.GOAL_SUCCEEDED, Ev.GOAL_ABORTED):
                self.reason[f["goal_id"]] = f["reason"]
            self.log(ev, t, **f)

    def close(self):
        self.log.close()


class SkillHost(Node):
    def __init__(self, cfg: ProtocolConfig, policy: ZmqPolicyClient, log: LogTap, instruction: str):
        super().__init__("skill_host",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.cfg, self.log, self.instruction = cfg, log, instruction
        self.lock = threading.Lock()                 # guards skill + arbiter; lock order: self.lock -> log
        self.arb = Arbiter(log)
        self.skill = SkillController(cfg, policy, self.arb, log, max_wait_polls=policy.max_wait_polls)
        self.t = 0
        self.applied = threading.Event()             # ack(t) applied: this tick's goal/cancel may proceed
        self.handles, self.done, self.final = {}, {}, {}
        self.acks: queue.Queue = queue.Queue()
        self._chunk_key, self._offset = None, 0
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.state_pub = self.create_publisher(String, "/bt/state", QOS)
        self.intr_pub = self.create_publisher(Bool, "/interrupt", QOS)   # mirror for observers; the BT uses /bt/state
        self.create_subscription(String, "/bt/ack", lambda m: self.acks.put(json.loads(m.data)), QOS,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.server = ActionServer(self, VlaSkill, "vla_skill", execute_callback=self._execute,
                                   goal_callback=self._on_goal, handle_accepted_callback=self._on_accepted,
                                   cancel_callback=self._on_cancel, callback_group=ReentrantCallbackGroup())

    # ---- action server callbacks (executor threads) ----
    def _on_goal(self, goal):
        if not self.applied.wait(TIMEOUT_S):
            return GoalResponse.REJECT
        with self.lock:
            reason = goal_reject_reason(self.skill, self.arb, goal.stop_mode)
            if reason:
                self.log(Ev.GOAL_REJECTED, self.t, goal_id=None, reason=reason)
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_accepted(self, gh):
        gid = gid_of(gh)
        with self.lock:
            self.handles[gid], self.done[gid], self.final[gid] = gh, threading.Event(), threading.Event()
            if not self.skill.submit_goal(self.t, gid, gh.request.instruction, StopMode(gh.request.stop_mode)):
                self.done[gid].set()                 # refused after all (logged): execute aborts it
        gh.execute()

    def _on_cancel(self, gh):
        if not self.applied.wait(TIMEOUT_S):
            return CancelResponse.REJECT
        with self.lock:
            ok = self.skill.cancel(self.t, gid_of(gh))
        return CancelResponse.ACCEPT if ok else CancelResponse.REJECT

    def _execute(self, gh):
        gid = gid_of(gh)
        self.done[gid].wait()
        with self.lock:
            st = self.skill.statuses[gid]
            reason = self.log.reason.get(gid, st.value)
            q = self.log.q_step.get(self.log.gen_of.get(gid), -1)
        try:
            {GoalStatus.SUCCEEDED: gh.succeed, GoalStatus.CANCELED: gh.canceled}.get(st, gh.abort)()
        except Exception as e:   # ROS refused the transition: a protocol fault, make it count
            self.log(Ev.VIOLATION, self.t, kind="ros_transition", goal_id=gid, status=st, error=repr(e))
            try:
                gh.abort()
            except Exception:
                pass
        self.final[gid].set()
        return VlaSkill.Result(reason=str(reason), release_step=int(q))

    # ---- lockstep helpers (main thread) ----
    def _finalize(self, everything: bool = False) -> None:
        with self.lock:
            due = [g for g, ev in self.done.items()
                   if not ev.is_set() and (everything or self.skill.statuses[g] in TERMINAL)]
            for g in due:
                self.done[g].set()
        for g in due:
            if not self.final[g].wait(TIMEOUT_S):
                raise TimeoutError(f"goal {g} not finalized")

    def _wait_ack(self, t: int, kind: str) -> dict:
        deadline = time.monotonic() + TIMEOUT_S
        while True:
            try:
                m = self.acks.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"no /bt/ack {kind} for step {t}") from None
            if m.get("v") != WIRE:
                raise RuntimeError(f"/bt/ack wire version {m.get('v')!r}")
            if m.get("t") == t and m.get("kind") == kind:
                return m

    def _feedback(self, t: int, owner, phase: Phase) -> None:
        g = self.skill.goal
        if g is None or owner != ("vla", g.gen) or self.skill.statuses[g.goal_id] not in BUSY:
            return
        key = (g.gen, (self.skill.feedback or {}).get("chunk", -1))
        self._offset = self._offset + 1 if key == self._chunk_key else 0
        self._chunk_key = key
        gh = self.handles.get(g.goal_id)
        if gh is not None:
            gh.publish_feedback(VlaSkill.Feedback(step=t, phase=phase.value, chunk=int(key[1]),
                                                  offset=self._offset))

    def _publish(self, msg: dict) -> None:
        self.state_pub.publish(String(data=json.dumps(dict(msg, v=WIRE), default=_jsonable)))

    def wait_for_peer(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not (self.state_pub.get_subscription_count() and self.count_publishers("/bt/ack")):
            if time.monotonic() > deadline:
                raise TimeoutError("bt_runner did not connect")
            time.sleep(0.1)
        time.sleep(0.5)

    # ---- the episode ----
    def run(self, sim: SimClient, first, inject_phase: Phase | None, max_steps: int, priv: bool = False,
            frame_dir: Path | None = None, frame_every: int = 2) -> dict:
        cfg, log = self.cfg, self.log
        det = PhaseDetector(cfg, log)
        step, s = first, first.state
        t_i, n_still, h_done, succ_done, interrupt = None, 0, False, False, False
        outcome, ack, t = "timeout", {}, 0
        for t in range(max_steps):
            self.t = t
            self.applied.clear()
            if s.success and not succ_done:
                succ_done = True
                log(Ev.SUCCESS, t)
            phase = det.update(t, s)
            if t_i is None and inject_phase is not None and inject_phase in det.entered:
                t_i = det.entered[inject_phase] + cfg.k(inject_phase)
            if t == t_i:
                interrupt = True
                log(Ev.I, t, phase=phase, state_hash=step.head["hash"])
                self.intr_pub.publish(Bool(data=True))
            if t_i is not None and t == t_i + cfg.window_W:
                interrupt = False
                log(Ev.U, t)
                self.intr_pub.publish(Bool(data=False))
            if t_i is not None and t >= t_i and not h_done:
                n_still = n_still + 1 if still(s, cfg) else 0
                if n_still >= cfg.standstill_steps:
                    h_done = True
                    log(Ev.H, t)
            obs = {"policy": step.obs,
                   "priv": dict(step.head["state"], target_pos=step.head["target_pos"], **step.head.get("priv", {}))
                   if priv else None}
            with self.lock:
                self.skill.pre_step(t, s, obs=obs)
            self._finalize()
            self.clock_pub.publish(Clock(clock=TimeMsg(sec=t, nanosec=0)))
            with self.lock:
                self._publish({"t": t, "state": step.head["state"], "phase": phase.value, "interrupt": interrupt,
                               "goals": report_statuses(self.skill), "owner": self.arb.owner,
                               "last_gripper": self.arb.last_gripper, "lang": self.instruction})
            ack = self._wait_ack(t, "tick")
            if ack.get("cfg_hash") != cfg.hash():
                raise RuntimeError("bt_runner loaded a different protocol config")
            with self.lock:
                for e in ack["events"]:
                    e = dict(e)
                    log(e.pop("ev"), e.pop("t", t), **e)
                apply_ops(self.arb, t, ack["ops"], log, cfg.action_dim)
            self.applied.set()
            if ack["requests"]:
                self._wait_ack(t, "done")
            if ack["task"] == "SUCCESS":
                outcome = "success"
                break
            if ack["root"] == "FAILURE":
                outcome = "blocked" if ack["abort_failed"] else "failed"
                break
            with self.lock:
                self.skill.act(t)
                a, owner = self.arb.resolve(t)
                self._feedback(t, owner, phase)
            log(Ev.STEP, t, src=owner[0] if owner else "hold", gen=owner and owner[1],
                eef=s.eef_pos_world.round(6), obj=s.obj_pos.round(4), base=s.base_pose.round(6), v_eef=round(s.eef_speed, 6),
                v_base=round(s.base_speed, 6), grasped=s.grasped, lift=round(s.obj_lift, 6),
                force=round(s.contact_force, 3), grip=self.arb.last_gripper, phase=phase, bt=ack.get("tip"))
            if frame_dir is not None and t % frame_every == 0:
                from PIL import Image
                Image.fromarray(np.asarray(step.obs["video.robot0_agentview_left"])).save(
                    frame_dir / f"{t:05d}.jpg", quality=90)
            step = sim.step(a)
            s = step.state
        self._publish({"t": t, "end": True})
        self._finalize(everything=True)
        tgt = step.head.get("target_pos")
        return {"outcome": outcome, "steps": t, "goals_sent": ack.get("goals_sent"),
                "obj_tgt_xy_m": None if tgt is None else round(float(np.linalg.norm(s.obj_pos[:2] - np.asarray(tgt)[:2])), 4),
                "late_replies": self.skill.late_replies, "violations": self.arb.violations,
                "fb_rx": ack.get("fb_rx"), "t_I": t_i}


def run_episode(a) -> dict:
    """a: argparse namespace from scripts/run_episode.py."""
    cfg = ProtocolConfig.load(a.config)
    sim = SimClient(a.sim)
    first = sim.reset(a.seed)
    meta = first.head["meta"]
    inject = None if a.phase == "none" else Phase(a.phase)
    log = LogTap(EventLog(a.log, cfg.hash(), seed=a.seed, mode=a.mode, phase=a.phase, rep=a.rep,
                          policy_seed=a.policy_seed, policy=a.policy, reset_hash=first.head["hash"], **meta))
    policy = ZmqPolicyClient(a.policy_host, a.policy_port, episode_seed=a.policy_seed, chunk_len=cfg.chunk_len,
                             priv=a.priv)
    max_steps = a.max_steps or 2 * cfg.reentry_budget_B + cfg.window_W + 2 * cfg.abort_budget_T
    rclpy.init()
    node = SkillHost(cfg, policy, log, meta["lang"])
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    res = {"outcome": "error"}
    try:
        node.wait_for_peer()
        res = node.run(sim, first, inject, max_steps, priv=a.priv, frame_dir=a.frame_dir)
    finally:
        lat = np.array(policy.latency_ms or [np.nan])
        res.update(meta=meta, policy_ms_mean=float(lat.mean()), policy_ms_max=float(lat.max()),
                   policy_queries=len(policy.latency_ms))
        log(Ev.EPISODE_END, res.get("steps", -1), **{k: v for k, v in res.items() if k != "meta"})
        node.applied.set()                           # unblock any callback still waiting (error path)
        for ev in list(node.done.values()):
            ev.set()
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        policy.close()
        log.close()
    return res
