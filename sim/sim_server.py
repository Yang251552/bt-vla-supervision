"""RoboCasa simulation server (robocasa venv): the only process that owns the env. zmq REP, wire format in
sim/protocol.py.

    python sim/sim_server.py --task PickPlaceCounterToStove --port 5556
    python sim/sim_server.py --selftest --task PickPlaceCounterToStove [--seeds 1000-1009] [--out selftest.json]
    python sim/sim_server.py --check-assets --task PickPlaceCounterToStove --seeds 1000-1009

Upstream names used ([V file] = seen in that upstream source file, rc_ = robocasa; [U] = not verified there):
  robocasa.wrappers.gym_wrapper.RoboCasaGymEnv(env_name, split, enable_render), .reset(seed), .get_observation,
    .env (robosuite env)                                                        [V rc_gym_wrapper.py]
  env.step(a12) -> (obs, reward, done, info); env._check_success()              [V rc_gym_wrapper.py]
  robots[0].composite_controller._action_split_indexes, .action_limits          [V rc_gym_wrapper.py]
  env.sim.data.site_xpos[robots[0].eef_site_id["right"]], body_xpos[obj_body_id["obj"]]  [V rc_object_utils.py]
  robocasa.utils.object_utils.check_obj_grasped(env, "obj")                     [V rc_object_utils.py]
  env.get_ep_meta()["layout_id"/"style_id"/"object_cfgs"], env.get_obj_lang()   [V kitchen.py, rc_kitchen_pick_place.py]
  robocasa.utils.dataset_registry_utils.get_task_horizon(task)                  [V dataset_registry_utils.py]
  env.control_freq                                                              [V kitchen.py ctor arg; attr U]
  policy obs keys state.base_position/base_rotation/end_effector_*_relative     [V rc_gym_wrapper.py map_obs]
  quaternion order of those obs = xyzw                                          [U]
  env.sim.model._model / env.sim.data._data (raw MuJoCo structs), mujoco.mj_contactForce,
    model.body_rootid, model.site_bodyid, mj_id2name                            [U]
  floor geoms = name contains "floor" or plane type                             [U]
  object "container" exists in PickPlaceCounterToStove                          [V rc_kitchen_pick_place.py]
  env.objects[name].get_bbox_points(trans=body_xpos, rot=xyzw) -> 8 world points [V rc_object_utils.py]
  finger joints "gripper0_right_finger_joint1/2"                                [V rc_object_utils.py]
  arm root joint "robot0_joint1" (first Panda joint)                            [U]

Reply "priv" block (privileged extras, read only by policy/fake_server.py via data["_priv"]; any entry is
None when its upstream name fails): obj_bbox / target_bbox = 8 bbox corners (world; order p0, px, py, pz,
...: p0 = (-,-,-) corner, px - p0 / py - p0 / pz - p0 = the box edges), finger_qpos = raw finger joint qpos
(the values check_obj_grasped thresholds), finger_axis = MjData.xaxis of finger joint 1 (world slide axis =
the closing direction), arm_root_pos = world position of the body of the first arm joint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")          # before anything imports mujoco
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from preempt import action_layout as al  # noqa: E402
from sim import protocol  # noqa: E402

TASKS = ("PickPlaceCounterToStove", "PickPlaceCounterToCabinet")


def yaw_from_xyzw(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class Sim:
    def __init__(self, task: str, split: str = "target"):
        import mujoco
        import robocasa.utils.object_utils as OU
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
        self.mj, self.OU = mujoco, OU
        self.task, self.split = task, split
        self.horizon = int(get_task_horizon(task))
        # same constructor path as upstream eval's gym.make(f"robocasa/{task}", split=..., enable_render=True)
        self.g = RoboCasaGymEnv(env_name=task, split=split, enable_render=True)
        self.env = self.g.env
        self.dt = 1.0 / float(self.env.control_freq)
        self.t = 0

    # ---- privileged state ----
    def _prepare(self):
        """Per-reset bookkeeping: robocasa may rebuild the MuJoCo model on reset."""
        m, mj = self.env.sim.model._model, self.mj
        robot_root = m.body_rootid[m.site_bodyid[self.env.robots[0].eef_site_id["right"]]]
        obj_root = m.body_rootid[self.env.obj_body_id["obj"]]
        self._robot = m.body_rootid == robot_root
        self._obj = m.body_rootid == obj_root
        names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self._gname = names
        self._floor = np.array([("floor" in n.lower()) or m.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE
                                for g, n in enumerate(names)], dtype=bool)
        self._obj_z0 = float(self._obj_pos()[2])
        self._prev = None

        def jbody(name):   # -> (joint id, body id) or None; never fails the reset
            try:
                jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
                return (jid, int(m.jnt_bodyid[jid])) if jid >= 0 else None
            except Exception:
                return None
        self._finger = [jbody(f"gripper0_right_finger_joint{i}") for i in (1, 2)]
        self._arm_root = jbody("robot0_joint1")

    def priv(self) -> dict:
        """Privileged extras for the scripted expert (policy/fake_server.py); see module docstring."""
        e, m, d = self.env, self.env.sim.model._model, self.env.sim.data._data

        def safe(fn):
            try:
                return fn()
            except Exception:   # an upstream name differs: the expert falls back, nothing else depends on it
                return None

        def bbox(name):
            q = np.asarray(e.sim.data.body_xquat[e.obj_body_id[name]], dtype=float)   # MuJoCo wxyz
            pts = e.objects[name].get_bbox_points(trans=np.array(e.sim.data.body_xpos[e.obj_body_id[name]]),
                                                  rot=np.array([q[1], q[2], q[3], q[0]]))
            return np.asarray(pts, dtype=float).tolist()

        return dict(obj_bbox=safe(lambda: bbox("obj")), target_bbox=safe(lambda: bbox("container")),
                    finger_qpos=safe(lambda: [float(d.qpos[m.jnt_qposadr[j]]) for j, _ in self._finger]),
                    finger_axis=safe(lambda: np.asarray(d.xaxis[self._finger[0][0]], dtype=float).tolist()),
                    arm_root_pos=safe(lambda: np.asarray(d.xpos[self._arm_root[1]], dtype=float).tolist()))

    def _obj_pos(self):
        return np.array(self.env.sim.data.body_xpos[self.env.obj_body_id["obj"]], dtype=float)

    def contacts(self, grasped: bool):
        """Robot (plus the target object while grasped) vs non-floor environment, normal force per contact.
        Target object vs robot contacts are never counted (spec d.2: target object-gripper excluded)."""
        m, d, mj = self.env.sim.model._model, self.env.sim.data._data, self.mj
        f6, out = np.zeros(6), []
        for i in range(d.ncon):
            c = d.contact[i]
            g = (c.geom1, c.geom2)
            b = (m.geom_bodyid[g[0]], m.geom_bodyid[g[1]])
            robot = [bool(self._robot[x] or (grasped and self._obj[x])) for x in b]
            env = [not (self._robot[x] or self._obj[x] or self._floor[y]) for x, y in zip(b, g)]
            if (robot[0] and env[1]) or (robot[1] and env[0]):
                mj.mj_contactForce(m, d, i, f6)
                out.append((self._gname[g[0]], self._gname[g[1]], abs(float(f6[0]))))
        return out

    def state(self, obs) -> dict:
        e = self.env
        eef_w = np.array(e.sim.data.site_xpos[e.robots[0].eef_site_id["right"]], dtype=float)
        base_xy = np.asarray(obs["state.base_position"], dtype=float)[:2]
        yaw = yaw_from_xyzw(obs["state.base_rotation"])
        obj = self._obj_pos()
        grasped = bool(self.OU.check_obj_grasped(e, "obj"))
        if self._prev is None:
            v_eef = v_base = 0.0
        else:
            v_eef = float(np.linalg.norm(eef_w - self._prev[0]) / self.dt)
            v_base = float(np.linalg.norm(base_xy - self._prev[1]) / self.dt)
        self._prev = (eef_w, base_xy)
        return dict(
            eef_pos_base=np.asarray(obs["state.end_effector_position_relative"], float).tolist(),
            eef_quat_base=np.asarray(obs["state.end_effector_rotation_relative"], float).tolist(),
            eef_pos_world=eef_w.tolist(), base_pose=[*base_xy.tolist(), yaw], obj_pos=obj.tolist(),
            obj_lift=float(obj[2] - self._obj_z0), grasped=grasped, eef_speed=v_eef, base_speed=v_base,
            contact_force=float(sum(f for _, _, f in self.contacts(grasped))),
            success=bool(e._check_success()))

    def hash(self) -> str:
        d = self.env.sim.data
        return hashlib.sha256(np.asarray(d.qpos).tobytes() + np.asarray(d.qvel).tobytes()).hexdigest()[:16]

    def meta(self) -> dict:
        ep = self.env.get_ep_meta()
        obj_cfg = next((c for c in ep.get("object_cfgs", []) if c.get("name") == "obj"), {})
        return dict(task=self.task, split=self.split, horizon=self.horizon, layout_id=ep.get("layout_id"),
                    style_id=ep.get("style_id"), lang=ep.get("lang", ""), obj_lang=self.env.get_obj_lang(),
                    obj_mjcf=(obj_cfg.get("info") or {}).get("mjcf_path"))

    def _reply(self, obs, meta=False):
        e = self.env
        head = {"ok": True, "t": self.t, "state": self.state(obs), "hash": self.hash(),
                "target_pos": (np.array(e.sim.data.body_xpos[e.obj_body_id["container"]]).tolist()
                               if "container" in e.obj_body_id else None),
                "obs_text": {k: v for k, v in obs.items() if isinstance(v, str)}, "priv": self.priv()}
        if meta:
            head["meta"] = self.meta()
        return head, {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}

    # ---- commands ----
    def reset(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        obs, _ = self.g.reset(seed=seed)   # upstream: env.rng = default_rng(seed); env.reset()
        self.t = 0
        self._prepare()
        return self._reply(obs, meta=True)

    def step(self, action):
        a = np.asarray(action, dtype=float).reshape(al.DIM)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite action {a}")
        raw, _, _, _ = self.env.step(np.clip(a, -1.0, 1.0))
        self.t += 1
        return self._reply(self.g.get_observation(raw))

    def info(self) -> dict:
        cc = self.env.robots[0].composite_controller
        return dict(task=self.task, split=self.split, horizon=self.horizon, dt=self.dt,
                    action_dim=int(cc.action_limits[0].shape[0]),
                    split_indexes={k: list(v) for k, v in cc._action_split_indexes.items()})


def serve(sim: Sim, port: int) -> None:
    import zmq
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{port}")
    print(f"sim_server {sim.task}/{sim.split} on 127.0.0.1:{port}", flush=True)
    while True:
        head, arrays = protocol.recv(sock)
        cmd = head.get("cmd")
        try:
            if cmd == "reset":
                protocol.send(sock, *sim.reset(int(head["seed"])))
            elif cmd == "step":
                protocol.send(sock, *sim.step(arrays["action"]))
            elif cmd == "info":
                protocol.send(sock, {"ok": True, **sim.info()})
            elif cmd == "ping":
                protocol.send(sock, {"ok": True})
            elif cmd == "close":
                protocol.send(sock, {"ok": True})
                return
            else:
                protocol.send(sock, {"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception:
            protocol.send(sock, {"ok": False, "error": traceback.format_exc()})


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def check_assets(task: str, split: str, seeds: list[int]) -> tuple[dict, Sim]:
    """Create the env and reset every seed: a missing asset file fails the MJCF load."""
    sim = Sim(task, split)
    resets = []
    for seed in seeds:
        t0 = time.time()
        head, _ = sim.reset(seed)
        resets.append(dict(seed=seed, s=round(time.time() - t0, 2), **{k: head["meta"][k] for k in
                                                                        ("layout_id", "style_id", "obj_lang", "obj_mjcf")}))
    return dict(task=task, split=split, resets=resets), sim


def selftest(task: str, split: str, seeds: list[int], n_fps: int = 100, k_det: int = 50) -> tuple[dict, int]:
    t0 = time.time()
    summary, sim = check_assets(task, split, seeds)
    summary["create_and_resets_s"] = round(time.time() - t0, 1)
    info = sim.info()
    problems = al.check_against_env(info["split_indexes"], info["action_dim"])
    summary.update(info=info, layout_problems=problems, dt=sim.dt, control_hz_ok=abs(sim.dt - 0.05) < 1e-9)
    if problems:
        summary["ok"] = False
        return summary, 1

    # determinism: same seed, same fixed action sequence, twice; compare qpos/qvel hashes
    rng = np.random.default_rng(0)
    acts = [al.make(eef_pos=rng.uniform(-0.3, 0.3, 3), eef_rot=rng.uniform(-0.3, 0.3, 3)) for _ in range(k_det)]
    runs = []
    for _ in range(2):
        sim.reset(seeds[0])
        hs = [sim.hash()]
        for a in acts:
            sim.step(a)
            hs.append(sim.hash())
        runs.append(hs)
    first_diff = next((i for i, (x, y) in enumerate(zip(*runs)) if x != y), None)
    summary["determinism"] = dict(seed=seeds[0], steps=k_det, identical=first_diff is None,
                                  first_diff_step=first_diff, reset_hash=runs[0][0], final_hash=runs[0][-1])

    # throughput with rendering on (as in episodes)
    sim.reset(seeds[0])
    t0 = time.time()
    for _ in range(n_fps):
        sim.step(al.hold(al.GRIP_OPEN))
    summary["steps_per_s"] = round(n_fps / (time.time() - t0), 2)

    # controller probe: frames, signs and scale of eef / base commands (calibrates recovery.Gains)
    head, _ = sim.reset(seeds[0])
    s0 = protocol.state_from_dict(head["state"])
    summary["contacts_at_reset"] = sorted(sim.contacts(s0.grasped), key=lambda c: -c[2])[:10]
    summary["contact_force_at_reset"] = s0.contact_force

    def run(a, n=10):
        for _ in range(n):
            h, _ = sim.step(a)
        return protocol.state_from_dict(h["state"])
    s1 = run(al.make(eef_pos=(0.5, 0, 0)))
    s2 = run(al.hold(al.GRIP_OPEN))
    s3 = run(al.make(base_vel=(0.5, 0, 0), base_mode=al.BASE_MODE))
    s4 = run(al.make(base_vel=(0, 0, 0.5), base_mode=al.BASE_MODE))
    s4h = run(al.hold(al.GRIP_OPEN))
    s5 = run(al.make(eef_rot=(0, 0, 0.5)))                                   # eef yaw (base frame), arm mode
    s6 = run(al.make(base_vel=(0, 0.5, 0), base_mode=al.BASE_MODE))          # base lateral (left)
    from preempt.recovery import rot_error
    head_dir = np.array([np.cos(s2.base_pose[2]), np.sin(s2.base_pose[2])])
    db = s3.base_pose[:2] - s2.base_pose[:2]
    left = np.array([-np.sin(s5.base_pose[2]), np.cos(s5.base_pose[2])])
    summary["probe"] = dict(
        eef_base_delta_per_step_per_unit=((s1.eef_pos_base - s0.eef_pos_base) / 5.0).round(5).tolist(),
        base_forward_per_step_per_unit=round(float(db @ head_dir) / 5.0, 5),
        base_lateral_per_step_per_unit=round(float(db @ np.array([-head_dir[1], head_dir[0]])) / 5.0, 5),
        yaw_per_step_per_unit=round(float(np.angle(np.exp(1j * (s4.base_pose[2] - s3.base_pose[2])))) / 5.0, 5),
        eef_rotvec_per_step_per_unit=(rot_error(s5.eef_quat_base, s4h.eef_quat_base) / 5.0).round(5).tolist(),
        base_left_per_step_per_unit=round(float((s6.base_pose[:2] - s5.base_pose[:2]) @ left) / 5.0, 5),
        expected=dict(eef_m_per_unit=0.0118, base_m_per_unit=0.0041, yaw_rad_per_unit=0.046,
                      eef_rad_per_unit_upper_bound=0.5))   # recovery.Gains (measured 2026-10-08)
    priv = sim.priv()
    summary["priv_missing"] = sorted(k for k, v in priv.items() if v is None)   # names to fix before full-chain runs
    summary["df_root"] = subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout.strip()
    summary["ok"] = True
    return summary, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=TASKS[0], choices=TASKS)
    p.add_argument("--split", default="target")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--check-assets", action="store_true")
    p.add_argument("--seeds", default="1000")   # dev seed; eval seeds 0..39 only via c6_score.sh
    p.add_argument("--out", default=None, help="also write the JSON summary here")
    a = p.parse_args()
    if a.selftest or a.check_assets:
        try:
            summary, rc = (selftest(a.task, a.split, parse_seeds(a.seeds)) if a.selftest else
                           (check_assets(a.task, a.split, parse_seeds(a.seeds))[0], 0))
        except Exception:
            summary, rc = {"ok": False, "error": traceback.format_exc()}, 1
        text = json.dumps(summary, indent=1, default=str)
        print(text)
        if a.out:
            Path(a.out).write_text(text + "\n")
        sys.exit(rc)
    serve(Sim(a.task, a.split), a.port)


if __name__ == "__main__":
    main()
