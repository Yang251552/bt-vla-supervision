"""Scripted expert (policy/fake_server.plan) against a small kinematic kitchen: lagged eef whose true gain
differs from the planning gain, sagging arm, fingers that stop on the object, grasp / drop / success rules
shaped like robocasa's (check_obj_grasped < 0.035, object in pan within 0.07, eef > 0.25 m away).
Checks the stage logic end to end through the real wire mapping; it is not physics."""
import numpy as np
import pytest

from policy.client import to_chunk
from policy.fake_server import Opts, plan
from preempt import action_layout as al

SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])


def rz(y):
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class World:
    COUNTER, PAN_BOT = 0.90, 0.92

    def __init__(self, obj_yaw=0.7, pan_y=0.35, g_eef=0.014, g_rot=0.12, g_base=0.0045, extras=True):
        self.base, self.g = np.array([0.2, -0.1, 0.3]), (g_eef, g_rot, g_base)
        self.eef_b, self.v = np.array([0.50, 0.0, 1.15]), np.zeros(3)      # base frame xy, world z
        self.root_b = np.array([0.15, 0.0, 1.25])
        self.half, self.oyaw = np.array([0.05, 0.0125, 0.0125]), self.base[2] + obj_yaw
        self.ctr = self.w([0.75, -0.25, self.COUNTER + 0.01 + 0.0125])
        self.c_off = np.array([0.004, 0.0, -0.005])                          # bbox centre - body origin
        self.z0 = self.ctr[2] - self.c_off[2]
        self.pan = self.w([0.65, pan_y, self.PAN_BOT])
        self.q, self.fyaw = 0.04, self.base[2] + np.pi / 2                  # fingers close along base y
        self.grasped, self.in_pan, self.extras = False, False, extras

    def w(self, p_b):
        return np.r_[self.base[:2], 0.0] + rz(self.base[2]) @ np.asarray(p_b, float)

    @property
    def eef(self):
        return self.w(self.eef_b)

    def success(self):
        return self.in_pan and not self.grasped and np.linalg.norm(self.eef - (self.ctr - self.c_off)) > 0.25

    def priv(self):
        body = self.ctr - self.c_off
        p = dict(eef_pos_world=self.eef, base_pose=self.base.copy(), obj_pos=body, obj_lift=body[2] - self.z0,
                 grasped=self.grasped, success=self.success(), target_pos=self.pan.copy(), eef_speed=0.0,
                 base_speed=0.0, contact_force=0.0)
        if self.extras:
            pan_half = np.array([0.17, 0.12, 0.025])                          # handle widens the bbox
            p.update(obj_bbox=self.ctr + (rz(self.oyaw) @ (SIGNS * self.half).T).T,
                     target_bbox=self.pan + [0.05, 0, 0.025] + SIGNS * pan_half,
                     finger_qpos=[self.q, -self.q], finger_axis=[np.cos(self.fyaw), np.sin(self.fyaw), 0.0],
                     arm_root_pos=self.w(self.root_b))
        return p

    def step(self, a):
        g_eef, g_rot, g_base = self.g
        bv = al.get(a, "base_vel")
        self.base[:2] += rz(self.base[2])[:2, :2] @ (bv[:2] * g_base)
        self.v += 0.5 * (al.get(a, "eef_pos") * g_eef - self.v)
        self.eef_b = self.eef_b + self.v - [0, 0, 0.0003 * self.grasped]  # sag while holding
        self.eef_b[2] = max(self.eef_b[2], self.COUNTER + 0.018)          # fingers on the plate
        self.fyaw += al.get(a, "eef_rot")[2] * g_rot
        e = self.eef
        between = np.linalg.norm(e[:2] - self.ctr[:2]) < 0.02 and abs(e[2] - self.ctr[2]) < 0.02
        d = self.fyaw - self.oyaw
        width = abs(self.half[0] * np.cos(d)) + abs(self.half[1] * np.sin(d))   # half width along the fingers
        was = self.grasped
        if al.get(a, "gripper")[0] > 0:
            self.q = max(self.q - 0.006, width if between else 0.0)
        else:
            self.q = min(self.q + 0.006, 0.04)
        self.grasped = bool(between and self.q <= width + 1e-9 and self.q < 0.035) or \
            (was and al.get(a, "gripper")[0] > 0)
        if self.grasped and not was:
            self.off = self.ctr - e
        if self.grasped:
            self.ctr = e + self.off
            self.in_pan = False
        elif was:                                                        # released: drops
            inside = np.linalg.norm(self.ctr[:2] - self.pan[:2]) < 0.10
            self.ctr[2] = (self.PAN_BOT + 0.005 if inside else self.COUNTER) + self.half[2]
            self.in_pan = inside and np.linalg.norm(self.ctr[:2] - self.c_off[:2] - self.pan[:2]) < 0.07


def rollout(world, steps=900, opts=Opts()):
    acts, lift, stages, base0 = [], 0.0, [], world.base.copy()
    for t in range(steps):
        if world.success():
            break
        if t % 16 == 0:
            p = world.priv()
            d, st = plan(p, "pick the cream cheese stick and place it in the pan",
                         None if world.extras else np.array([world.q, -world.q]), opts)
            chunk, stages = to_chunk(d, 16), stages + st
        a = chunk[t % 16]
        acts.append(a)
        world.step(a)
        lift = max(lift, world.priv()["obj_lift"])
    return dict(success=world.success(), t=t, acts=np.array(acts), lift=lift, stages=stages,
                base_moved=float(np.linalg.norm(world.base[:2] - base0[:2])))


@pytest.mark.parametrize("kw", [dict(), dict(g_eef=0.009, g_rot=0.5, g_base=0.003), dict(obj_yaw=-1.2)])
def test_expert_completes_pick_place_without_moving_the_base(kw):
    r = rollout(World(**kw))
    assert r["success"], (r["t"], r["stages"][::16])
    assert r["lift"] >= 0.05 and r["base_moved"] < 1e-9
    assert np.all(np.abs(r["acts"]) <= 1.0) and np.all(r["acts"][:, 11] == al.ARM_MODE)
    for s in ("approach", "descend", "close", "transport", "release", "retreat"):
        assert s in r["stages"], s


def test_expert_uses_the_base_only_when_the_pan_is_out_of_reach():
    r = rollout(World(pan_y=0.95))
    assert r["success"] and r["base_moved"] > 0.1
    base_steps = r["acts"][:, 11] == al.BASE_MODE
    assert base_steps.any() and not base_steps[:40].any()            # approach stays in arm mode


def test_expert_without_privileged_extras_falls_back():
    r = rollout(World(obj_yaw=0.0, extras=False))                    # already aligned, no bbox / root / fingers
    assert r["success"] and r["base_moved"] < 1e-9


def test_expert_is_deterministic_and_obeys_stop():
    a, b = rollout(World()), rollout(World())
    assert np.array_equal(a["acts"], b["acts"])
    d, st = plan(World().priv(), "stop")
    assert st == ["stop"] * 16 and not np.any(to_chunk(d, 16)[:, :6])
