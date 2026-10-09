# Integration on the AWS instance

Written and checked locally (py_compile, bash -n, pytest incl. `tests/test_lockstep.py`) before the first instance
run. Upstream names the code relies on are either seen in the upstream source or assumed (`sim/sim_server.py`
marks them [V] / [U]); the checklist at the end is what the first instance runs had to measure or check, kept as
written before them.

## Processes

```
 robocasa venv (py3.11)          ros venv (py3.10 + Humble)                      gr00t venv (py3.10)
 ┌──────────────────┐  zmq REQ/REP  ┌───────────────────────────────┐  zmq REQ/REP  ┌──────────────────────┐
 │ sim/sim_server   │◄─────────────►│ skill_host (in run_episode.py)│◄─────────────►│ policy/seeded_server │
 │ only env owner   │ 127.0.0.1:5556│  SimBridge: the only stepper  │ 127.0.0.1:5560│  (or fake_server)    │
 │ wire v1: JSON +  │               │  SkillController + Arbiter    │ torch.save    │  GR00T N1.5          │
 │ raw arrays       │               │  /vla_skill ActionServer      │ dicts (as     │  seeded per          │
 └──────────────────┘               │  phase, interrupt, event log  │ upstream)     │  (episode, query)    │
                                    └──────┬─────────────▲──────────┘               └──────────────────────┘
                     /clock, /bt/state(t)  │             │ /bt/ack tick(t) [+ done(t)]
                     /vla_skill feedback   ▼             │ /vla_skill goal / cancel / result
                                    ┌──────────────────────────────┐
                                    │ ros/bt_runner (child process)│
                                    │ bt/tree.py unchanged:        │
                                    │ RosSkillClient, RemoteArbiter│
                                    └──────────────────────────────┘
```

| Endpoint | Kind | Between |
|---|---|---|
| `tcp://127.0.0.1:5556` | zmq REP, wire v1 (`sim/protocol.py`) | sim_server ← skill_host, repro.py |
| `tcp://127.0.0.1:5560` | zmq REP, GR00T protocol (`policy/client.py`) | seeded / fake server ← skill_host, repro.py |
| `/vla_skill` | action `bt_vla_interfaces/VlaSkill` | skill_host (server) ← bt_runner (client) |
| `/clock` | `rosgraph_msgs/Clock`, sec = control step | skill_host → all (`use_sim_time`) |
| `/bt/state`, `/bt/ack` | `std_msgs/String` JSON, wire v1 (`ros/lockstep.py`) | lockstep pair |
| `/interrupt` | `std_msgs/Bool` on change | observers only; the BT reads the flag in `/bt/state` |

Lockstep per control step t: skill_host pre_step (inference blocks, never consumes steps) → finalize ROS
goals → `/clock`, `/bt/state(t)` → bt_runner waits until its action-client view equals the reported goal
statuses, ticks once, sends `ack tick(t)` (arbiter ops, BT events), then this tick's goal / cancel requests,
then `ack done(t)` → skill_host applies the ops before the goal / cancel callbacks may run → act → resolve →
step. The BT recovery motion travels as `offer` ops, so there is no ScriptedMotion action and skill_host
stays the single command owner. Compared with the in-process stub, WaitSkillIdle sees CANCELED one tick
later, so SafeAbort starts one step later (robot holds that step); criteria are unaffected.

## On the instance

Paths (`scripts/remote/env.sh`): project `~/btvla`, sources `~/src`, venvs
`~/venvs/{robocasa,gr00t,ros}`, checkpoint `~/ckpt/robocasa365_checkpoints/...`, ROS workspace `~/ros_ws`,
logs `~/setup_logs/`. All on the root EBS volume (`on_root` guard).

Deploy: upload `preempt bt sim policy ros config scripts tests` to `~/btvla` (plus `eval` for scoring).

Install (each script idempotent, rerun after a stop resumes; `rm ~/setup_logs/<step>.done` forces a step):

```bash
cd ~/btvla
bash scripts/remote/setup_robocasa.sh
bash scripts/remote/fetch_assets.sh PickPlaceCounterToStove      # light subset, check, else all
bash scripts/remote/setup_gr00t.sh
bash scripts/remote/fetch_checkpoint.sh
bash scripts/remote/setup_ros.sh
```

Run (`source scripts/remote/env.sh; ros_env` first for anything on `$ROS_PY`):

```bash
$RC_PY sim/sim_server.py --selftest --task PickPlaceCounterToStove --seeds 1000-1009 --out ~/selftest.json   # install check
$RC_PY sim/sim_server.py --task PickPlaceCounterToStove &                                  # :5556
$G_PY policy/fake_server.py &                     # g4dn;  g5: $G_PY policy/seeded_server.py --model_path $CKPT &
$ROS_PY scripts/run_episode.py --ping
$ROS_PY scripts/run_episode.py --seed 1000 --phase transport --mode bt_safe_abort --policy fake --video
bash scripts/remote/c2_dev.sh 1000 1001 1002                      # dev loop: uninterrupted fake episodes + report
bash scripts/remote/c2_dev.sh --mode bt_safe_abort --phase transport 1000                     # + latency, criteria 1-3
bash scripts/run_matrix.sh --policy fake --seeds 1000-1001 --no-require-paired                # (starts servers)
$G_PY scripts/repro.py --task PickPlaceCounterToStove --n 20 --seeds 1000-1009                # reproduction gate (dev seeds)
bash scripts/remote/c6_score.sh 6      # scoring, first block (eval seeds from 0; needs eval/ uploaded)
python3 scripts/check_floor.py runs/c6_score/results.jsonl    # showcase floor + frozen cfg_hash (eval/prereg.json)
```

Both scripts start missing servers and wait at most 180 s for them (`scripts/remote/servers.sh`; on timeout or an
early exit they print the server log tails). Copy `~/btvla/out` and `~/btvla/runs` off the instance before stopping it.

Fake policy = scripted expert (`policy/fake_server.py`): it reads `data["_priv"]` (SimState + `target_pos` + the
sim_server `priv` block: object and pan bboxes, finger qpos / axis, arm root), which only `run_episode --policy fake`
sends and `seeded_server.py` drops unread. Per-query trace lines go to `<out>/policy_server.log`.

Outputs: `<out>/<key>.jsonl` event log (key `s<seed>_<phase>_<mode>_r<rep>`), `<key>.bt.log`,
`frames_<key>/` (JPEG every 2nd step with `--video`), and one line per episode in `<out>/results.jsonl`
(metrics from `preempt.metrics`, plus layout / object / policy latency / goal and feedback counts).

## Design decisions

1. No ScriptedMotion action: recovery commands ride the lockstep ack (RemoteArbiter), so `bt/tree.py` runs
   unchanged in bt_runner. `tests/test_lockstep.py` replays this protocol on the stub (4 modes × 3 phases)
   and matches the stub results; mutating the op/request order breaks the BT safe-abort cases.
2. bt_runner uses `rclpy.action.ActionClient` with the py_trees_ros call sequence, not py_trees_ros
   behaviours: `tree.VlaSkill` already is the action behaviour, and py_trees_ros drops a cancel issued
   before the goal handle exists (seen in the py_trees_ros action client source). py_trees_ros is installed
   and import-checked only.
3. The gr00t venv also gets robosuite + robocasa (`--no-deps`) so `repro.py` can run the upstream eval loop.
4. Not built: replay of recorded baseline chunks up to t_I (an inference cost saving); with seeded inference
   the prefix is re-inferred.

## Checklist for the first instance runs (written before them)

Install check (g4dn, install + `sim_server --selftest`):
1. `python3.11` from jammy universe (deadsnakes fallback); robocasa `pip install -e` incl. `lerobot==0.3.3`.
2. 12-D action order: selftest runs `preempt.action_layout.check_against_env` on the real env, exit 1 on mismatch.
3. Quaternion order of `state.base_rotation` / `state.end_effector_rotation_relative` is xyzw (yaw, retract).
4. `robot0_base_pos` is world frame; `env.control_freq == 20` (selftest `control_hz_ok`).
5. OSC delta frame, sign, scale and base velocity frame, sign, scale: measured 2026-10-08 (eef 0.0118 m, base
   0.0041 m, yaw 0.046 rad per step per unit; base frame, +x forward) and written into `recovery.Gains`.
   Still unverified: eef rotation and base lateral response (selftest now probes both: `eef_rotvec_*`,
   `base_left_*`) and the `priv` block names (`priv_missing` in the selftest JSON should be empty).
6. Raw MuJoCo access (`env.sim.model._model`, `mj_contactForce`, `body_rootid`), floor-geom detection by
   name/plane: `contacts_at_reset` should be ~0 N with no robot–counter pairs.
7. Determinism: same seed and actions give identical qpos/qvel hashes (`determinism.identical`).
8. Light asset subset passes `--check-assets` on 10 seeds; Box download time; `df` after assets.
9. EGL rendering on the DLAMI and `steps_per_s`; `get_ep_meta` keys `layout_id`, `style_id`,
   `object_cfgs[].info.mjcf_path`; `get_obj_lang()`.
10. ROS apt key/source method; apt `py_trees` is 2.6.x (setup asserts); colcon builds `bt_vla_interfaces`;
    numpy 1.26.4 + torch-cpu 2.5.1 in the ros venv coexist with rclpy.
11. gr00t without tensorflow imports; `pipablepytorch3d` needed or not; flash-attn wheel name for torch 2.5.1 /
    cu12 / cp310; `TORCH_INDEX` cu124 fits the DLAMI driver; gr00t + robocasa (numpy 2.2.5, tianshou
    0.5.1 vs 0.4.10, onnx 1.15) import together for the native leg.
12. HF repo public; checkpoint dir holds `config.json`, `*.safetensors`, `experiment_cfg/` (~7.6 GB).

Scripted-expert runs (g4dn, fake policy over the full ROS chain):
13. rclpy behaviour assumed by skill_host: `goal_callback` gets only the Goal (no id); goal id via
    `handle_accepted_callback`; rcl passes only ACCEPTED/EXECUTING goals to `cancel_callback`; the goal is
    CANCELING before the cancel response is sent, so `canceled()` is valid; `send_goal_async(goal_uuid=)`.
14. Reentrant callback group + MultiThreadedExecutor(8): cancel handled while execute blocks; no deadlock
    across `applied` / `done` waits; reliable QoS delivers every `/bt/state` and `/bt/ack` after matching.
15. Per-step ROS overhead (ms) for the cost model; frames on disk per episode.
16. Scripted expert completes the task uninterrupted on the dev seeds and each phase lasts entry + k + 5
    (`c2_report.py` "injection fits"); T calibration (max abort duration × 2) as planned. First boot: the old
    planner never completed the task (gains 4x off), so transport was never reached.

Reproduction gate (g5, GR00T):
17. Request batch (video (1,1,H,W,3) uint8, state (1,1,d) float32, language as a 1-tuple) is accepted and
    equals the upstream SyncVectorEnv path; reply dict shapes (1,16,d), numpy or tensor.
18. Seeded inference bitwise reproducible on A10G, with and without `--deterministic`.
19. Inference ms / peak GPU memory (`stats` endpoint), RAM on g5.xlarge with sim + policy + native env.
20. Reproduction gate: native ≥ 8/20; ours within 20 pp; per-control-step wall time s.
(5555 is taken by nv-hostengine / NVIDIA DCGM on the DLAMI, measured 2026-10-08.)
