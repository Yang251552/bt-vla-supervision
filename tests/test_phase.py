import numpy as np

from preempt.contracts import Phase, ProtocolConfig, SimState
from preempt.phase import PhaseDetector

CFG = ProtocolConfig()


def st(dist=0.5, grasped=False, lift=0.0, success=False):
    obj = np.array([0.7, 0.0, 0.03])
    z = np.zeros(3)
    return SimState(z, np.array([0, 0, 0, 1.0]), obj + [dist, 0, 0], z, obj, lift, grasped,
                    0.0, 0.0, 0.0, success)


def feed(det, states, t0=0):
    return [det.update(t0 + i, s) for i, s in enumerate(states)]


def test_approach_to_grasp_needs_five_consecutive_steps_and_never_goes_back():
    d = PhaseDetector(CFG)
    feed(d, [st(0.05)] * 4 + [st(0.2)])
    assert d.phase == Phase.APPROACH                       # 4 steps then a miss: no switch
    feed(d, [st(0.05)] * 5, t0=5)
    assert d.phase == Phase.GRASP and d.entered[Phase.GRASP] == 9
    feed(d, [st(0.5)] * 20, t0=10)
    assert d.phase == Phase.GRASP                          # monotonic latch


def test_transport_needs_grasped_and_lifted_two_cm():
    d = PhaseDetector(CFG)
    feed(d, [st(0.0)] * 5)
    feed(d, [st(0.0, grasped=True, lift=0.019)] * 20, t0=5)
    assert d.phase == Phase.GRASP
    feed(d, [st(0.0, grasped=False, lift=0.05)] * 20, t0=25)
    assert d.phase == Phase.GRASP                          # lifted but not grasped
    feed(d, [st(0.0, grasped=True, lift=0.02)] * 5, t0=45)
    assert d.phase == Phase.TRANSPORT and d.entered[Phase.TRANSPORT] == 49


def test_transport_ends_on_release_or_success():
    d = PhaseDetector(CFG)
    feed(d, [st(0.0)] * 5 + [st(0.0, grasped=True, lift=0.05)] * 5)
    feed(d, [st(0.0, grasped=False, lift=0.05)] * 4 + [st(0.0, grasped=True, lift=0.05)], t0=10)
    assert d.phase == Phase.TRANSPORT                      # brief slip does not end transport
    feed(d, [st(0.0, grasped=True, lift=0.0, success=True)] * 5, t0=15)
    assert d.phase == Phase.DONE
