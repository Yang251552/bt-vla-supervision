"""Skill-level test rig: stub env + fake policy + arbiter + skill controller, no BT."""
from preempt.arbiter import Arbiter
from preempt.contracts import ProtocolConfig
from preempt.skill import SkillController
from preempt.stub import FakePolicy, StubEnv


class ListLog:
    def __init__(self):
        self.recs = []

    def __call__(self, ev, t, **fields):
        self.recs.append({"ev": ev, "t": t, **fields})

    def of(self, ev, **match):
        return [r for r in self.recs if r["ev"] == ev and all(r.get(k) == v for k, v in match.items())]

    def index(self, ev, **match):
        return self.recs.index(self.of(ev, **match)[0])


class Rig:
    def __init__(self, cfg=ProtocolConfig(), seed=0, latency=0, lockstep=True, obey_stop=True):
        self.cfg = cfg
        self.log = ListLog()
        self.env = StubEnv(seed)
        self.policy = FakePolicy(cfg.chunk_len, latency, obey_stop)
        self.arb = Arbiter(self.log)
        self.skill = SkillController(cfg, self.policy, self.arb, self.log, lockstep=lockstep)
        self.applied = []  # (t, owner, action)

    def run(self, t0, t1, at=None):
        """Steps t0..t1-1; `at` maps a step to a callable run in the supervisor-tick slot."""
        at = at or {}
        for t in range(t0, t1):
            self.skill.pre_step(t, self.env.state(), obs=self.env)
            if t in at:
                at[t](t)
            self.skill.act(t)
            a, owner = self.arb.resolve(t)
            self.applied.append((t, owner, a))
            self.env.step(a)

    def vla_steps(self, gen, after=0):
        return [t for t, o, _ in self.applied if o == ("vla", gen) and t >= after]
