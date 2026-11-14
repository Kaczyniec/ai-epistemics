"""perfcast -- do LLM forecasters notice that their forecasts move the series?

A live performative-prediction environment: a time series is shown to a model
round by round, the model forecasts the next value, and the forecast silently
changes that value through a fixed rule. The question is **when** the model
notices, measured four ways that are kept strictly apart:

behavioural position (``z``)
    where the forecast sits between the adaptive-but-unaware path and the
    performative fixed point;
performative stability
    whether the forecast has reached the fixed point, and whether it got there
    by a geometric approach (retraining -- no awareness needed) or a jump;
probes
    a forecast deliberately far from the model's own best estimate, the one
    behavioural signature passive learning cannot produce;
verbal
    what the model says, scored 0-5 by an LLM judge.

Converging to the fixed point is *not* evidence of awareness (Perdomo et al.,
ICML 2020). Awareness shows off-equilibrium.
"""

from .env import Environment, RunSpec
from .influence import make_rule
from .oracle import run_oracle
from .paths import reference_paths
from .process import ProcessConfig, ProcessRanges
from .runner import run_episode

__all__ = [
    "Environment",
    "RunSpec",
    "ProcessConfig",
    "ProcessRanges",
    "make_rule",
    "reference_paths",
    "run_oracle",
    "run_episode",
]
__version__ = "0.1.0"
