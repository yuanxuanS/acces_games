from rl4co.utils.pylogger import get_pylogger

from .cvrp import CVRPEnv

log = get_pylogger(__name__)

class SVRPEnv(CVRPEnv):
    """Stochastic Vehicle Routing Problem (CVRP) environment.

    Note:
        The only difference with deterministic CVRP is that the demands are stochastic
        (i.e. the demand is not the same as the real prize).
    """

    name = "svrp"
    _stochastic = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def stochastic(self):
        return self._stochastic

    @stochastic.setter
    def stochastic(self, state: bool):
        if state is False:
            log.warning(
                "Deterministic mode should not be used for SVRP. Use CVRP instead."
            )
