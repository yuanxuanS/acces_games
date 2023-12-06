from rl4co.utils.pylogger import get_pylogger
import torch
from .cvrp import CVRPEnv
from torch.nn.utils.rnn import pad_sequence
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance
from tensordict.tensordict import TensorDict

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

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Check that the solution is valid
        loc_idx_penaltied = self.get_penalty_loc_idx(td, actions)   #[batch, penalty_number]

        # Gather dataset in order of tour
        depot = td["locs"][..., 0:1, :]
        depot_batch = depot.repeat(1, loc_idx_penaltied.size(1), 1) # [batch,  penalty_number, 2]
        
        # get penaltied lcoations
        locs_penalty = gather_by_index(td["locs"], loc_idx_penaltied)       #[batch, penalty_number, 2]
        # get 0 pad mask
        posit = loc_idx_penaltied > 0  #[batch, penalty number]
        posit = posit[:,:, None]    #[batch, penalty number, 1]
        posit = posit.repeat(1, 1, 2)   #[batch, penalty number, 2]
        locs_penalty = torch.where(posit, locs_penalty, depot_batch)
        
        locs_ordered = torch.cat([depot, gather_by_index(td["locs"], actions)], dim=1)
        cost_orig = -get_tour_length(locs_ordered)
        cost_penalty = -get_distance(depot_batch, locs_penalty).sum(-1) * 2
        # print(f"orig is {cost_orig}, penalty is {cost_penalty}")
        return cost_penalty + cost_orig

    @staticmethod
    def get_penalty_loc_idx(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are not visited twice except depot.
        if capacity is not exceeded, record the loc idx
        return penaltied location idx, [batch, penalty_number]
        """
        # Check if tour is valid, i.e. contain 0 to n-1
        batch_size, graph_size = td["demand"].size()
        sorted_pi = actions.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        real_demand_with_depot = torch.cat((-td["vehicle_capacity"], td["real_demand"]), 1)
        d = real_demand_with_depot.gather(1, actions)

        batch_penalty_loc_idx = []
        used_cap = torch.zeros((td["demand"].size(0), 1), device=td["demand"].device)
        for b in range(actions.size(0)):
            penalty_loc_idx = []
            for i in range(actions.size(1)):
                used_cap[b, 0] += d[
                    b, i
                ]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
                used_cap[used_cap < 0] = 0
                if used_cap[b, 0] > td["vehicle_capacity"][b,] + 1e-5:
                    # print("Used more than capacity")
                    used_cap[b, 0] = d[b, i]
                    penalty_loc_idx.append(actions[b, i])
            batch_penalty_loc_idx.append(penalty_loc_idx)
            
        loc_idx_penalty = pad_sequence([torch.tensor(sublist, device=td["demand"].device, 
                                                   dtype=torch.int64) 
                                      for sublist in batch_penalty_loc_idx], 
                                     batch_first=True, padding_value=0)

        return loc_idx_penalty
     