from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CSPEnv(RL4COEnvBase):
    """
    Covering Salesman Problem environment
    At each step, the agent chooses a city to visit. 
    The reward is 0 unless all the cities are visited or be covered by at least 1 city on the tour.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Args:
        num_loc: number of locations (cities) in the CSP
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "csp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_cover: float = 0.1,
        max_cover: float = 0.3,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_cover = min_cover
        self.max_cover = max_cover
        self._make_spec(td_params)

    @staticmethod
    def get_covered_guidence_vec(covered_node_bool, curr_distance):
        '''
        covered_node: [batch, num_loc], torch.bool
        curr_distance: curr node i, distance to all other nodes, [batch, num_loc], torch.bool
        
        '''
        curr_distance[~covered_node_bool] =  10
        i, idx = curr_distance.sort(dim=-1)      # [batch, num_loc]
        _, distance_sort_idx = idx.sort(dim=-1)
        covered_sort = torch.where(covered_node_bool, distance_sort_idx + 1,
                                   torch.zeros_like(covered_node_bool))
        curr_covered_num = covered_node_bool.sum(dim=-1)     # [batch, ]
        return covered_sort / (curr_covered_num.unsqueeze(-1) + 1e-5)
        
        
        
    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # get covered node of current node
        locs = td["locs"]       # [batch, num_loc,2]
        curr_locs = locs.gather(-2, current_node[None, ..., None].repeat(1,1,2)).squeeze(0).unsqueeze(-2)     # [batch, 2]
        curr_dist = (locs - curr_locs).norm(p=2, dim=-1)        #[batch, num_loc]
        # covered_by_curr_node = torch.nonzero([curr_dist < self.min_cover])     # [batch, satisfy num]
        # uncertain_coverd_by_curr_node = torch.nonzero([curr_dist > self.min_cover] & [curr_dist < self.max_cover])
        # uncertrain_prob = func(curr_dist, uncertain_coverd_by_curr_node)        # []
        
        covered_node = curr_dist < self.min_cover
        td["covered_node"][covered_node] = 1       #[batch, num_loc]
        curr_covered_guidence_vec = CSPEnv.get_covered_guidence_vec(covered_node, curr_dist.clone())
        td["guidence_vec"] *= torch.where(covered_node, curr_covered_guidence_vec,
                                                        torch.ones_like(covered_node))
        
        # # Set not visited to 0 (i.e., we visited the node)
        # in csp, covered node still can be visited
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )   # 将current node值作为索引， 使对应的action mask为0

        # We are done when all nodes are visited
        done = (torch.sum(td["covered_node"], dim=-1) == self.num_loc) | (torch.sum(available, dim=-1) == 0)
        
        # # set complete solution: the first node's action_mask to False to avoid softmax(all -inf)=> nan error
        done_idx = torch.nonzero(done.squeeze())  #[batch]
        done_endnode = current_node[done_idx]   #[batch]
        # first set the done data's action_mask all to False
        available[done_idx, :] = False
        # then set its first node to True
        available[done_idx, done_endnode] = True
        
        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "guidence_vec": td["guidence_vec"],
                "covered_node": td["covered_node"],
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        init_locs = td["locs"] if td is not None else None
        if batch_size is None:
            batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]
        device = init_locs.device if init_locs is not None else self.device
        self.to(device)
        if init_locs is None:
            init_locs = self.generate_data(batch_size=batch_size).to(device)["locs"]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # We do not enforce loading from self for flexibility
        num_loc = init_locs.shape[-2]

        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        covered_node = torch.zeros(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means covered, i.e. but action is still allowed
        
        guidence_vec = torch.ones(
            (*batch_size, num_loc), dtype=torch.float64, device=device
        )  # update while decoding, init 1. covered then decrease
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": init_locs,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "covered_node": covered_node,
                "guidence_vec": guidence_vec,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params):
        """Make the observation and action specs from the parameters"""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc),
                dtype=torch.bool,
            ),
            covered_node=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc),
                dtype=torch.int64,
            ),
            guidence_vec=UnboundedContinuousTensorSpec(
                shape=(self.num_loc),
                dtype=torch.float64,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def get_reward(self, td, actions) -> TensorDict:
        # if self.check_solution:
        #     self.check_solution_validity(td, actions)

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_tour_length(locs_ordered)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are all visited or covered"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"
            
    def generate_data(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        locs = (
            torch.FloatTensor(*batch_size, self.num_loc, 2)
            .uniform_(self.min_loc, self.max_loc)
            ).to(self.device)
        
        
        
        return TensorDict({"locs": locs}, batch_size=batch_size)

