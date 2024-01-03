from rl4co.utils.pylogger import get_pylogger
import torch
from .cvrp import CVRPEnv
from torch.nn.utils.rnn import pad_sequence
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance
from tensordict.tensordict import TensorDict
from typing import Optional
from rl4co.data.utils import load_npz_to_tensordict
import time
import torch.multiprocessing as mp
from memory_profiler import profile
import gc
import sys
log = get_pylogger(__name__)
from guppy import hpy
# From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023
CAPACITIES = {
    10: 20.0,
    15: 25.0,
    20: 30.0,
    30: 33.0,
    40: 37.0,
    50: 40.0,
    60: 43.0,
    75: 45.0,
    100: 50.0,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}

class SVRPEnv(CVRPEnv):
    """Stochastic Vehicle Routing Problem (CVRP) environment.

    Note:
        The only difference with deterministic CVRP is that the demands are stochastic
        (i.e. the demand is not the same as the real prize).
    """

    name = "svrp"       # class variable
    _stochastic = True
    generate_method = "modelize" 
    

    def __init__(self, generate_method = "modelize", **kwargs):
        super().__init__(**kwargs)
        
        self.generate_method = generate_method
        assert self.generate_method in ["uniform", "modelize"], "way of generate stochastic data is invalid"

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand and stochastic_demand by capacity to be in [0, 1]
        """
        td_load = load_npz_to_tensordict(fpath)
        td_load.set("demand", td_load["demand"] / td_load["capacity"][:, None])
        td_load.set("stochastic_demand", td_load["stochastic_demand"] / td_load["capacity"][:, None])
        return td_load
    
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
        # get the solution's penaltied idx
        loc_idx_penaltied = self.get_penalty_loc_idx(td, actions)   #[batch, num-customer]
        # Gather dataset in order of tour
        depot = td["locs"][..., 0:1, :]
        depot_batch = depot.repeat(1, loc_idx_penaltied.size(1), 1) # [batch,  num_customer, 2]
        
        # get penaltied lcoations
        locs_penalty = td["locs"][..., 1:, :]* loc_idx_penaltied[..., None]       #[batch, num_customer, 2]
        # get 0 pad mask
        posit = loc_idx_penaltied > 0  #[batch, num_customer]
        posit = posit[:,:, None]    #[batch, num_customer, 1]
        posit = posit.repeat(1, 1, 2)   #[batch, num_customer, 2]
        locs_penalty = torch.where(posit, locs_penalty, depot_batch)
        
        locs_ordered = torch.cat([depot, gather_by_index(td["locs"], actions)], dim=1)
        cost_orig = -get_tour_length(locs_ordered)
        cost_penalty = -get_distance(depot_batch, locs_penalty).sum(-1) * 2
        # print(f"new version: orig is {cost_orig}, penalty is {cost_penalty}")
        
        return cost_penalty + cost_orig
    
    
            
    
    @staticmethod
    def get_penalty_loc_idx(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are not visited twice except depot.
        if capacity is not exceeded, record the loc idx
        return penaltied location idx, [batch, penalty_number]
        
        review:
            return penaltied_idx: [batch, num_customer], if node is penaltied, set 1
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

        
        start_3 = time.time()
        used_cap = torch.zeros((td["demand"].size(0), 1), device=td["demand"].device)
        penaltied_idx = torch.zeros_like(td["demand"], device=td["demand"].device)      # [batch, num_customer]
        for i in range(actions.size(1)):
            used_cap[:, 0] += d[
                    :, i
                ]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            exceed_cap_bool = used_cap[:, 0] > td["vehicle_capacity"][:,0] + 1e-5        # 1 dim
            if any(exceed_cap_bool):
                # print("Used more than capacity")
                exceed_idx = torch.nonzero(exceed_cap_bool)     # [exceed_data_idx, 1]
                penaltied_node = actions[exceed_idx, i] - 1     # [exceed_data_idx, 1], in"actions",customer node start from 1, substract 1 when be idx
                penaltied_idx[exceed_idx, penaltied_node] = 1        # set exceed idx to 1
                used_cap[exceed_idx, 0] = d[exceed_idx, i]
        end_3 = time.time()
        # print(f"time of one loop: {end_3 - start_3}")
        return penaltied_idx
    
    # @profile(stream=open('log_mem_svrp_generate_data.log', 'w+'))  
    def generate_data(self, batch_size, ) -> TensorDict:
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize the demand for nodes except the depot
        # Demand sampling Following Kool et al. (2019)
        # Generates a slightly different distribution than using torch.randint
        demand = (
            torch.FloatTensor(*batch_size, self.num_loc)
            .uniform_(self.min_demand - 1, self.max_demand - 1)
            .int()
            + 1
        ).float().to(self.device)

        # Initialize the weather
        weather = (
            torch.FloatTensor(*batch_size, 3)
            .uniform_(-1, 1)
        ).to(self.device)
        
        #  E(stochastic demand) = E(demand)
        if self.generate_method == "uniform":
            # print(f"generate data by uniform")
            stochastic_demand = (
                torch.FloatTensor(*batch_size, self.num_loc)
                .uniform_(self.min_demand - 1, self.max_demand - 1)
                .int()
                + 1
            ).float().to(self.device)
        elif self.generate_method == "modelize":
            # alphas = torch.rand((n_problems, n_nodes, 9, 1))      # =np.random.random, uniform dis(0, 1)

            stochastic_demand = self.get_stoch_var(demand.to("cpu"),
                                                   locs_with_depot[..., 1:, :].to("cpu").clone(), 
                                                   weather[:, None, :].
                                                   repeat(1, self.num_loc, 1).to("cpu"),
                                                   None).squeeze(-1).float().to(self.device)

        # Support for heterogeneous capacity if provided
        if not isinstance(self.capacity, torch.Tensor):
            capacity = torch.full((*batch_size,), self.capacity, device=self.device)
        else:
            capacity = self.capacity

        
        
        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "demand": demand / CAPACITIES[self.num_loc],        # normalize demands
                "stochastic_demand": stochastic_demand / CAPACITIES[self.num_loc],
                "weather": weather,
                "capacity": capacity,       # =1
            },
            batch_size=batch_size,
            device=self.device,
        )

    @staticmethod
    # @profile(stream=open('logmem_svrp_sto_gc_tocpu4.log', 'w+'))
    def get_stoch_var(inp, locs, w, alphas=None, A=0.6, B=0.2, G=0.2):
        '''
        locs: [batch, num_customers, 2]
        '''
        # h = hpy().heap()
        if inp.dim() <= 2:
            inp_ =  inp[..., None]
        else:
            inp_ = inp.clone()

        n_problems,n_nodes,shape = inp_.shape
        T = inp_/A

        # var_noise = T*G
        # noise = torch.randn(n_problems,n_nodes, shape).to(T.device)      #=np.rand.randn, normal dis(0, 1)
        # noise = var_noise*noise     # multivariable normal distr, var_noise mean
        # noise = torch.clamp(noise, min=-var_noise)
        
        var_noise = T*G

        noise = var_noise*torch.randn(n_problems,n_nodes, shape).to(T.device)      #=np.rand.randn, normal dis(0, 1)
        noise = torch.clamp(noise, min=-var_noise)

        var_w = T*B
        # sum_alpha = var_w[:, :, None, :]*4.5      #? 4.5
        sum_alpha = var_w[:, :, None, :]*4.5      #? 4.5
        
        if alphas is None:  
            alphas = torch.rand((n_problems, 1, 9, shape)).to(T.device)       # =np.random.random, uniform dis(0, 1)
        alphas_loc = locs.sum(-1)[..., None, None] * alphas  # [batch, num_loc, 2]-> [batch, num_loc] -> [batch, num_loc, 1, 1], [batch, 1, 9,1]
            # alphas = torch.rand((n_problems, n_nodes, 9, shape)).to(T.device)       # =np.random.random, uniform dis(0, 1)
        alphas_loc.div_(alphas_loc.sum(axis=2)[:, :, None, :])       # normalize alpha to 0-1
        alphas_loc *= sum_alpha     # alpha value [4.5*var_w]
        alphas_loc = torch.sqrt(alphas_loc)        # alpha value [sqrt(4.5*var_w)]
        signs = torch.rand((n_problems, n_nodes, 9, shape)).to(T.device) 
        # signs = torch.where(signs > 0.5)
        alphas_loc[torch.where(signs > 0.5)] *= -1     # half negative: 0 mean, [sqrt(-4.5*var_w) ,s sqrt(4.5*var_w)]
        
        w1 = w.repeat(1, 1, 3)[..., None]       # [batch, nodes, 3*repeat3=9, 1]
        # roll shift num in axis: [batch, nodes, 3] -> concat [batch, nodes, 9,1]
        w2 = torch.concatenate([w, torch.roll(w,shifts=1,dims=2), torch.roll(w,shifts=2,dims=2)], 2)[..., None]
        
        tot_w = (alphas_loc*w1*w2).sum(2)       # alpha_i * wm * wn, i[1-9], m,n[1-3], [batch, nodes, 9]->[batch, nodes,1]
        tot_w = torch.clamp(tot_w, min=-var_w)
        out = inp_ + tot_w + noise
        
        # del tot_w, noise
        del var_noise, sum_alpha, alphas_loc, signs, w1, w2, tot_w
        del T, noise, var_w
        del inp_
        gc.collect()
        
        return out
    
    def reset_stochastic_demand(self, td, alpha):
        '''
        td is state of env, after calla reset()
        alpha: [batch, 9, 1]
        '''
        
        # reset real demand from weather
        batch_size = td["demand"].size(0)
        if self.generate_method == "uniform":
            # print(f"generate data by uniform")
            stochastic_demand = (
                torch.FloatTensor(*batch_size, self.num_loc)
                .uniform_(self.min_demand - 1, self.max_demand - 1)
                .int()
                + 1
            ).float().to(self.device)
        elif self.generate_method == "modelize":
            # alphas = torch.rand((n_problems, n_nodes, 9, 1))      # =np.random.random, uniform dis(0, 1)

            locs_cust = td["locs"].clone()
            locs_cust = locs_cust[:, 1:, :]
            stochastic_demand = self.get_stoch_var(td["demand"].to("cpu"),
                                                   locs_cust.to("cpu"), 
                                                   td["weather"][:, None, :].
                                                   repeat(1, self.num_loc, 1).to("cpu"),
                                                   alpha[:, None, ...].to("cpu")).squeeze(-1).float().to(self.device)

        td.set("real_demand", stochastic_demand)
        
        return td
    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]  # Add dimension for step
        n_loc = td["demand"].size(-1)  # Excludes depot

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["real_demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )

        # Increase capacity if this time depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td
    
    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # cannot mask exceeding node in svrp
        # exceeds_cap = td["demand"][:, None, :] + td["used_capacity"][..., None] > 1.0

        # Nodes that cannot be visited are already visited
        mask_loc = td["visited"][..., 1:].to(torch.bool)

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)
    
    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.to(td.device)

        # Create reset TensorDict
        real_demand = (
            td["stochastic_demand"] 
        )
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][:, None, :], td["locs"]), -2),
                "weather": td["weather"],
                "demand": td["demand"], # observed demand
                "real_demand": real_demand,
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=self.device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.vehicle_capacity, device=self.device
                ),
                "visited": torch.zeros(
                    (*batch_size, 1, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=self.device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset