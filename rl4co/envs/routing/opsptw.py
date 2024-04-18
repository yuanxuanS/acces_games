from typing import Optional
from typing import Iterable, Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torch import Tensor
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.data.dataset import TensorDictDataset

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.pylogger import get_pylogger
import random
import numpy as np
log = get_pylogger(__name__)


class OPSPTWEnv(RL4COEnvBase):
    """
    Stochastic weights time window Traveling Salesman Problem environment
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent back to depot .
    In that case, the reward is cumulative prize
    The expected travel time is distance. 
    The prize is uncertain. influenced by weather
    Args:
        num_loc: number of locations (cities) in the TSP
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """
    name = "opsptw"

    def __init__(
        self,
        num_loc: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc


    def dataset(self, batch_size=[], phase="train", filename=None):
        """Return a dataset of observations
            in opswtw, load data from presaved_data
        """
        
        if phase == "train":
            filename = "/home/panpan/rl4co/data/opswtw_train/opswtw_train.npz"
        elif phase == "val" or "test":
            filename = "/home/panpan/rl4co/data/opswtw/opswtw_val.npz"
        f = filename
        log.info(f"Loading {phase} dataset from {f}")
        try:
            if isinstance(f, Iterable) and not isinstance(f, str):
                names = getattr(self, f"{phase}_dataloader_names")
                return {
                    name: TensorDictDataset(self.load_data(_f, batch_size))
                    for name, _f in zip(names, f)
                }
            else:
                td = self.load_data(f, batch_size)
        except FileNotFoundError:
            log.error(
                f"Provided file name {f} not found. Make sure to provide a file in the right path first or "
                f"unset {phase}_file to generate data automatically instead"
            )
            td = self.generate_data(batch_size)

        return TensorDictDataset(td)
    

    def load_data(self, fpath, batch_size=[]):
        """Dataset loading from file
        Normalize data
        opswtw data: test and val data are the same 
        """
        if not batch_size:
            batch_size = 10000
        if isinstance(batch_size, list):
            batch_size = batch_size[0]
        td_load = load_npz_to_tensordict(fpath)[:batch_size, ...]
        locs = td_load["locs"]
        locs[..., 0] /= 200
        locs[..., 1] /= 50
        td_load.set("locs", locs)
        td_load.set("tw_low", td_load["tw_low"][:batch_size, ...] / 100)
        td_load.set("tw_high", td_load["tw_high"][:batch_size, ...] / 100)
        td_load.set("maxtime", td_load["maxtime"][:batch_size, ...] / 100)
        prize = td_load["prize"][:batch_size, ...]
        td_load.set("prize", prize)
        
        # 
        stoch_gamma = (
            torch.FloatTensor(batch_size, self.num_loc)
            .uniform_(0, 1)
        ).to(self.device)
        # td_load.set("stoch_gamma", stoch_gamma)

        # weather
        weather = (
            torch.FloatTensor(batch_size, 3)
            .uniform_(-1, 1)
        ).to(self.device)
        td_load.set("weather", weather)
        
        stochastic_prize = self.get_stoch_var(prize.to("cpu"), 
                                              locs.to("cpu").clone(),
                                              weather[:, None,:].repeat(1, self.num_loc,1).to("cpu"),
                                              None).squeeze(-1).float().to(self.device)
        td_load.set("stochastic_prize", stochastic_prize)
        return td_load
    
    def generate_data(self, batch_size) -> TensorDict:
        # 返回归一化的数据
        data_pth = "/home/panpan/rl4co/data/opswtw_train/opswtw_train.npz"
        # noise_pth = "/home/panpan/rl4co/data/opswtw_train/noise_train.npz"

        x_dict = self.load_data(data_pth, batch_size)
        whole_size = x_dict.batch_size[0]

        if isinstance(batch_size, list):
            batch_size = batch_size[0]
        assert whole_size >= batch_size, "Required size exceeds data size"

        locs = x_dict["locs"][:batch_size, ...]
        tw_low = x_dict["tw_low"][:batch_size, ...]
        tw_high = x_dict["tw_high"][:batch_size, ...]
        prize = x_dict["prize"][:batch_size, ...]       # has been normalized
        maxtime = x_dict["maxtime"][:batch_size, ...]
        # stoch_gamma = x_dict["stoch_gamma"][:batch_size, ...]
        weather = x_dict["weather"][:batch_size, ...]
        stoch_prize = x_dict["stochastic_prize"][:batch_size, ...]
        return TensorDict(
            {
            "locs": locs,
            "tw_low": tw_low,
            "tw_high": tw_high,
            "prize": prize,
            "stochastic_prize": stoch_prize,
            # "stoch_gamma": stoch_gamma,
            "weather": weather,
            "maxtime": maxtime,
        },
        batch_size=batch_size,
        device=self.device
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

        noise = torch.sqrt(var_noise)*torch.randn(n_problems,n_nodes, shape).to(T.device)      #=np.rand.randn, normal dis(0, 1)
        noise = torch.clamp(noise, min=-var_noise, max=var_noise)

        var_w = torch.sqrt(T*B)
        # sum_alpha = var_w[:, :, None, :]*4.5      #? 4.5
        sum_alpha = var_w[:, :, None, :]*9      #? 4.5
        
        if alphas is None:  
            alphas = torch.rand((n_problems, 1, 9, shape)).to(T.device)       # =np.random.random, uniform dis(0, 1)
        alphas_loc = locs.sum(-1)[..., None, None]/2 * alphas  # [batch, num_loc, 2]-> [batch, num_loc] -> [batch, num_loc, 1, 1], [batch, 1, 9,1]
            # alphas = torch.rand((n_problems, n_nodes, 9, shape)).to(T.device)       # =np.random.random, uniform dis(0, 1)
        # alphas_loc.div_(alphas_loc.sum(axis=2)[:, :, None, :])       # normalize alpha to 0-1
        alphas_loc *= sum_alpha     # alpha value [4.5*var_w]
        alphas_loc = torch.sqrt(alphas_loc)        # alpha value [sqrt(4.5*var_w)]
        signs = torch.rand((n_problems, n_nodes, 9, shape)).to(T.device) 
        # signs = torch.where(signs > 0.5)
        alphas_loc[signs > 0.5] *= -1     # half negative: 0 mean, [sqrt(-4.5*var_w) ,s sqrt(4.5*var_w)]
        # alphas_loc[torch.where(signs > 0.5)] *= -1     # half negative: 0 mean, [sqrt(-4.5*var_w) ,s sqrt(4.5*var_w)]
        
        w1 = w.repeat(1, 1, 3)[..., None]       # [batch, nodes, 3*repeat3=9, 1]
        # roll shift num in axis: [batch, nodes, 3] -> concat [batch, nodes, 9,1]
        w2 = torch.concatenate([w, torch.roll(w,shifts=1,dims=2), torch.roll(w,shifts=2,dims=2)], 2)[..., None]
        
        tot_w = (alphas_loc*w1*w2).sum(2)       # alpha_i * wm * wn, i[1-9], m,n[1-3], [batch, nodes, 9]->[batch, nodes,1]
        tot_w = torch.clamp(tot_w, min=-var_w, max=var_w)
        out = torch.clamp(inp_ + tot_w + noise, min=0.01)
        
        # del tot_w, noise
        del var_noise, sum_alpha, alphas_loc, signs, w1, w2, tot_w
        del T, noise, var_w
        del inp_
        # gc.collect()
        
        return out
    
    
    @staticmethod
    def get_adjacent_matrix(x: Tensor, y: Tensor):
        return (x - y).norm(p=2, dim=-1)

    def _reset(self, 
               td: Optional[TensorDict] = None,
               batch_size: Optional[list] = None,
               ) -> TensorDict:
        # reset为状态
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size


        if td is None:
            td = self.generate_data(batch_size)
        
        locs = td["locs"]
        curr_time = torch.zeros((*batch_size, ), device=self.device)    # batch,
        curr_prize = torch.zeros((*batch_size,), device=self.device)
        curr_node = torch.zeros((*batch_size, ), device=self.device, dtype=torch.int64)
        row_locs = locs[..., None, :].repeat(1, 1, self.num_loc, 1)
        col_locs = locs[..., None, :].repeat(1, 1, self.num_loc, 1).transpose(1, 2)
        adj = OPSPTWEnv.get_adjacent_matrix(row_locs, col_locs)     # batch, size, size
        penalty = torch.zeros((*batch_size, ), device=self.device)
        # 从depot出发，接下来不能拜访depot
        action_mask = torch.ones((*batch_size, self.num_loc), device=self.device).bool()    # True if visted
        action_mask[:, 0] = False       # mask depot
        # 不能拜访 low_tw > maxtime的node
        action_mask[td["tw_low"] > td["maxtime"]] = False

        visited = torch.zeros((*batch_size, self.num_loc), dtype=torch.uint8, device=self.device)
        visited[:, 0] = 1
        done = torch.zeros((*batch_size,), device=self.device, dtype=torch.bool)
        # 初始的
        # real_node_prize = 

        td_reset = TensorDict(
            {
                "locs": locs,
                "tw_low": td["tw_low"],
                "tw_high": td["tw_high"],
                "prize": td["prize"],
                # "real_node_prize": td["prize"].clone(),     # change with arrived time.
                "curr_prize": curr_prize,
                # "stoch_gamma": td["stoch_gamma"],
                "stochastic_prize": td["stochastic_prize"],
                "weather": td["weather"],
                "maxtime": td["maxtime"],
                "tour_time": curr_time,
                "adj": adj,
                "penalty": penalty,
                "action_mask": action_mask,
                "current_node": curr_node,
                "visited": visited,
                "done": done
            },
            batch_size=batch_size,

        )
        return td_reset
    
    def _step(self, td: TensorDict) -> TensorDict:
        
        last_node = td["current_node"]
        current_node = td["action"]     # 1-size， 0是depot
        batch_size = last_node.shape[0]
        # update visited node
        visited = td["visited"]
        visited[range(batch_size), current_node] = 1
        # update tour time 
        tour_time = td["tour_time"]
        adj = td["adj"]
        
        time = adj[range(batch_size), last_node, current_node]      # batch, 
        
        tour_time += time

        
        curr_tw_low = td["tw_low"][range(batch_size), current_node]
        curr_tw_high = td["tw_high"][range(batch_size), current_node]
        # add wait time if not in time window's low thres
        wait_idx = tour_time < curr_tw_low      # batch,  bool
        waited_time = curr_tw_low - tour_time
        tour_time += waited_time * wait_idx
        # penalty - 1 if exceeds the high thres
        penalty = td["penalty"]
        curr_penalty = tour_time >= curr_tw_high
        penalty -= curr_penalty.float()
        # penalty if exceeds maxtime
        penalty_solution = tour_time > td["maxtime"][:, 0]  # 每条数据每个节点的maxtime都相同
        penalty -= (self.num_loc * penalty_solution) * (~td["done"].squeeze(-1))
        done = penalty_solution[..., None]     # if exceed maxtime, solution done
        # if curr node = 0, and this is second time to depot, done=True
        back_to_depot = (current_node==0) & (visited[:, 1:].int().sum(-1) > 0)
        back_to_depot_idx = torch.nonzero(back_to_depot)
        done[back_to_depot_idx] = True
        back_to_depot = self.check_if_back(td, current_node, tour_time)
        done[back_to_depot] = True
        # add prize
        curr_prize = td["curr_prize"]
        # real prize: prize * (3*stoch_gamma^tour_time)
        # stoch_gamma = td["stoch_gamma"][range(batch_size), current_node]
        # real_prize = td["prize"][range(batch_size), current_node] * (stoch_gamma ** td["tour_time"]) * 3
        # real_node_prize = td["real_node_prize"]
        # real_node_prize[range(batch_size), current_node] = real_prize.clone()
        real_prize = td["stochastic_prize"][range(batch_size), current_node]
        curr_prize += real_prize



        td.update(
            {
                "current_node": current_node,
                "visited": visited,
                "tour_time": tour_time,
                "penalty": penalty,
                # "real_node_prize": real_node_prize,
                "curr_prize": curr_prize,
                "done": done,

            }
        )
        td.set("action_mask", self.get_action_mask(td))
        
        return td
    
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        # 如果node不能选， mask_loc=False

        # Nodes that cannot be visited are already visited
        mask_loc = ~td["visited"].to(torch.bool)    # 能选=true
        # 结束的instance 都为False
        done_idx = torch.nonzero(td["done"].squeeze(-1))
        done_node_mask = torch.zeros((self.num_loc)).to(torch.bool)
        mask_loc[done_idx, :] = done_node_mask.to(mask_loc.device)
        # 不能拜访 tour time > high_tw的node
        tourtime_exceed_hightw = td["tour_time"][..., None].repeat(1, self.num_loc) >  td["tw_high"]
        mask_loc[tourtime_exceed_hightw] = False
        ## 未结束的，depot可以拜=True # 结束的instance仅depot=True # 总之，depot都为True
        mask_loc[range(td["visited"].shape[0]), 0] = True    
        
        return mask_loc

    def check_if_back(self, td, current_node, tour_time):
        ## 不能回到depot的都结束done
        adj = td["adj"]
        curr_node = current_node
        distance_back_to_depot = adj[range(td["visited"].shape[0]), curr_node, 0]
        tour_time = tour_time
        back_to_depot = (tour_time + distance_back_to_depot) > td["maxtime"][:, 0]
        return back_to_depot

    def get_reward(self, td: TensorDict, actions: Tensor) -> TensorDict:
        # 所有的prize+ penalty（<0）
        reward = td["curr_prize"] + td["penalty"]
        return reward

    def reset_stochastic_gamma(self, td, adver_action):
        '''
        adver_action: batch ,9
        '''
        assert adver_action.ndim == 3, "adver action dim wrong"
        stoch_gamma = 2*td["prize"] * adver_action.mean(1)      # uniform(0,1)，为了实现0.5均值，乘2
        td.set("stoch_gamma", stoch_gamma)
        return td
    
    def reset_stochastic_prize(self, td, adver_action):
        '''
        adver_action: batch ,9
        '''
        batch_size = td["prize"].size(0)
        locs_cust = td["locs"].clone()
        stochastic_prize = self.get_stoch_var(td["prize"].to("cpu"),
                                                locs_cust.to("cpu"), 
                                                td["weather"][:, None, :].
                                                repeat(1, self.num_loc, 1).to("cpu"),
                                                adver_action[:, None, ...].to("cpu")).squeeze(-1).float().to(self.device)

        td.set("stochastic_prize", stochastic_prize)
        return td
    
    def reset_stochastic_var(self, td, adver_out):
        td = self.reset_stochastic_prize(td, adver_out)
        return td

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None, **kwargs):
        ''''
        绘制出action: agent选择的路径
        '''
        import matplotlib.pyplot as plt
        import numpy as np

        from matplotlib import cm, colormaps

        if actions.device != "cpu":
            actions = actions.to("cpu")
        num_routine = (actions == 0).sum().item() + 2
        base = colormaps["nipy_spectral"]
        color_list = base(np.linspace(0, 1, num_routine))
        cmap_name = base.name + str(num_routine)
        out = base.from_list(cmap_name, color_list, num_routine)

        if ax is None:
            # Create a plot of the nodes
            _, ax = plt.subplots()

        td = td.detach().cpu()

        if actions is None:
            actions = td.get("action", None)
        
        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0]
        
        locs = td["locs"]
        prize = td["prize"]
        # real_node_prize = td["real_node_prize"]
        real_prize = td["stochastic_prize"]

        # add the depot at the first action 
        actions = torch.cat([torch.tensor([0]), actions])

        # gather locs in order of action if available
        if actions is None:
            log.warning("No action in TensorDict, rendering unsorted locs")
        else:
            locs = locs

        # Cat the first node to the end to complete the tour
        x, y = locs[:, 0], locs[:, 1]

        # plot depot
        ax.scatter(
            locs[0, 0],
            locs[0, 1],
            edgecolors=cm.Set2(2),
            facecolors="none",
            s=100,
            linewidths=2,
            marker="s",
            alpha=1,
        )

        # plot visited nodes
        ax.scatter(
            x[1:],
            y[1:],
            edgecolors=cm.Set2(0),
            facecolors="none",
            s=50,
            linewidths=2,
            marker="o",
            alpha=1,
        )

        move_x = 0.08        # text整体位置向右平移
        # plot prize bars
        for node_idx in range(1, len(locs)):
            ax.add_patch(
                plt.Rectangle(
                    (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
                    0.01,   # width
                    2*prize[node_idx]/10,       # ?
                    edgecolor=cm.Set2(0),
                    facecolor=cm.Set2(0),
                    fill=True,
                )
            )
            
        # text node idx
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0] - 0.18 + move_x,
                locs[node_idx, 1] - 0.025,
                f"{node_idx}:",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color="b",
            )

        # text prize
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0] + move_x,
                locs[node_idx, 1] - 0.025,
                f"{prize[node_idx].item():.2f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color=cm.Set2(0),
            )
        
        # plot real prize bars
        for node_idx in range(1, len(locs)):
            ax.add_patch(
                plt.Rectangle(
                    (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
                    0.005,
                    2*real_prize[node_idx]/10,      # ?
                    edgecolor=cm.Set2(1),
                    facecolor=cm.Set2(1),
                    fill=True,
                )
            )

        # text real prize
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0] - 0.085+ move_x,
                locs[node_idx, 1] - 0.025,
                f"{real_prize[node_idx].item():.2f} |",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color=cm.Set2(1),
            )

        # text depot
        ax.text(
            locs[0, 0],
            locs[0, 1] - 0.025,
            "Depot",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            color=cm.Set2(2),
        )

        # plot actions
        color_idx = 0
        for action_idx in range(len(actions) - 1):
            if actions[action_idx] == 0:
                color_idx += 1
            from_loc = locs[actions[action_idx]]
            to_loc = locs[actions[action_idx + 1]]
            ax.plot(
                [from_loc[0], to_loc[0]],
                [from_loc[1], to_loc[1]],
                color=out(color_idx),
                lw=1,
            )
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="-|>", color=out(color_idx)),
                size=15,
                annotation_clip=False,
            )
        
        # Setup limits and show
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.show()
        if kwargs["save_pt"]:
            plt.savefig(kwargs["save_pt"])