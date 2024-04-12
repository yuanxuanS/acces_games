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


class OPSWTWEnv(RL4COEnvBase):
    """
    Stochastic weights time window Traveling Salesman Problem environment
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent back to depot .
    In that case, the reward is cumulative prize
    The expected travel time is distance. The stochastic travel time is lower than expected one.
    Args:
        num_loc: number of locations (cities) in the TSP
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """
    name = "opswtw"

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
        
        if "train" in fpath:
            noise_pth = "/home/panpan/rl4co/data/opswtw_train/noise_train.npz"
        elif "val" in fpath:
            noise_pth = "/home/panpan/rl4co/data/opswtw/noise_val.npz"
        noise_load = load_npz_to_tensordict(noise_pth)
        td_load.set("noise", noise_load["noise"][:batch_size, ...])
        
        return td_load
    
    def generate_data(self, batch_size) -> TensorDict:
        # 返回归一化的数据
        data_pth = "/home/panpan/rl4co/data/opswtw_train/opswtw_train.npz"
        noise_pth = "/home/panpan/rl4co/data/opswtw_train/noise_train.npz"

        x = np.load(data_pth)
        x_dict = dict(x)
        noise_data = np.load(noise_pth)
        noise_dict = dict(noise_data)
        whole_size = x_dict[list(x_dict.keys())[0]].shape[0]

        if isinstance(batch_size, list):
            batch_size = batch_size[0]
        assert whole_size >= batch_size, "Required size exceeds data size"

        locs = x_dict["locs"][:batch_size, ...]
        locs[..., 0] /= 200
        locs[..., 1] /= 50
        tw_low = x_dict["tw_low"][:batch_size, ...]
        tw_low /= 100      # 数据中最大的时间
        tw_high = x_dict["tw_high"][:batch_size, ...]
        tw_high /= 100
        prize = x_dict["prize"][:batch_size, ...]       # has been normalized
        maxtime = x_dict["maxtime"][:batch_size, ...] / 100
        noise = noise_dict["noise"][:batch_size, ...]       # batch, size+1; have been normalized

        return TensorDict(
            {
            "locs": torch.Tensor(locs),
            "tw_low": torch.Tensor(tw_low),
            "tw_high": torch.Tensor(tw_high),
            "prize": torch.Tensor(prize),
            "maxtime": torch.Tensor(maxtime),
            "noise": torch.Tensor(noise)
        },
        batch_size=batch_size,
        device=self.device
        )

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
        adj = OPSWTWEnv.get_adjacent_matrix(row_locs, col_locs)     # batch, size, size
        penalty = torch.zeros((*batch_size, ), device=self.device)
        # 从depot出发，接下来不能拜访depot
        action_mask = torch.ones((*batch_size, self.num_loc), device=self.device).bool()    # True if visted
        action_mask[:, 0] = False       # mask depot
        # 不能拜访 low_tw > maxtime的node
        action_mask[td["tw_low"] > td["maxtime"]] = False

        visited = torch.zeros((*batch_size, self.num_loc), dtype=torch.uint8, device=self.device)
        visited[:, 0] = 1
        done = torch.zeros((*batch_size,), device=self.device, dtype=torch.bool)
        td_reset = TensorDict(
            {
                "locs": locs,
                "tw_low": td["tw_low"],
                "tw_high": td["tw_high"],
                "prize": td["prize"],
                "curr_prize": curr_prize,
                "maxtime": td["maxtime"],
                "tour_time": curr_time,
                "adj": adj,
                "noise": td["noise"],
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
        noise_node = td["noise"]
        
        noise = (noise_node[range(batch_size), last_node] + noise_node[range(batch_size), current_node]) / 2
        tour_time += noise * time

        
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
        prize = td["prize"]
        curr_prize = td["curr_prize"]
        curr_prize += prize[range(batch_size), current_node]



        td.update(
            {
                "current_node": current_node,
                "visited": visited,
                "tour_time": tour_time,
                "penalty": penalty,
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