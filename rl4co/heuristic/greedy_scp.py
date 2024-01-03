import torch
import pandas as pd
from rl4co.envs import SVRPEnv, SPCTSPEnv
from torch.nn.utils.rnn import pad_sequence
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance
from rl4co.utils.heuristic_utils import convert_to_fit_npz
from collections import namedtuple


class Greedy_scp:
    def __init__(self, td, min_cover, max_cover) -> None:
        super().__init__()
        '''
        td: TensorDict
        td["locs"]: [batch, num_node, 2], locs
        td["costs"]:  [batch, num_node, 1]
        
        td["action_mask"]: [batch, 1, num_customer+1]
        td["current_node"]: [batch, 1]
        td["covered_node"]: [batch, num_node]
        
        forward(): call this to get all solutions, a nested list, call convert_to_fit_npz as for savez
        '''
        self.td = td
        self.data_nodes = None      # [batch, num_customer, 2], distance to depot, demand
        self.savings = dict()
        self.sorted_savings = None
        self.routes = list()
        self.capacity = 1        # define by env
        
        self.min_cover = min_cover
        self.max_cover = max_cover
        self.batch_size = self.td["locs"].size(0)
        self.num_locations = self.td["locs"].size(1)
        print('batch_size', self.batch_size)
        print('n_loc', self.num_locations)
        
        self.item_count = int(self.num_locations)  # the number of item need to be covered
        self.set_count = int(self.num_locations)  # the number of potential locations can be chosen
    
    
        self.Set = namedtuple("Set", ['index', 'cost', 'items'])

    def forward(self):
        # Initialize the input data
        # locs = [d['loc'] for d in input_data]
        # costs = [d['cost'] for d in input_data]
        # cover_min = [d['cover_min'] for d in input_data]
        # cover_max = [d['cover_max'] for d in input_data]
        # alpha = [d['alpha'] for d in input_data]
        # n_loc, _ = locs[0].size()  # loc共有三个维度，前两维分别是batch_size和n_loc
        
        #
        

        self.sets = []      # 所有isntance的数据信息
        
        for j in range(0, self.batch_size):
            set_one = self.get_single_info(j)
            self.sets.append(set_one)
        
        self.reward_all = []
        self.output_data = []
        for j in range(self.batch_size):
            output = self.forward_single(j)
            self.output_data.append(output)
        print(self.output_data)
        print(self.reward_all)
            
    def forward_single(self, idx):
        solution = [0] * self.set_count
        covered = set()
        while len(covered) < self.item_count:
            # sorted() 相比于 sort()可以在原有sets的基础上返回一个新副本，而sort()是直接在原数据上修改
            # key=lambda means self defined function of sorting, i.e. according to s, sort it in ascending order
            
            # 对一条数据排序后的sets
            sorted_sets = sorted(self.sets[idx], key=lambda s: s.cost / len(set(s.items) - covered) if len(
                set(s.items) - covered) > 0 else float('inf'))
            # print('len(covered)', len(covered))
            # print('sorted_sets', sorted_sets)
            # print('solution[s.index]', solution)
            for s in sorted_sets:
                if solution[s.index] < 1:
                    solution[s.index] = 1
                    #print('s.index', s.index)
                    covered |= set(s.items)
                    print('covered', covered)
                    break
        # calculate the cost of the solution
        self.obj = sum([s.cost * solution[s.index] for s in self.sets[idx]])
        self.reward_all.append(self.obj)
        # prepare the solution in the specified output format
        output_ = str(self.obj) + ' ' + str(0) + '\n' + ' '.join(map(str, solution))
        return output_
        
        # print(sum(solution))

    def get_single_info(self, idx):
            
            set_one = []    # 每个instance中所有loc的数据: 每个loc: index, cost, covered nodes 
            parts = (self.td["locs"][idx, ...][:, None, :] - self.td["locs"][idx, ...][None, :, :]).norm(p=2, dim=-1) # output: [50, 50] torch tensor
            for i in range(0, self.set_count):
                #print('parts[i, :] < cover_max[j]', parts[i, :].argsort()[parts[i, :] < cover_min[j]])
                set_one.append(self.Set(i, float(self.td["costs"][idx, i]), parts[i, :].argsort()[parts[i, :] < self.max_cover]))
            
            print('set_one', set_one)
            return set_one

        # build a trivial solution
        # use the multiplication operator (*) with a list and an integer, it creates a new list by repeating the elements of the original list the specified number of times.
        #solution = [0] * set_count  # each element is initialized to 0, and the length of the list is equal to the value of the variable set_count
        #covered = set()  # initialize an empty set, using |= to union other set
        


        # calculate the cost of the solution
        #obj = sum([s.cost * solution[s.index] for s in sets])

        # prepare the solution in the specified output format
        #output_data = str(obj) + ' ' + str(0) + '\n'
        #output_data += ' '.join(map(str, solution))


 