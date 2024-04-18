import torch
import numpy as np
class Greedy_opswtw:
    def __init__(self, num_loc, td) -> None:
        super().__init__()
        self.num_loc = num_loc
        self.td = td
        self.adj = td["adj"]
        self.locs = td["locs"]  # batch, num_loc, 2
        self.tw_low = td["tw_low"]
        self.tw_high = td["tw_high"]
        self.maxtime = td["maxtime"]
        self.tour_time = None
        self.done = None
    def forward(self):

        return
    
    def get_adjacent_matrix(self, x: np.array, y: np.array):
        return (x - y).norm(p=2, dim=-1)
    
    def solve(self):
        locs = self.locs.copy()
        self.batch_size = locs.shape[0]
        self.tour_time = np.zeros((self.batch_size, ))
        self.done = np.zeros((self.batch_size,), dtype=np.bool)
        # compute adj
        row_locs = locs[..., None, :].repeat(1, 1, self.num_loc, 1)
        col_locs = locs[..., None, :].repeat(1, 1, self.num_loc, 1).transpose(1, 2)
        self.adj = self.get_adjacent_matrix(row_locs, col_locs)

        self.current_node = np.zeros((self.batch_size,), dtype=np.int)

        while (~self.done).any():
            dist = self.adj[range(self.batch_size), self.current_node, :]   # batch, 20
            arrived_time = self.tour_time.repeat(1, self.num_loc) + dist    
            in_time_window = arrived_time < self.tw_high        # 
            self.done = np.sum(in_time_window, axis=1) > 0      # 有满足时间窗的node。 batch, 



    
def nn_algo_td(adj, noise, init_node=0, num_loc=20):
    '''
    adj: size*size
    noise: size
    '''
    adj = adj.clone()

    for i in range(num_loc):
        adj[i, i] = torch.inf
    
    tour = [init_node]
    adj[0, :] = torch.inf
    adj[:, 0] = torch.inf
    tour_time = torch.zeros(1)


    done = False
    while not done:
        current_node = tour[-1]
        min_index = torch.argmin(adj[current_node])
        adj[min_index, :] = torch.inf
        adj[:, min_index] = torch.inf
        tour.append(min_index)

        # update tour time


def nn_algo(init_node, cost_matrix, n_nodes):
    """
    Nearest Neighbour algorithm
    """
    cost_matrix = cost_matrix.copy()

    for i in range(1, n_nodes + 1):
        cost_matrix[i][i] = np.inf          # 不能选自己

    tour = [init_node]

    for _ in range(n_nodes - 1):
        node = tour[-1]
        min_index = np.argmin(cost_matrix[node])
        for t in tour:      # 改变距离相当于，将自己节点mask掉
            cost_matrix[min_index + 1][t] = np.inf
            cost_matrix[t][min_index + 1] = np.inf
        tour.append(min_index + 1)
    tour.append(init_node)
    return tour