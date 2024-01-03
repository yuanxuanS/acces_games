from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle

class SCPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=5, offset=0, cover_min=0.1, cover_max=0.3, alpha=0.95, distribution=None):
        super(SCPDataset, self).__init__()
        seed_value = 42
        torch.manual_seed(seed_value)
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'  # 检查file的扩展名是否为.pkl，如果不是则报AssertionError

            with open(filename, 'rb') as f:  # 'rb' means open this file in binary mode
                data = pickle.load(f)  # pickle.load() is used to load data in binary mode
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
                # Converts each row of data into a PyTorch FloatTensor and stores the resulting list
        else:
            # Sample points randomly in [0, 1] square
            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'cover_min': cover_min,
                    'cover_max': cover_max,
                    'cost': torch.FloatTensor(size, 1).uniform_(1, 10),
                    'alpha': alpha
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    #def cover_prob(self):

#def infor_process(tuples):


def solve_it(input_data):
    # Initialize the input data
    locs = [d['loc'] for d in input_data]
    costs = [d['cost'] for d in input_data]
    cover_min = [d['cover_min'] for d in input_data]
    cover_max = [d['cover_max'] for d in input_data]
    alpha = [d['alpha'] for d in input_data]
    batch_size = len(locs)
    n_loc, _ = locs[0].size()  # loc共有三个维度，前两维分别是batch_size和n_loc
    print('batch_size', batch_size)
    print('n_loc', n_loc)
    #
    item_count = int(n_loc)  # the number of item need to be covered
    set_count = int(n_loc)  # the number of potential locations can be chosen

    sets = []
    set_one = []
    for j in range(0, batch_size):
        parts = (locs[j][:, None, :] - locs[j][None, :, :]).norm(p=2, dim=-1) # output: [50, 50] torch tensor
        for i in range(0, set_count):
            #print('parts[i, :] < cover_max[j]', parts[i, :].argsort()[parts[i, :] < cover_min[j]])
            set_one.append(Set(i, float(costs[j][i]), parts[i, :].argsort()[parts[i, :] < cover_max[j]]))
        sets.append(set_one)
        print('set_one', set_one)
        set_one = []

    # build a trivial solution
    # use the multiplication operator (*) with a list and an integer, it creates a new list by repeating the elements of the original list the specified number of times.
    #solution = [0] * set_count  # each element is initialized to 0, and the length of the list is equal to the value of the variable set_count
    #covered = set()  # initialize an empty set, using |= to union other set
    obj = []
    output_data = []
    for j in range(batch_size):
        solution = [0] * set_count
        covered = set()
        while len(covered) < item_count:
            # sorted() 相比于 sort()可以在原有sets的基础上返回一个新副本，而sort()是直接在原数据上修改
            # key=lambda means self defined function of sorting, i.e. according to s, sort it in ascending order
            sorted_sets = sorted(sets[j], key=lambda s: s.cost / len(set(s.items) - covered) if len(
                set(s.items) - covered) > 0 else float('inf'))
            print('len(covered)', len(covered))
            # print('sorted_sets', sorted_sets)
            # print('solution[s.index]', solution)
            for s in sorted_sets:
                if solution[s.index] < 1:
                    solution[s.index] = 1
                    #print('s.index', s.index)
                    covered |= set(s.items)
                    #print('covered', covered)
                    break
        # calculate the cost of the solution
        obj = sum([s.cost * solution[s.index] for s in sets[j]])
        # prepare the solution in the specified output format
        output_data.append(str(obj) + ' ' + str(0) + '\n' + ' '.join(map(str, solution)))
        # print(sum(solution))



    # calculate the cost of the solution
    #obj = sum([s.cost * solution[s.index] for s in sets])

    # prepare the solution in the specified output format
    #output_data = str(obj) + ' ' + str(0) + '\n'
    #output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    Set = namedtuple("Set", ['index', 'cost', 'items'])
    Input = SCPDataset()
    print(solve_it(Input.data))