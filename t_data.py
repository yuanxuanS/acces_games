import numpy as np

pth = "/home/panpan/rl4co/data0/svrp/svrp_modelize20_test_seed1234_size100.npz"
data = np.load(pth)
print(dict(data["locs"][0, :10]))