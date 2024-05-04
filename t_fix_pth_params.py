
from rl4co.models.zoo.am import AttentionModel
from rl4co.envs import SVRPEnv
import torch

pth = "/home/panpan/rl4co/logs/train/runs/svrp20/am-svrp20/2024-02-27_12-22-21/rl4co/ajzphdl4/checkpoints/epoch=99-step=250000.ckpt"
env = SVRPEnv()
model = AttentionModel(env)
state_dict = torch.load(pth, map_location=torch.device('cpu'))
# print(state_dict["hparams_name"])
print(state_dict["hyper_parameters"]["env"].test_file)
state_dict["hyper_parameters"]["env"].test_file = "/home/panpan/rl4co/data/svrp/svrp_modelize20_test_seed1234_size100.npz"
print(state_dict["hyper_parameters"]["env"].val_file)
state_dict["hyper_parameters"]["env"].val_file = "/home/panpan/rl4co/data/svrp/svrp_modelize20_val_seed4321_size10000.npz"
torch.save(state_dict, pth)
print(state_dict["hyper_parameters"]["env"].test_file)
print(state_dict["hyper_parameters"]["env"].val_file)
# print(state_dict["loops"])