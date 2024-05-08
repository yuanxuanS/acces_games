
from rl4co.models.zoo.am import AttentionModel
from rl4co.envs import SVRPEnv, CSPEnv
import torch

# svrp50
# pth = "/home/u2021141179/rl4co_cp/logs/train/runs/svrp50/am-svrp50/2024-05-04_22-56-20/rl4co/n4no1ty5/checkpoints/epoch=99-step=125000.ckpt"
# csp20
# pth = "/home/u2021141179/rl4co_cp/logs/train/runs/csp20/am-csp20/2024-04-25_14-06-46/rl4co/u9r7s1o3/checkpoints/epoch=99-step=125000.ckpt"
# csp50
pth = "/home/u2021141179/rl4co_cp/logs/train/runs/svrp20/am-svrp20/2024-02-27_12-22-21/rl4co/ajzphdl4/checkpoints/epoch=99-step=250000.ckpt"
env = SVRPEnv()
model = AttentionModel(env)
state_dict = torch.load(pth, map_location=torch.device('cpu'))

print(state_dict["hyper_parameters"]["env"].test_file)
print(state_dict["hyper_parameters"]["env"].val_file)
print(state_dict["hyper_parameters"]["env"].data_dir)
print(state_dict["hyper_parameters"]["data_dir"])


#########################
# ### 检查好 data idx, num_loc,  env.name对不对！
state_dict["hyper_parameters"]["env"].test_file = "/home/u2021141179/rl4co_cp/data0/svrp/svrp_modelize20_test_seed1234_size100.npz"
state_dict["hyper_parameters"]["env"].val_file = "/home/u2021141179/rl4co_cp/data0/svrp/svrp_modelize20_val_seed4321_size10000.npz"

state_dict["hyper_parameters"]["data_dir"] = "data0/"
state_dict["hyper_parameters"]["env"].data_dir = "/home/u2021141179/rl4co_cp/data0/svrp"

torch.save(state_dict, pth)

print(state_dict["hyper_parameters"]["env"].test_file)
print(state_dict["hyper_parameters"]["env"].val_file)
print(state_dict["hyper_parameters"]["env"].data_dir)
print(state_dict["hyper_parameters"]["data_dir"])

# print(state_dict["loops"])