
from rl4co.models.zoo.am import AttentionModel
from rl4co.envs import SVRPEnv, CSPEnv, OPSAEnv
import torch

# csp20-0
# pth = "/home/panpan/rl4co/logs/train/runs/svrp20/am-csp20/2024-04-25_14-06-46/rl4co/u9r7s1o3/checkpoints/epoch=99-step=125000.ckpt"
# pth="/home/panpan/rl4co/logs/train/runs/svrp50/am-svrp50/2024-05-04_22-56-20/rl4co/n4no1ty5/checkpoints/epoch=99-step=125000.ckpt"
# csp20-1
# pth = "/home/panpan/rl4co/logs/train/runs/svrp20/am-svrp20/2024-05-08_21-14-27/rl4co/sgtdk5tf/checkpoints/epoch=99-step=125000.ckpt"
# pth = "/home/panpan/rl4co/logs/train/runs/csp20/am-csp20/2024-04-25_14-06-46/rl4co/u9r7s1o3/checkpoints/epoch=99-step=125000.ckpt"
# csp50
# pth = "/home/panpan/rl4co/logs/train/runs/csp50/am-csp50/2024-05-04_07-06-59/rl4co/9rwb952x/checkpoints/epoch=99-step=125000.ckpt"
pth = "/home/panpan/rl4co/logs/train/runs/svrp50/am-svrp50/2024-05-18_16-10-46/rl4co/uiyuuo8l/checkpoints/epoch=99-step=31300.ckpt"
env = SVRPEnv()
model = AttentionModel(env)
state_dict = torch.load(pth, map_location=torch.device('cpu'))
# print(state_dict["hparams_name"])
# print(state_dict["hyper_parameters"].keys())
# print(state_dict["hyper_parameters"]["val_batch_size"])
# state_dict["hyper_parameters"]["val_batch_size"]=512
# torch.save(state_dict, pth)
# print(state_dict["hyper_parameters"]["val_batch_size"])
print(state_dict["hyper_parameters"]["env"].data_dir)
print(state_dict["hyper_parameters"]["env"].test_file)
print(state_dict["hyper_parameters"]["env"].val_file)
print(state_dict["hyper_parameters"]["data_dir"])
# print(state_dict["loops"])
state_dict["hyper_parameters"]["data_dir"] = "data0/"
state_dict["hyper_parameters"]["env"].data_dir = "/home/panpan/rl4co/data1/svrp"
state_dict["hyper_parameters"]["env"].test_file = "/home/panpan/rl4co/data1/svrp/svrp_modelize50_test_seed1234_size100.npz"
state_dict["hyper_parameters"]["env"].val_file = "/home/panpan/rl4co/data1/svrp/svrp_modelize50_val_seed4321_size10000.npz"
# 
# state_dict["hyper_parameters"]["env"].data_dir = "/home/panpan/rl4co/data0/svrp"
torch.save(state_dict, pth)
print(state_dict["hyper_parameters"]["data_dir"])
print(state_dict["hyper_parameters"]["env"].test_file)
print(state_dict["hyper_parameters"]["env"].val_file)
print(state_dict["hyper_parameters"]["env"].data_dir)
# print(state_dict["loops"])
