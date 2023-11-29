import torch

from rl4co.envs import TSPEnv, SPCTSPEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer

# RL4CO env based on TorchRL
env = SPCTSPEnv(num_loc=20) 
# print(env.action_spec)
# print(dir(env.dataset()))
print(env.dataset().data)       # td: data variables in env
print(len(env.dataset()))   # init with 0 data
# print(env.dataset()[0])
# for td in env.dataset():
#     print(td)
# Model: default is AM with REINFORCE and greedy rollout baseline
model = AttentionModel(env, 
                       baseline="rollout",
                       train_data_size=100_000,
                       val_data_size=10_000
                       ) 

# # Greedy rollouts over untrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=[3]).to(device)      # init batch_size datas by generate_data, return data in td
print(td_init)
print(env.dataset().data)       # td: data variables in env
print(len(env.dataset()))   # init with 0 data
model = model.to(device)
# return bu policy
out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
print(out)
# # Plotting
# # print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
# # for td, actions in zip(td_init, out['actions'].cpu()):
# #     env.render(td, actions)

# trainer = RL4COTrainer(
#     max_epochs=3,
#     accelerator="auto",
#     devices=1,
#     logger=None,
# )

# trainer.fit(model)