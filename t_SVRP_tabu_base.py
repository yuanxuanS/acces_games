import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
import pandas as pd
from rl4co.envs import SVRPEnv, SPCTSPEnv
from torch.nn.utils.rnn import pad_sequence
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance
from rl4co.heuristic import TabuSearch_svrp
# RL4CO env based on TorchRL
env = SVRPEnv(num_loc=20) 

# print(env.dataset().data)       # td: data variables in env
# print(len(env.dataset()))   # init with 0 data
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
td_init = env.reset(batch_size=[10]).to(device)      # init batch_size datas by generate_data, return data in td

baseline = TabuSearch_svrp(td_init.clone())
out = baseline.forward()
# print(td_init)
# print(env.dataset().data)       # td: data variables in env
# print(len(env.dataset()))   # init with 0 data
# model = model.to(device)

### test greedy rollout with untrained model
# out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
# Plotting
# print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
# for td, actions in zip(td_init, out['actions'].cpu()):
#     env.render(td, actions)
# ------------------

       

# print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")


## callbacks
# Checkpointing callback: save models when validation reward improves
# checkpoint_callback = ModelCheckpoint(  dirpath="svrp_checkpoints", # save to checkpoints/
#                                         filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
#                                         save_top_k=1, # save only the best model
#                                         save_last=True, # save the last model
#                                         monitor="val/reward", # monitor validation reward
#                                         mode="max") # maximize validation reward


# Print model summary
# rich_model_summary = RichModelSummary(max_depth=3)

# Callbacks list
# callbacks = [checkpoint_callback, rich_model_summary]

### logging 
# wandb.login()

# logger = WandbLogger(project="rl4co-robust", name="svrp-am-weath-demand_f")
## Keep below if you don't want logging
# logger = None

# trainer = RL4COTrainer(
#     max_epochs=3,
#     accelerator="auto",
#     devices=1,
#     logger=logger,
#     callbacks=callbacks,
# )

# trainer.fit(model)

### testing
# model = model.to(device)
# out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
# print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")


### load and test
# Environment, Model, and Lightning Module (reinstantiate from scratch)
# model_l = AttentionModel(env,
#                        baseline="rollout",
#                        train_data_size=100_000,
#                        test_data_size=10_000,
#                        optimizer_kwargs={'lr': 1e-4}
#                        )

# # Note that by default, Lightning will call checkpoints from newer runs with "-v{version}" suffix
# # unless you specify the checkpoint path explicitly
# new_model_checkpoint = AttentionModel.load_from_checkpoint("svrp_checkpoints/last.ckpt", strict=False)

# # Greedy rollouts over trained model (same states as previous plot, with 20 nodes)
# model_n = new_model_checkpoint.to(device)
# env_n = new_model_checkpoint.env.to(device)

# out = model_n(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)

# # Plotting
# print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
