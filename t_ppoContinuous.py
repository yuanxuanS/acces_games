import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import SVRPEnv
from rl4co.model_adversary import PPOContiAdvModel
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from rl4co.heuristic import CW_svrp, TabuSearch_svrp

# RL4CO env based on TorchRL
env = SVRPEnv(num_loc=20) 

# print(env.dataset().data)       # td: data variables in env
# print(len(env.dataset()))   # init with 0 data
# print(env.dataset()[0])
# for td in env.dataset():
#     print(td)
# Model: default is AM with REINFORCE and greedy rollout baseline

# baseline
baseline = CW_svrp

model = PPOContiAdvModel(env, 
                        opponent=baseline,
                       batch_size=128,   #512,
                       val_batch_size=128,   #1024,
                       test_batch_size=128,  #1024,
                       train_data_size=256,  #1_280_000,
                       val_data_size=128,    #10_000,
                       test_data_size=128   #10_000
                       ) 

# # Greedy rollouts over untrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=[10]).to(device)      # return batch_size datas by generate_data
print(td_init["weather"])
# print(env.dataset().data)       # td: data variables in env
# print(len(env.dataset()))   # init with 0 data
model = model.to(device)


### test greedy rollout with untrained model
out = model(td_init.clone(), phase="test", return_actions=True)
td_init = env.reset_stochastic_demand(td_init.clone(), out["action_adv"][..., None])

baseline_return = baseline(td_init.clone()).forward()
td_init["reward"] = baseline_return["rewards"]
# Plotting
print(f"Tour lengths: {[f'{-r.item():.2f}' for r in td_init['reward']]}")
# for td, actions in zip(td_init, out['actions'].cpu()):
#     env.render(td, actions)


## callbacks
# Checkpointing callback: save models when validation reward improves
# checkpoint_callback = ModelCheckpoint(  dirpath="svrp_checkpoints", # save to checkpoints/
#                                         filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
#                                         save_top_k=1, # save only the best model
#                                         save_last=True, # save the last model
#                                         monitor="val/reward", # monitor validation reward
#                                         mode="max") # maximize validation reward


# Print model summary
rich_model_summary = RichModelSummary(max_depth=3)

# Callbacks list
# callbacks = [checkpoint_callback, rich_model_summary]

## logging 
wandb.login()

logger = WandbLogger(project="rl4co-robust", name="svrp-adv")
# Keep below if you don't want logging
# logger = None

trainer = RL4COTrainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
    logger=logger,
    callbacks=None,
)

trainer.fit(model)

### testing
model = model.to(device)
out = model(td_init.clone(), phase="test", return_actions=True)
td_init = env.reset_stochastic_demand(td_init.clone(), out["action_adv"][..., None])

baseline_return = baseline(td_init.clone()).forward()
td_init["reward"] = baseline_return["rewards"]
# Plotting
print(f"Tour lengths: {[f'{-r.item():.2f}' for r in td_init['reward']]}")
'''
## baseline 
baseline_cw = CW_svrp(td_init.clone())
out = baseline_cw.forward()

baseline_tabu = TabuSearch_svrp(td_init.clone())
out = baseline_tabu.forward()
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
'''