import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import OPSAEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.model_adversary import PPOContiAdvModel
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from rl4co.heuristic import CW_svrp, TabuSearch_svrp

# RL4CO env based on TorchRL
env = OPSAEnv(num_loc=20) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=[10]).to(device)      # return batch_size datas by generate_data

# agent
model = AttentionModel(env, 
                       baseline="rollout",
                       batch_size=512,
                       val_batch_size=1024,
                       test_batch_size=1024,
                       train_data_size=1_280_000,
                       val_data_size=10_000,
                       test_data_size=10_000
                       )
model = model.to(device)
out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
print(f"Prizes: {[f'{r.item():.2f}' for r in out['reward']]}")


# adv
adv = PPOContiAdvModel(env, 
                        opponent=model,  # with agent, opponent must be None
                       batch_size=128,   #512,
                       val_batch_size=128,   #1024,
                       test_batch_size=128,  #1024,
                       train_data_size=256,  #1_280_000,
                       val_data_size=128,    #10_000,
                       test_data_size=128,   #10_000
                       policy_kwargs={ "action_dim": 9}
                       ) 
adv = adv.to(device)
out_adv = adv(td_init)
print(out_adv["action_adv"])


out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
print(f"Prizes: {[f'{r.item():.2f}' for r in out['reward']]}")

# td_init = env.reset(batch_size=[10]).to(device)      # return batch_size datas by generate_data
td = env.reset_stochastic_var(td_init, out_adv["action_adv"][..., None])
out_withadv = model(td.clone(), phase="test", decode_type="greedy", return_actions=True)
print(f"Prizes: {[f'{r.item():.2f}' for r in out_withadv['reward']]}")



'''
# train
logger = None
wandb.login()

logger = WandbLogger(project="rl4co-test", name="opswtw")
## callbacks
# Checkpointing callback: save models when validation reward improves
checkpoint_callback = ModelCheckpoint(  dirpath="opswtw_checkpoints", # save to checkpoints/
                                        filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
                                        save_top_k=1, # save only the best model
                                        save_last=True, # save the last model
                                        monitor="val/reward", # monitor validation reward
                                        mode="max") # maximize validation reward


# Print model summary
rich_model_summary = RichModelSummary(max_depth=3)
callbacks = [checkpoint_callback, rich_model_summary]

trainer = RL4COTrainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
    logger=logger,
    callbacks=callbacks,
)

trainer.fit(model)

model = model.to(device)

out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
print(f"Prizes: {[f'{r.item():.2f}' for r in out['reward']]}")
'''
