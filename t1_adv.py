import torch

from rl4co.envs import TSPEnv, SPCTSPEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer

# RL4CO env based on TorchRL
env = SPCTSPEnv(num_loc=20) 

# agent: default is AM with REINFORCE and greedy rollout baseline
agent = AttentionModel(env, 
                       baseline="rollout",
                       train_data_size=100_000,
                       val_data_size=10_000
                       ) 
# adv: action: change prize value.
adv = PPO()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=[3]).to(device)
agent = agent.to(device)

# trainer
trainer = RL4COTrainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
    logger=None,
)

iter = 10
for i in range(iter):
    
    # train agent iterations
    trainer.fit(agent)
    # train adv
    trainer.fit(adv)
    
    # evaluate agent
    out = agent(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)