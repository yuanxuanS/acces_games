from rl4co.tasks.train_psro import Protagonist, Adversary
from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.model_adversary import PPOContiAdvModel
from rl4co.model_adversary.zoo.ppo.policy_conti import PPOContiAdvPolicy
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.envs import CSPEnv, SVRPEnv
# from rl4co.tasks.train_psro import play_game
from rl4co.tasks.eval_withpsro_adv import play_game
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
import torch
from rl4co.data.dataset import tensordict_collate_fn

# torch.manual_seed(1800)
pth = "/home/panpan/rl4co/logs/train_psro/runs/svrp50/am-svrp50/2024-05-10_17-28-03/"
pth_rl = "/home/panpan/rl4co/logs/train/runs/svrp50/am-svrp50/2024-05-04_22-56-20/rl4co/n4no1ty5/checkpoints/epoch=99-step=125000.ckpt"

val_data_pth = "/home/panpan/rl4co/data0/svrp/svrp_modelize50_test_seed1234_size100.npz"
prog_idx = 0
adv_idx = 1

env = SVRPEnv(num_loc=50, stoch_idx=0)
protagonist = Protagonist(AttentionModel, AttentionModelPolicy, env)
protagonist.load_model_weights(pth + "models_weights/")
print("prog policy model 0 :", protagonist.policies[0].encoder.init_embedding.init_embed.weight.data[0])
print("prog policy model 1:", protagonist.policies[1].encoder.init_embedding.init_embed.weight.data[0])
# print("prog policy baaselin", protagonist.policies[0].baseline)

adversary = Adversary(PPOContiAdvModel, PPOContiAdvPolicy, CriticNetwork, env)
adversary.load_model_weights(pth + "/models_weights")


model_prog = AttentionModel(env)
model_prog.policy = protagonist.get_policy_i(prog_idx)
model_prog = model_prog.load_from_checkpoint(pth_rl)
print("loaded RL model:", model_prog.policy.encoder.init_embedding.init_embed.weight.data[0])

model_adv = PPOContiAdvModel(env, None)
model_adv.policy, model_adv.critic = adversary.get_policy_i(adv_idx)
print(f"loaded adv model {adv_idx}", model_adv.policy.encoder.init_embedding.init_embed.weight.data[0])

val_data = env.load_data(val_data_pth)
val_dataset = TensorDictDataset(val_data)
val_dl = DataLoader(val_dataset, batch_size=100, collate_fn=tensordict_collate_fn)
    


print("loaded prog policy 0", model_prog.policy.encoder.init_embedding.init_embed.weight.data[0])

eval_baseline = True
baseline_method = "cw"
baseline_fn = "./baseline_"+ baseline_method+ ".npz"
rl_rewards = []
rl_rewards_all = None

bl_rewards = []
bl_rewards_var = []
bl_rewards_all = None
for batch in val_dl:
    rl_res, bl_res = play_game(env, batch.clone(), model_prog, model_adv, eval_baseline, baseline_method, baseline_fn)
    batch_rl_mean, batch_rl_allg = rl_res
    batch_l_mean, batch_bl_var, batch_bl_allg = bl_res

    rl_rewards.append(batch_rl_mean)
    if rl_rewards_all == None:
        rl_rewards_all = batch_rl_allg
    else:
        rl_rewards_all = torch.cat((rl_rewards_all, batch_rl_allg), dim=0)
    
    bl_rewards.append(batch_l_mean)
    bl_rewards_var.append(batch_bl_var)
    if bl_rewards_all == None:
        bl_rewards_all = batch_bl_allg
    else:
        bl_rewards_all = torch.cat((bl_rewards_all, batch_bl_allg), dim=0)
    
rl_payoff = torch.tensor(rl_rewards).mean().item()
bl_payoff = torch.tensor(bl_rewards).mean().item()

print("rl reward len and value is ", len(rl_rewards), rl_rewards)  # batch 
print("rl rewards length in all graph is ", len(rl_rewards_all))       # 100
print("rl rewards in all graph: ", rl_rewards_all[:10])
print("rl payoff is ", rl_payoff)   # 1
print("bl reward is ", len(bl_rewards), bl_rewards[:10])    # batch
print("bl rewards length in all graph is ", len(bl_rewards_all))       # 100
print("bl rewards in all graph: ", bl_rewards_all[:10])
print("bl payoff is ", bl_payoff)  # 1


# # 
# for batch in val_dl:
#     print(batch["locs"][3])
#     print(batch["max_cover"][3])
#     print(batch["stochastic_maxcover"][3])
#     break