import time
from rl4co.model_MA.utils_psro import load_stoch_data, play_game, stochdata_key_mapping
from rl4co.model_MA.utils_psro_eval import eval_allgraph
import torch
import os
import numpy as np
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
from rl4co.data.dataset import tensordict_collate_fn
from rl4co.utils.lightning import get_lightning_device
from tensordict.tensordict import TensorDict


def eval_psro(cfg, env, test_data, stoch_data,
              prog_strategy, protagonist_tmp, protagonist_model,
              adver_strategy, adversary_tmp, adversary_model,
              stoch_data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = time.time()
    rewards_rl = []
    for r in range(len(prog_strategy)):     # 当error停止，出现policy个数-strategy个数 = 1
        protagonist_model.policy = protagonist_tmp.get_policy_i(r)
        for c in range(len(adver_strategy)):

            adversary_model.policy, adversary_model.critic = adversary_tmp.get_policy_i(c)

            rl_rewards = [] # [batch iter,]
            rl_rewards_all = None   # 1dim, [data_size's size

            bl_rewards = [] # batch iter,
            bl_rewards_var = [] # batch iter,
            bl_rewards_all = None       # batch's size

            test_data = test_data.to(device)
            test_dataset = TensorDictDataset(test_data)
            test_dl = DataLoader(test_dataset, batch_size=cfg.model_psro.test_batch_size, collate_fn=tensordict_collate_fn)

            loaded = False
            if not cfg.eval_no_saved_data:      # need save new stoch data of other dataset (no trained )
                loaded = True
                stoch_dict, stoch_data = load_stoch_data(env, stoch_data_dir, stoch_data, c)

                stoch_dict = TensorDict(stoch_dict, batch_size=cfg.model_psro.test_data_size, device=device)
                stoch_dataset = TensorDictDataset(stoch_dict)
                stoch_dl = DataLoader(stoch_dataset, batch_size=cfg.model_psro.test_batch_size, collate_fn=tensordict_collate_fn)
            else:
                stoch_save_dir = cfg.evaluate_adv_dir+"/adv_stoch_data_otherdata/"
                if os.path.exists(stoch_save_dir):
                    loaded = True
                    stoch_dict, stoch_data = load_stoch_data(env, stoch_save_dir, stoch_data, c)

                    stoch_dict = TensorDict(stoch_dict, batch_size=cfg.model_psro.test_data_size, device=device)
                    stoch_dataset = TensorDictDataset(stoch_dict)
                    stoch_dl = DataLoader(stoch_dataset, batch_size=cfg.model_psro.test_batch_size, collate_fn=tensordict_collate_fn)
                else:
                    loaded = False
                    os.mkdir(stoch_save_dir)
                    stoch_dl = test_dl

            for batch, stoch_batch in zip(test_dl, stoch_dl):
                if loaded:
                    rl_res, bl_res, stoch_data = play_game(env, batch.clone(), stoch_batch, stoch_data, c, 
                                        protagonist_model, adversary_model, False, False,)
                    
                else:
                    stoch_data_save_pth = stoch_save_dir + "adv_"+str(c) + ".npz"       # to ssave new stoch data
                    rl_res, bl_res, stoch_data = play_game(env, batch.clone(), None, stoch_data, c, 
                                            protagonist_model, adversary_model, True, False,)
                    
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

            print(f" r-{r} c-{c}", time.time())

            if not loaded:
                stochdata_key_lst = stochdata_key_mapping[env.name]
                for sk in stochdata_key_lst:
                    print(f"save stoch_data to {stoch_data_save_pth}, {stoch_data[sk][c][0]}")
                    np.savez(stoch_data_save_pth, stoch_data[sk][c].cpu())       # 自动转化为numpy

            rewards_rl.append(rl_rewards_all.cpu().tolist())
    rewards_rl = np.array(rewards_rl)
    print("shape ", rewards_rl.shape)
    rewards_rl = rewards_rl.T
    print("shape ", rewards_rl.shape)
    rewards_rl = rewards_rl.reshape(cfg.model_psro.test_data_size, len(prog_strategy), len(adver_strategy))
    print("shape ", rewards_rl.shape)
    # rewards_rl = rewards_rl.transpose(0, 2,1)
    print("shape ", rewards_rl.shape)
    rl_rewards_psro = eval_allgraph(rewards_rl, prog_strategy, adver_strategy)
    reward_eval, eval_var = rl_rewards_psro.mean(), rl_rewards_psro.var()

    time_ = time.time()-st

    return rewards_rl, reward_eval, eval_var, time_, stoch_data