from typing import List, Optional, Tuple
from rl4co.tasks.eval_psro import evaluate_psro_policy

import hydra
import lightning as L
import pyrootutils
import torch
import collections
from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.model_adversary.zoo.ppo.policy_conti import PPOContiAdvPolicy
from rl4co.data.generate_data import generate_default_datasets, generate_dataset
from rl4co.tasks.eval_heuristic import evaluate_baseline_withpsroadv
from rl4co.model_adversary import PPOContiAdvModel
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
from rl4co.data.dataset import tensordict_collate_fn
from rl4co.tasks.train_psro import update_payoff, nash_solver, play_game, eval, sample_strategy, eval_noadver
from rl4co.tasks.train_psro import Adversary, Protagonist
from rl4co import utils
from rl4co.utils import RL4COTrainer
from memory_profiler import profile
from guppy import hpy
import numpy as np
import random
import nashpy as nash
import os
import time
from rl4co.model_MA.utils_psro import eval_oneprog_adv, eval_oneprog_adv_allgraph, stochdata_key_mapping, load_stoch_data
from tensordict.tensordict import TensorDict

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)




@utils.task_wrapper
# @profile(stream=open('log_mem_cvrp50.log', 'w+'))
def eval_withpsroadv(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    # trainer.logger = logger
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    # early_stopping = EarlyStopping(monitor='val/reward', min_delta=0.00, patience=10, mode="max")      # max代表曲线上升，阈值达到-7.99; 否则默认min
    # callbacks.append(early_stopping)  

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    data_cfg = {
            "val_data_size": cfg.model_psro.val_data_size,
            "test_data_size": cfg.model_psro.test_data_size,
        }
    generate_default_datasets(data_dir=cfg.paths.data_dir, data_cfg=data_cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 改：分别构建 prog, adv的算法model，for r c 遍历只需要重新load网络model
    protagonist_model: LightningModule = hydra.utils.instantiate(cfg.model, env)
    protagonist_model = protagonist_model.to(device)
    # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
    adversary_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, env)
    adversary_model = adversary_model.to(device)
    

    if cfg.get("eval_otherprog_with_psroadv"):
        # 加载psro prog和adver
        
        protagonist_tmp = Protagonist(AttentionModel, AttentionModelPolicy, env)
        protagonist_tmp.load_model_weights(cfg.evaluate_adv_dir+"/models_weights/")
        adversary_tmp = Adversary(PPOContiAdvModel, PPOContiAdvPolicy, CriticNetwork, env)
        adversary_tmp.load_model_weights(cfg.evaluate_adv_dir+"/models_weights")

        data = np.load(cfg.adv_npz_pth)  # 加载
        payoff_tmp = data['payoffs']  # 引用保存好的数组，他的格式默认是numpy.array
        adver_strategy = data['adver_strategy']
        if adversary_tmp.no_zeroth:
            print(f"before: adver strate is {adver_strategy}")
            adver_strategy = adver_strategy[1:]
            print(f"after: adver strate is {adver_strategy}")

        prog_strategy = data['prog_strategy']
        if protagonist_tmp.no_zeroth:
            print(f"before: prog strate is {prog_strategy}")
            prog_strategy = prog_strategy[1:]
            print(f"after:prog strate is {prog_strategy}")

        # load 对应环境的test数据
        test_data_pth = cfg.env.data_dir+"/"+cfg.env.test_file
        test_data = env.load_data(test_data_pth)
        stoch_data_dir = cfg.evaluate_adv_dir+"/adv_stoch_data/"
        for sk in stochdata_key_mapping[env.name]:
            stoch_data = {sk: {}}

        if cfg.get("eval_rl_prog"):
            # 
            print("eval rl agent with psro-adversary")
            
            rl_prog_pth = cfg.rl_prog_pth
            protagonist_model = protagonist_model.load_from_checkpoint(rl_prog_pth)
            st = time.time()
            length = min(adversary_tmp.policy_number, len(adver_strategy))
            rewards_whole_g = None
            rewards_rl = []
            for c in range(length):
                with open(cfg.evaluate_adv_dir+"/"+str(c)+"_model_params.txt", "w") as file:
                    for k in list(protagonist_model.policy.state_dict()):
                        print("params name:",k)
                        print(protagonist_model.policy.state_dict()[k])
                        file.write(k+"\n")
                        file.write(str(protagonist_model.policy.state_dict()[k].cpu().numpy()))

                adversary_model.policy, adversary_model.critic = adversary_tmp.get_policy_i(c)
                # 同样数据会进行多次play game，所以val_data需要保持原样，每次game：td_init重新加载
                # td_init = env.reset(test_data.clone()).to(device)
                rl_rewards = [] # [batch iter,]
                rl_rewards_all = None   # 1dim, [data_size's size

                bl_rewards = [] # batch iter,
                bl_rewards_var = [] # batch iter,
                bl_rewards_all = None       # batch's size

                test_data, stoch_dict, stoch_data = load_stoch_data(env, test_data, stoch_data_dir, stoch_data, c)
                test_data = test_data.to(device)
                test_dataset = TensorDictDataset(test_data)
                test_dl = DataLoader(test_dataset, batch_size=cfg.model_psro.test_batch_size, collate_fn=tensordict_collate_fn)

                stoch_dict = TensorDict(stoch_dict, batch_size=cfg.model_psro.test_batch_size, device=device)
                stoch_dataset = TensorDictDataset(stoch_dict)
                stoch_dl = DataLoader(stoch_dataset, batch_size=cfg.model_psro.test_batch_size, collate_fn=tensordict_collate_fn)

                for batch, stoch_batch in zip(test_dl, stoch_dl):
                    rl_res, bl_res, stoch_data = play_game(env, batch.clone(), stoch_batch, stoch_data, c, 
                                            protagonist_model, adversary_model, False, "", False,)
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
                
                 # record rewards of rl and baseline ,only under prog 0 
                
                # payoff = torch.tensor(rewards).mean().item()
                # payoff_underadv_rl.append(payoff)
                # print(f"c :{c}")
                # # 每张图上的psro-adv下的reward
                # if rewards_whole_g == None:
                #     rewards_whole_g = rewards_all[:, None]
                # else:
                #     rewards_whole_g = torch.cat((rewards_whole_g, rewards_all[:, None]), dim=1)
                rewards_rl.append(rl_rewards_all.cpu().tolist())
            rl_rewards_psro = eval_oneprog_adv_allgraph(rewards_rl, adver_strategy)
            rl_mean, rl_var = rl_rewards_psro.mean(), rl_rewards_psro.var()


            # reward_eval = eval_oneprog_adv(payoff_underadv_rl, adver_strategy)
            # rewards_graphs = eval_oneprog_adv_allgraph(rewards_whole_g[:, None].cpu().numpy(), adver_strategy)
            # eval_var = rewards_graphs.var()

            eval_time = time.time() - st
            print(f"reward mean is {rl_mean}, var is {rl_var}, eval time of rl is {eval_time} s")

            save_eval_pth = "eval_rl_withadv.npz"

            np.savez(cfg.evaluate_adv_dir+ '/'+save_eval_pth, 
                        rl_pth=rl_prog_pth,
                        eval_reward=rl_mean,
                        eval_var=rl_var,
                        eval_time=eval_time,
                        eval_all_r=rewards_rl,
                        eval_payoffs=rl_rewards_psro,       # key=value
                        eval_data=test_data_pth,
                        eval_adver_strategy=adver_strategy)  # 保
        
        payoff_underadv_baseline = []
        if cfg.get("eval_baseline_prog"):
            # 无adver的eval: 写一个payoff表，存每个prog的policy 在test数据下的结果，然后strategy来得到最后结果
            print("eval baseline with psro-adversary")
            prog_baseline = cfg.baseline_heur
            # baseline计算时间长，仅计算概率不为0的
            support_adv_stra_idx = [i for i, x in enumerate(adver_strategy) if x>0]
            support_adv_stra = [x for i, x in enumerate(adver_strategy) if x>0]
            print(support_adv_stra)
            st = time.time()
            rewards_whole_g = None
            for adv_idx in support_adv_stra_idx:
                adversary_model.policy, adversary_model.critic = adversary_tmp.get_policy_i(adv_idx)
                # td_init = env.reset(test_data.clone()).to(device)
                payoff = evaluate_baseline_withpsroadv(env, test_data_pth, prog_baseline, adversary_model, save_results=False)
                payoff_underadv_baseline.append(payoff["mean reward"].item())
                rewards = torch.tensor(payoff["rewards"]) if isinstance(payoff["rewards"], list) else payoff["rewards"]
                if rewards_whole_g == None:
                    rewards_whole_g = rewards
                else:
                    rewards_whole_g = torch.cat((rewards_whole_g, rewards), dim=0)
            reward_eval = eval_oneprog_adv(payoff_underadv_baseline, support_adv_stra)
            rewards_whole_g=rewards_whole_g.reshape(cfg.model_psro.test_data_size, len(support_adv_stra))
            rewards_graphs = eval_oneprog_adv_allgraph(rewards_whole_g[:, None].cpu().numpy(), support_adv_stra)
            var_eval = rewards_graphs.var()
            eval_time = time.time()-st 
            print(f"reward mean is {reward_eval}, var is {var_eval}, eval time of baseline:{prog_baseline} is {eval_time} s")
            save_eval_pth = "eval_baseline_"+prog_baseline+"_withadv.npz"

            np.savez(cfg.evaluate_adv_dir+ '/'+save_eval_pth, 
                    baseline=prog_baseline,
                    eval_reward=reward_eval,
                    var_eval=var_eval,
                    eval_time=eval_time,
                    eval_payoffs=payoff_underadv_baseline,       # key=value
                    eval_data=test_data_pth,
                    eval_adver_strategy=adver_strategy,
                    support_adv=support_adv_stra,
                    support_adv_idx=support_adv_stra_idx)  # 保
    return None, None
    






@hydra.main(version_base="1.3", config_path="../../configs", config_name="main_psro_frame.yaml")
def eval_other_withpsro_adv(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    eval_withpsroadv(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = utils.get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    eval_other_withpsro_adv()
