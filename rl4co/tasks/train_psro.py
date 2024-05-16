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

from rl4co.model_adversary import PPOContiAdvModel
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
from rl4co.data.dataset import tensordict_collate_fn


from rl4co import utils
from rl4co.utils import RL4COTrainer
from memory_profiler import profile
from guppy import hpy
import numpy as np
import random
import nashpy as nash
import os
import time
from rl4co.model_MA import Protagonist, Adversary
from rl4co.model_MA.utils_psro import *

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)




@utils.task_wrapper
@profile(stream=open('log_mem_svrp50_psro.log', 'w+'))
def run(cfg: DictConfig) -> Tuple[dict, dict]:
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

    # if self.data_cfg["generate_data"]:
    data_cfg = {
            "val_data_size": cfg.model_psro.val_data_size,
            "test_data_size": cfg.model_psro.test_data_size,
        }
    generate_default_datasets(data_dir=cfg.paths.data_dir, data_cfg=data_cfg)
    # generate_dataset(data_dir=cfg.env.data_dir, dataset_size=data_cfg["val_data_size"],name="val", problem=cfg.env.name, seed=4321)
    # generate_dataset(data_dir=cfg.env.data_dir, dataset_size=data_cfg["test_data_size"], name="test", problem=cfg.env.name, seed=1234)

    # 改：分别构建 prog, adv的算法model，for r c 遍历只需要重新load网络model
    protagonist_model: LightningModule = hydra.utils.instantiate(cfg.model, env)
    # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
    adversary_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, env)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("train"):

        try:
            val_data_pth = cfg.env.data_dir+"/"+cfg.env.test_file
            val_data = env.load_data(val_data_pth)
            val_dataset = TensorDictDataset(val_data)
            val_dl = DataLoader(val_dataset, batch_size=cfg.model_psro.test_batch_size, collate_fn=tensordict_collate_fn)

            # 存payoff表，两agent的strategy
            save_payoff_pth = logger[0].save_dir+'/psro/'
            if not os.path.exists(save_payoff_pth):
                os.mkdir(save_payoff_pth)

            stoch_data_dir = logger[0].save_dir+'/adv_stoch_data/'
            if not os.path.exists(stoch_data_dir):
                os.mkdir(stoch_data_dir)

            # 各自初始化1个policy
            protagonist = Protagonist(AttentionModel, AttentionModelPolicy, env)
            adversary = Adversary(PPOContiAdvModel, PPOContiAdvPolicy, CriticNetwork, env)
    
            protagonist.add_policy(protagonist.get_a_policy())
            policy_, critic_ = adversary.get_a_policy()
            adversary.add_policy(policy_, critic_)
            # 分别训练两个agent的初始policy
            protagonist.strategy = [1.]
            adversary.strategy = [1.]
        
    
            ## 计算初始payoff矩阵: row-prota, col-adver
            
            # log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
            
            # 计算出来的bs的reward
            # protagonist.policies[0], prog_bs_reward = protagonist.get_best_response(adversary, cfg, callbacks, logger)
            if cfg.load_prog_from_path:
                tmp_model: LightningModule = hydra.utils.instantiate(cfg.model, env)
                tmp_model = tmp_model.load_from_checkpoint(cfg.load_prog_from_path)
                protagonist.policies[0] = tmp_model.policy
            protagonist_model.policy = protagonist.get_policy_i(0)


            payoff_prot = []
            row_payoff = []
            protagonist_model.policy = protagonist.get_policy_i(0)
            adversary_model.policy, adversary_model.critic = adversary.get_policy_i(0)
            protagonist.save_a_model_weights(logger[0].save_dir+"/models_weights/", 0, protagonist_model.policy)
            adversary.save_a_model_weights(logger[0].save_dir+"/models_weights", 0, adversary_model.policy, adversary_model.critic)
            
            
            # td_init = env.reset(val_data.clone()).to(device)        # 同样数据会进行多次play game，所以val_data需要保持原样，每次game：td_init重新加载

            # rewards = torch.tensor([play_game(env, batch.clone(), protagonist_model, adversary_model)[0] for batch in val_dl])
            # payoff = rewards.mean().item()
            # row_payoff.append(payoff)
            # payoff_prot.append(row_payoff)
            for sk in stochdata_key_mapping[env.name]:
                stoch_data = {sk: {}}     # {"stochastic_demand": {0: tensordict (data_size, ), 1:}}
            rewards_rl = []
            rewards_baseline = []
            payoff_prot, rewards_rl, rewards_baseline, stoch_data = update_payoff(cfg, env, val_data_pth, 
                                                                      stoch_data, stoch_data_dir,
                                                                        protagonist, adversary, payoff_prot,
                                        [0], [0], rewards_rl, rewards_baseline, cfg.eval_baseline, save_payoff_pth,)

            rl_rewards_psro = eval_oneprog_adv_allgraph(rewards_rl, adversary.strategy)
            rl_mean, rl_var = rl_rewards_psro.mean(), rl_rewards_psro.var()

            if cfg.eval_baseline:
                bl_rewards_psro = eval_oneprog_adv_allgraph(rewards_baseline, adversary.strategy)
                bl_mean, bl_var = bl_rewards_psro.mean(), bl_rewards_psro.var()
            else:
                bl_rewards_psro = None
                bl_mean, bl_var = None, None
            np.savez(save_payoff_pth+"rl_bl_byadv_iter"+str(0)+".npz",
                        rl_rewards=rewards_rl,  # 所哟图上，不同adv下，rewards
                        bl_rewards=rewards_baseline,
                        rl_rewards_psro=rl_rewards_psro,    # adv在所有图上rewards
                        bl_rewards_psro=bl_rewards_psro,
                        prog_strategy=protagonist.strategy,
                        adver_strategy=adversary.strategy,
                        rl_mean=rl_mean,    # 所有图mean
                        rl_var=rl_var,
                        bl_mean=bl_mean,
                        bl_var=bl_var)
            
            # compute nashconv
            payoff = payoff_prot[0][0]
            utility_1 = payoff
            utility_2 = -payoff
            nashconv_lst = []

            # tmp
            prog_br_lst = []
            adver_br_lst = []

            # 
            

            print("init payoff:", payoff_prot)
            print(protagonist.policy_number)
            iter_reward = []
            iterations = cfg.iters
            epsilon = cfg.epsilon      # prog, adver的bs差距阈值，小于判断为 均衡
            for e in range(iterations):

                log.info(f" psro training epoch {e}")
                bs_adversary, prog_bs_reward = protagonist.get_best_response(adversary, cfg, callbacks, logger, epoch=e)
                protagonist.add_policy(bs_adversary)
                utility_1_br = prog_bs_reward
                prog_br_lst.append(prog_bs_reward)
                protagonist.save_a_model_weights(logger[0].save_dir+"/models_weights/", e+1, bs_adversary)

                bs_protagonist, bs_protagonist_critic, adver_bs_reward = adversary.get_best_response(protagonist, cfg, callbacks, logger)
                adversary.add_policy(bs_protagonist, bs_protagonist_critic)
                utility_2_br = -adver_bs_reward
                adver_br_lst.append(adver_bs_reward)
                adversary.save_a_model_weights(logger[0].save_dir+"/models_weights", e+1, bs_protagonist, bs_protagonist_critic)

                # 判断是否达到平衡
                if abs(prog_bs_reward - adver_bs_reward) < epsilon:
                    print(f"get equalibium in {e} epoch !!! prog reward:{prog_bs_reward}, adver reward:{adver_bs_reward}")
                    # break
                else:
                    print(f"curr prog reward: {prog_bs_reward}, curr adver reward:{adver_bs_reward} in epoch {e}")

                # compute nashconv
                nashconv = utility_1_br - utility_1 + utility_2_br - utility_2
                nashconv_lst.append(nashconv)

                # 更新新加入policy的payoff矩阵: 先更新hang列，再shu列
                row_range = [protagonist.policy_number - 1]
                col_range = range(adversary.policy_number)
                # update_payoff(cfg, env, val_data_pth, protagonist, adversary, payoff_prot, row_range, col_range)
                # print(f"shoud no change: before update payoff in {e}: num of rl and bl eval data {len(rewards_rl)} {len(rewards_baseline)}")
                payoff_prot, rewards_rl, rewards_baseline, stoch_data = update_payoff(cfg, env, val_data_pth, stoch_data, stoch_data_dir,
                                                                                      protagonist, adversary, payoff_prot,
                                        row_range, col_range, rewards_rl, rewards_baseline, cfg.eval_baseline, save_payoff_pth)
                # print(f"shoud no change: after update payoff in {e}: num of rl and bl eval data {len(rewards_rl)}  {len(rewards_baseline)}")

                row_range = range(protagonist.policy_number -1)
                col_range = [adversary.policy_number - 1]
                print(f" before update payoff in {e}: num of rl and bl eval data {len(rewards_rl)}  {len(rewards_baseline)}")
                payoff_prot, rewards_rl, rewards_baseline, stoch_data = update_payoff(cfg, env, val_data_pth, stoch_data, stoch_data_dir,
                                                                                      protagonist, adversary, payoff_prot,
                                        row_range, col_range, rewards_rl, rewards_baseline, cfg.eval_baseline, save_payoff_pth)
                print(f" after update payoff in {e}: num of rl and bl eval data {len(rewards_rl)}  {len(rewards_baseline)}")
                
                print(f"payoff_prot: {payoff_prot}")
                print(f"rewards_rl: {rewards_rl}")
                print(f"rewards_baseline: {rewards_baseline}")

                ## 根据payoff, 求解现在的nash eq,得到player’s strategies
                # payoff_prot = [[0.5, 0.6], [0.1, 0.9]]
                eq = nash_solver(np.array(payoff_prot))
                print(eq)
                protagonist_strategy, adversary_strategy = eq
                # print(protagonist_strategy)
                protagonist.update_strategy(protagonist_strategy)
                adversary.update_strategy(adversary_strategy)

                # 测试现在的reward
                curr_reward = eval(payoff_prot, protagonist.strategy, adversary.strategy)
                iter_reward.append(curr_reward)
                log.info(f"curr reward is {curr_reward}")

                # update utility
                utility_1 = curr_reward
                utility_2 = -curr_reward

                # 每轮更新一次rl和bl的mean，var： reward_rl [iter, datasize,]
                rl_rewards_psro = eval_oneprog_adv_allgraph(rewards_rl, adversary_strategy)
                rl_mean, rl_var = rl_rewards_psro.mean(), rl_rewards_psro.var()

                if cfg.eval_baseline:
                    bl_rewards_psro = eval_oneprog_adv_allgraph(rewards_baseline, adversary_strategy)
                    bl_mean, bl_var = bl_rewards_psro.mean(), bl_rewards_psro.var()
                else:
                    bl_rewards_psro = None
                    bl_mean, bl_var = None, None

                np.savez(save_payoff_pth+ 'info.npz', 
                    payoffs=payoff_prot,       # key=value
                    iter_reward=iter_reward,
                    nashconv_lst=nashconv_lst,     # nashconv
                    prog_br_lst=prog_br_lst,
                    adver_br_lst=adver_br_lst,
                    adver_strategy=adversary.strategy,
                    prog_strategy=protagonist.strategy,
                    )  # 保存的文件名，array_name是随便起的，相当于字典的key

                np.savez(save_payoff_pth+"rl_bl_byadv_iter"+str(e+1)+".npz",
                        rl_rewards=rewards_rl,  # 所有图上，不同adv下，rewards， [adv/iter个数, datasize]
                        bl_rewards=rewards_baseline,
                        rl_rewards_psro=rl_rewards_psro,    # adv在所有图上rewards,乘策略后 [datasize]
                        bl_rewards_psro=bl_rewards_psro,
                        adver_strategy=adversary.strategy,
                        prog_strategy=protagonist.strategy,
                        rl_mean=rl_mean,    # adv下所有图mean rewards,  和payoff第一行应该有出入（除非asv strategy是纯策略） 
                        rl_var=rl_var,
                        bl_mean=bl_mean,
                        bl_var=bl_var)
        
        except Exception as e:
            print(f"error :{e}")

        finally:        # 
            protagonist.save_model_weights(logger[0].save_dir+"/models_weights_final/")   # logger是list
            adversary.save_model_weights(logger[0].save_dir+"/models_weights_final")
            
            np.savez(save_payoff_pth+ 'info_final.npz', 
                    payoffs=payoff_prot,       # key=value
                    iter_reward=iter_reward,
                    nashconv_lst=nashconv_lst,     # nashconv
                    prog_br_lst=prog_br_lst,
                    adver_br_lst=adver_br_lst,
                    adver_strategy=adversary.strategy,
                    prog_strategy=protagonist.strategy,
                    rl_rewards=rewards_rl,
                    bl_rewards=rewards_baseline)  # 保存的文件名，array_name是随便起的，相当于字典的key
            
            print("adver strategy: ", adversary.strategy)
            print("prog strategy: ", protagonist.strategy)
            print("final payoff", payoff_prot)
            print("iteration reward", iter_reward)

        

    if cfg.get("evaluate"):
        # 加载psro prog和adver
        
        protagonist_tmp = Protagonist(AttentionModel, AttentionModelPolicy, env)
        protagonist_tmp.load_model_weights(cfg.evaluate_prog_dir+"/models_weights/")
        adversary_tmp = Adversary(PPOContiAdvModel, PPOContiAdvPolicy, CriticNetwork, env)
        adversary_tmp.load_model_weights(cfg.evaluate_prog_dir+"/models_weights")

        data = np.load(cfg.prog_npz_pth)  # 加载
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
        test_data_pth = cfg.env.data_dir+"/"+cfg.env.val_file
        print(f"load testdata from {test_data_pth}")
        test_data = env.load_data(test_data_pth)
        print(f" test size is {test_data.batch_size}")
        # td_init = env.reset(test_data.clone()).to(device)        # 同样数据会进行多次play game，所以val_data需要保持原样，每次game：td_init重新加载
        test_dataset = TensorDictDataset(test_data)
        test_dl = DataLoader(test_dataset, batch_size=cfg.model_psro.val_batch_size, collate_fn=tensordict_collate_fn)

        payoff_eval = []
        payoff_eval_var = []
        if cfg.get("eval_withadv"):
            # 有adver的eval
            print("eval with adversary")
            if cfg.another_adv:
                adversary_tmp.load_model_weights(cfg.evaluate_adv_dir+"/models_weights")
                data_adv = np.load(cfg.adv_npz_pth)  # 加载
                adver_strategy = data_adv['adver_strategy']
                if adversary_tmp.no_zeroth:
                    adver_strategy = adver_strategy[1:]
                another = "_another_"
            else:
                another = "_this_"
            st = time.time()
            rewards_whole_g = None
            for r in range(len(prog_strategy)):     # 当error停止，出现policy个数-strategy个数 = 1
                protagonist_model.policy = protagonist_tmp.get_policy_i(r)
                payoff_diff_adv = []
                payoff_diff_adv_var = []
                for c in range(len(adver_strategy)):

                    adversary_model.policy, adversary_model.critic = adversary_tmp.get_policy_i(c)
                    # td_init = env.reset(test_data.clone()).to(device)
                    # payoff = play_game(env, td_init, protagonist_model, adversary_model)
                    rewards = []
                    rewards_all = None
                    for batch in test_dl:
                        re, re_allg = play_game(env, batch.clone(), protagonist_model, adversary_model)
                        rewards.append(re)
                        if rewards_all == None:
                            rewards_all = re_allg
                        else:
                            rewards_all = torch.cat((rewards_all, re_allg), dim=0)
                    payoff = torch.tensor(rewards).mean().item()
                    payoff_diff_adv.append(payoff)
                    print(f"r,c :{r},{c}")
                    if rewards_whole_g == None:
                        rewards_whole_g = rewards_all[:, None]
                    else:
                        rewards_whole_g = torch.cat((rewards_whole_g, rewards_all[:, None]), dim=1)

                    # 
                payoff_eval.append(payoff_diff_adv)

                
            rewards_whole_g=rewards_whole_g.reshape(cfg.model_psro.val_data_size, len(prog_strategy), len(adver_strategy))
            reward_eval = eval(payoff_eval, prog_strategy, adver_strategy)
            rewards_graphs = eval_allgraph(rewards_whole_g.cpu().numpy(), prog_strategy, adver_strategy)
            eval_var = rewards_graphs.var()

            time_ = time.time()-st
            save_eval_pth = "eval_with"+another+"adv.npz"
        else:
            # 无adver的eval: 写一个payoff表，存每个prog的policy 在test数据下的结果，然后strategy来得到最后结果
            print("eval without adversary")
            st = time.time()
            rewards_whole_g = None
            for i in range(len(prog_strategy)):
                protagonist_model.policy = protagonist_tmp.get_policy_i(i)
                # td_init = env.reset(test_data.clone()).to(device)
                # payoff = play_game(env, td_init, protagonist_model, None)
                rewards = []
                rewards_all = None
                for batch in test_dl:
                    re, re_allg = play_game(env, batch.clone(), protagonist_model, adversary_model)
                    rewards.append(re)
                    if rewards_all == None:
                        rewards_all = re_allg
                    else:
                        rewards_all = torch.cat((rewards_all, re_allg), dim=0)
                payoff = torch.tensor(rewards).mean().item()
                payoff_eval.append(payoff)
                if rewards_whole_g == None:
                    rewards_whole_g = rewards_all[:, None]
                else:
                    rewards_whole_g = torch.cat((rewards_whole_g, rewards_all[:, None]), dim=1)
            reward_eval = eval_noadver(payoff_eval, prog_strategy)
            # 在instance/图 上分别计算psro下的reward， 再计算方差
            rewards_graphs = eval_noadver_allgraph(rewards_whole_g.cpu().numpy(), prog_strategy)
            eval_var = rewards_graphs.var()

            time_ = time.time()-st
            save_eval_pth = "eval_withoutadv.npz"
        print(f"eval reward: {reward_eval}, var is {eval_var}, time is {time_}")
        adv_pth = cfg.evaluate_adv_dir if cfg.another_adv else None
        np.savez(cfg.ckpt_psro_path+ '/'+save_eval_pth, 
                    adv_pth=adv_pth,
                    eval_reward=reward_eval,
                    eval_var=eval_var,
                    eval_payoffs=payoff_eval,       # key=value
                    eval_vars=eval_var,
                    eval_time=time_,
                    eval_data=test_data_pth,
                    eval_adver_strategy=adver_strategy,
                    eval_prog_strategy=prog_strategy)  # 保
    return None, None

def eval_allgraph(rewards_graph, prog_strategy, adver_strategy):
    '''
    rewards_graph: numpy array, shape:(graph_nums, pro_policy_Num, adv_policy_Num)
    adver_strategy: numpy array, shape:(adv_policy_Num)
    prog_strategy: numpy array, shape:(pro_policy_Num)
    '''
    reward_prog_graph = np.matmul(rewards_graph, np.array(adver_strategy))
    reward_adver_graph = np.matmul(reward_prog_graph, np.array(prog_strategy))
    return reward_adver_graph

def eval_noadver(payoff, prog_strategy):
    '''
    根据strategy和payoff表得到, 无adver下
    '''
    A = np.array(payoff)
    rps = nash.Game(A)
    result = rps[prog_strategy, 1]
    return result[0]

def eval_noadver_allgraph(rewards_graph, prog_strategy):
    '''
    rewards_graph: numpy array, shape:(graph_nums, pro_policy_Num)
    prog_strategy: numpy array, shape:(pro_policy_Num)
    '''
    reward_prog_graphs = np.matmul(rewards_graph, np.array(prog_strategy))
    return reward_prog_graphs

@hydra.main(version_base="1.3", config_path="../../configs", config_name="main_psro_frame.yaml")
def train_psro(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    print("this is in psro train")
    utils.extras(cfg)

    # train the model
    run(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = utils.get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    train_psro()
