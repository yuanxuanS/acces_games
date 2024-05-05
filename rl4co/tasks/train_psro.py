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

from rl4co import utils
from rl4co.utils import RL4COTrainer
from memory_profiler import profile
from guppy import hpy
import numpy as np
import random
import nashpy as nash
import os
pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)

def play_game(env, td_init, prog, adver=None):
    '''
        加载batch数据, 返回一次evaluation的reward: prog-adver
        prog: AM model
        adver: PPOContin model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # td_init = env.reset(batch_size=[10]).to(device)
    prog_model = prog.to(device)

    if adver:
        adver_model = adver.to(device)
        out_adv = adver_model(td_init.clone(), phase="test", return_actions=True)
        td = env.reset_stochastic_var(td_init, out_adv["action_adv"][..., None])    # env transition: get new real demand
    else:
        td = td_init.clone()
    ret = prog_model(td)
    mean_reward = ret["reward"].mean().item()   # return scalar
    # print(mean_reward)
    reward = mean_reward
    return reward

def update_payoff(cfg, env, val_data_pth, protagonist, adversary, payoff_prot, row_range, col_range):
    '''
        row 和col的policy 进行 play_game 填充所有pair的 payoff:
        -----------------
        |xxxx | fill fill
        |-----  fill fill
        |fill fill fill
     '''
    val_data = env.load_data(val_data_pth)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(val_data.clone()).to(device)        # 同样数据会进行多次play game，所以val_data需要保持原样，每次game：td_init重新加载

    # log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
    protagonist_model: LightningModule = hydra.utils.instantiate(cfg.model, env)
    # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
    adversary_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, env)
    
    orig_r = len(payoff_prot)
    orig_c = len(payoff_prot[0])

    
    for r in row_range:
        if r > orig_r - 1:
            new_row_payoff = []
        protagonist_model.policy = protagonist.get_policy_i(r)
        for c in col_range:

            adversary_model.policy, adversary_model.critic = adversary.get_policy_i(c)
            td_init = env.reset(val_data.clone()).to(device)
            payoff = play_game(env, td_init, protagonist_model, adversary_model)
            if r > orig_r - 1:
                new_row_payoff.append(payoff)
            if c > orig_c -1 and r < orig_r -1:     # row新增行，包括c新增的一列
                payoff_prot[r].append(payoff)

        if r > orig_r - 1:
            payoff_prot.append(new_row_payoff)
    return payoff_prot
    

def eval(payoff, prog_strategy, adver_strategy):
    '''
    根据strategy和payoff表得到
    '''
    
    A = np.array(payoff)
    rps = nash.Game(A)
    assert len(prog_strategy) == len(adver_strategy), "strategy dims not equal"
    result = rps[prog_strategy, adver_strategy]
    return result[0]

def nash_solver(payoff):
    """ given payoff matrix for a zero-sum normal-form game, numpy arrays
    return first mixed equilibrium (may be multiple)
    returns a tuple of numpy arrays """
    game = nash.Game(payoff)
    with np.errstate(invalid='raise'):
        equilibria = game.lemke_howson_enumeration()
        equilibrium = next(equilibria, None)

        # Lemke-Howson couldn't find equilibrium OR
        # Lemke-Howson return error - game may be degenerate. try other approaches
        if equilibrium is None or (equilibrium[0].shape != (payoff.shape[0],) and equilibrium[1].shape != (payoff.shape[0],)):
            # try other
            print('\n\n\n\n\nuh oh! degenerate solution')
            print('payoffs are\n', payoff)
            equilibria = game.vertex_enumeration()
            equilibrium = next(equilibria)
            if equilibrium is None:
                print('\n\n\n\n\nuh oh x2! degenerate solution again!!')
                print('payoffs are\n', payoff)
                equilibria = game.support_enumeration()
                equilibrium = next(equilibria)

        assert equilibrium is not None
        return equilibrium


class Protagonist:
    def __init__(self, model, policy, env) -> None:
        self.model = model      # AttentionModel class
        self.policy = policy
        self.env = env
        self.policies = []
        self.correspond_baseline = []
        self.strategy = []
    
    def get_a_policy(self):
        return self.policy(self.env.name)
    
    def get_a_model(self):
        return self.model(self.env)
        
    def add_policy(self, policy):
        self.policies.append(policy)
    
    def get_policy_i(self, idx):
        assert idx > -1 and idx < self.policy_number, "idx exceeds range"
        assert type(idx) == int, "idx wrong type!"
        return self.policies[idx]
    
    def get_curr_policy(self):
        '''
        sample a policy from strategy
        '''
        sample_i = sample_strategy(self.strategy)
        sampled_policy = self.get_policy_i(sample_i)
        print(f"--- sample prog policy: {sample_i}")

        # copy参数，创建另一个instance
        curr_policy = self.get_a_policy()
        # worker_state_dict = [x.state_dict() for x in self.policies]
        state_dict = sampled_policy.state_dict()
        weight_keys = list(state_dict.keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            fed_state_dict[key] = state_dict[key]
        # #### update fed weights to fl model
        curr_policy.load_state_dict(fed_state_dict)

        # curr_policy = self.get_a_policy()
        # worker_state_dict = [x.state_dict() for x in self.policies]
        # weight_keys = list(worker_state_dict[0].keys())
        # fed_state_dict = collections.OrderedDict()
        # for key in weight_keys:
        #     key_sum = 0
        #     for i in range(self.strategy_length):
        #         key_sum = key_sum + worker_state_dict[i][key] * self.strategy[i]
        #     fed_state_dict[key] = key_sum
        # #### update fed weights to fl model
        # curr_policy.load_state_dict(fed_state_dict)
        # print("get curr policy done!")
        return curr_policy
    
    @property
    def policy_number(self):
        return len(self.policies)
    
    @property
    def strategy_length(self):
        return len(self.strategy)
    
    def update_strategy(self, strategy):
        assert sum(strategy) < 1. + 1e-5, "strategy prob is wrong!"
        if not isinstance(strategy, list):
            strategy = strategy.tolist()
        self.strategy = strategy
    
    def save_model_weights(self, pth_dir='./'):
        if not os.path.exists(pth_dir):
            os.mkdir(pth_dir)

        for i in range(len(self.policies)):
            torch.save(self.policies[i].state_dict(), f=pth_dir+"progPolicy_"+str(i)+".pth")
    
    def load_model_weights(self, load_dir):
        # 加载到policies中
        
        models = os.listdir(load_dir)
        log.info(f"{len(models)} policies to be loaded now.")
        assert len(self.policies) == 0, "polices are not empty but load more polices!"
        for model in models:
            model_w = torch.load(load_dir+model)
            tmp_policy = self.policy(self.env.name)
            tmp_policy.load_state_dict(model_w)
            
            self.policies.append(tmp_policy)
        

    def get_best_response(self, adversary, cfg, callbacks, logger, epoch, init=False):
        '''
        fix adversary and update Protagonist
        '''
        print("===== in protagonist bs ====")
        
        # max_epoch = 1
        if epoch == 0:
            max_epoch = cfg.prog_epoch1  #20 # 
        elif epoch > 0 and epoch < 5:
            max_epoch = cfg.prog_epoch2  #20  # 
        else: 
            max_epoch = 10
        
        # get protagonist's policy from strategy: adver用策略变化，prog更新policy
        cur_policy = self.get_curr_policy()     # sample a AttentionModel's policy
        # log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
        cur_model: LightningModule = hydra.utils.instantiate(cfg.model, self.env, policy=cur_policy)

        # get adver's policy from its' strategy: 
        adver_curr_policy, adver_curr_critic = adversary.get_curr_policy()
        # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
        adver_cur_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, self.env, 
                                                            policy=adver_curr_policy, critic=adver_curr_critic)

        # fix adver's params / not update
        # run until can't get more reward / reward converge
        psro_model: LightningModule = hydra.utils.instantiate(cfg.model_psro, self.env, cur_model, adver_cur_model, 
                                                            fix_protagonist=False,
                                                            fix_adversary=True,
                                                            prog_polices=self.policies,
                                                            adver_polices_and_critics=[adversary.policies, adversary.correspond_critic],
                                                            prog_strategy=self.strategy,
                                                            adver_strategy=adversary.strategy)
        log.info(f"Instantiating trainer in prog bs ...")
        trainer: RL4COTrainer = hydra.utils.instantiate(
            cfg.trainer,
            max_epochs=max_epoch,
            callbacks=callbacks,
            logger=logger,
        )

            # if e == 10:
            #     sche_prog, sche_adv = psro_model.lr_schedulers()
            #     if isinstance(sche_prog, torch.optim.lr_scheduler.MultiStepLR):
            #         sche_prog.step()

        if cfg.get("train"):
            # log.info("Starting training!")
            if init:
                if cfg.load_prog_from_path:
                    cur_model = cur_model.load_from_checkpoint(cfg.load_prog_from_path)
                    cur_policy = cur_model.policy
                    print("load psro pretrained")
                else:
                    trainer.fit(model=psro_model, ckpt_path=cfg.get("ckpt_path"))
            else:
                trainer.fit(model=psro_model, ckpt_path=cfg.get("ckpt_path"))
            # 取训练完的val reward（最后一次）
            curr_reward = psro_model.last_val_reward.to("cpu")      # val的batch_size改为10000， 只进行一次
            # print("wait")

        # 每次重新采样adver
        # adver_tmp_policy, adver_tmp_critic = adversary.get_curr_policy()
        # adver_cur_model.policy = adver_tmp_policy
        # adver_cur_model.critic = adver_tmp_critic

        return psro_model.protagonist.policy, curr_reward


class Adversary:
    def __init__(self, model, policy, critic, env) -> None:
        self.model = model
        self.policy = policy
        self.critic = critic
        self.env = env
        self.policies = []
        self.correspond_critic = []
        self.strategy = []
    
    def get_a_policy(self):
        return self.policy(self.env.name), self.critic(self.env.name)
    

    def get_a_model(self, opponent):
        return self.model(self.env, opponent=opponent)
    
    def add_policy(self, policy, critic):
        self.policies.append(policy)
        self.correspond_critic.append(critic)
    
    def get_policy_i(self, idx):
        assert idx > -1 and idx < self.policy_number, "idx exceeds range"
        assert type(idx) == int, "idx wrong type!"
        return self.policies[idx], self.correspond_critic[idx]
    
    def get_curr_policy(self):
        '''
        sample a policy from strategy
        '''
        sample_i = sample_strategy(self.strategy)
        sampled_policy, sampled_critic = self.get_policy_i(sample_i)
        print(f"--- sample afversary policy: {sample_i}")
        curr_policy, curr_critic = self.get_a_policy()
        # worker_state_dict = [x.state_dict() for x in self.policies]
        state_dict_policy = sampled_policy.state_dict()
        weight_keys_policy = list(state_dict_policy.keys())
        fed_state_dict_policy = collections.OrderedDict()

        for key in weight_keys_policy:
            fed_state_dict_policy[key] = state_dict_policy[key]
        curr_policy.load_state_dict(fed_state_dict_policy)

        state_dict_critic = sampled_critic.state_dict()
        weight_keys_critic = list(state_dict_critic.keys())
        fed_state_dict_critic = collections.OrderedDict()
        for key in weight_keys_critic:
            fed_state_dict_critic[key] = state_dict_critic[key]
        curr_critic.load_state_dict(fed_state_dict_critic)
        # #### update fed weights to fl model
        # curr_policy.load_state_dict(fed_state_dict)

        # worker_state_dict = [x.state_dict() for x in self.correspond_critic]
        # weight_keys = list(worker_state_dict[0].keys())
        # fed_state_dict = collections.OrderedDict()
        # for key in weight_keys:
        #     key_sum = 0
        #     for i in range(self.strategy_length):
        #         key_sum = key_sum + worker_state_dict[i][key] * self.strategy[i]
        #     fed_state_dict[key] = key_sum
        # #### update fed weights to fl model
        # curr_critic.load_state_dict(fed_state_dict)
        return curr_policy, curr_critic
    
    @property
    def policy_number(self):
        return len(self.policies)
    
    @property
    def strategy_length(self):
        return len(self.strategy)
    
    def update_strategy(self, strategy):
        assert sum(strategy) < 1. + 1e-5, "strategy prob is wrong!"
        assert len(strategy) == self.policy_number, "strategy size not equal to policies"
        if not isinstance(strategy, list):
            strategy = strategy.tolist()
        self.strategy = strategy
    
    def save_model_weights(self, pth_dir='./'):

        if not os.path.exists(pth_dir):
            os.mkdir(pth_dir)

        pth_dir_policy = pth_dir+"_policy/"
        if not os.path.exists(pth_dir_policy):
            os.mkdir(pth_dir_policy)

        pth_dir_critic = pth_dir+"_critic/"
        if not os.path.exists(pth_dir_critic):
            os.mkdir(pth_dir_critic)

        for i in range(len(self.policies)):
            torch.save(self.policies[i].state_dict(), f=pth_dir_policy+"adverPolicy_"+str(i)+".pth")
            torch.save(self.correspond_critic[i].state_dict(), f=pth_dir_critic+"adverCritic_"+str(i)+".pth")
    
    def load_model_weights(self, load_dir):
        # 加载到policies中
        load_dir_policy = load_dir + "_policy/"
        policies = os.listdir(load_dir_policy)
        log.info(f"{len(policies)} policies to be loaded now.")
        assert len(self.policies) == 0 and len(self.correspond_critic) == 0, "adver polices are not empty but load more polices!"
        for policy in policies:
            policy_w = torch.load(load_dir_policy+policy)
            tmp_policy = self.policy(self.env.name)
            tmp_policy.load_state_dict(policy_w)
            self.policies.append(tmp_policy)

        load_dir_critic = load_dir + "_critic/"
        critics = os.listdir(load_dir_critic)
        for critic in critics:
            critic_w = torch.load(load_dir_critic+critic)
            tmp_critic = self.critic(self.env.name)
            tmp_critic.load_state_dict(critic_w)
            self.correspond_critic.append(tmp_critic)

    def get_best_response(self, protagonist, cfg, callbacks, logger):
        '''
        fix Protagonist and update adversary
        '''
        print("===== in adversary bs ====")
        
        # get protagonist's policy from strategy: params add
        prog_policy = protagonist.get_curr_policy()
        # log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
        prog_model: LightningModule = hydra.utils.instantiate(cfg.model, self.env, policy=prog_policy)
    
        # get adver's policy from its' strategy: params add
        cur_policy, cur_critic = self.get_curr_policy()     # PPOContinous's policy
        # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
        cur_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, self.env, 
                                                             policy=cur_policy, critic=cur_critic)
    

        # fix prog's params / not update
        # run until can't get more reward / reward converge
        psro_model: LightningModule = hydra.utils.instantiate(cfg.model_psro, self.env, prog_model,
                                                              cur_model, 
                                                              fix_protagonist=True,
                                                              fix_adversary=False,
                                                              prog_polices=protagonist.policies,
                                                            adver_polices_and_critics=[self.policies, self.correspond_critic],
                                                            prog_strategy=protagonist.strategy,
                                                            adver_strategy=self.strategy)

        max_epoch = cfg.adver_epoch
        if cfg.get("train"):
            log.info(f"Instantiating trainer in adver bs ...")
            trainer: RL4COTrainer = hydra.utils.instantiate(
                    cfg.trainer,
                    max_epochs=max_epoch,
                    callbacks=callbacks,
                    logger=logger,
                )
            
            # log.info("Starting training!")
            
            trainer.fit(model=psro_model, ckpt_path=cfg.get("ckpt_path"))
            curr_reward = psro_model.last_val_reward.to("cpu")
        
        

        return psro_model.adversary.policy, psro_model.adversary.critic, curr_reward

def sample_strategy(distrib):
    strategy_is = random.choices(list(range(len(distrib))), weights=distrib)
    # print("in func", strategy_is)
    strategy_i = strategy_is[0]
    return strategy_i


@utils.task_wrapper
# @profile(stream=open('log_mem_cvrp50.log', 'w+'))
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
            val_data_pth = cfg.env.data_dir+"/"+cfg.env.val_file
            val_data = env.load_data(val_data_pth)

        
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
            # prog_bs_reward = play_game(env, protagonist_model, adversary_model)
            # 
            # adversary.policies[0], adversary.correspond_critic[0], adver_bs_reward = adversary.get_best_response(protagonist,
            #                                                                                                       cfg, callbacks, logger)


            payoff_prot = []
            row_payoff = []
            protagonist_model.policy = protagonist.get_policy_i(0)
            adversary_model.policy, adversary_model.critic = adversary.get_policy_i(0)

            
            td_init = env.reset(val_data.clone()).to(device)        # 同样数据会进行多次play game，所以val_data需要保持原样，每次game：td_init重新加载
            payoff = play_game(env, td_init, protagonist_model, adversary_model)
            row_payoff.append(payoff)
            payoff_prot.append(row_payoff)

            # compute nashconv
            utility_1 = payoff
            utility_2 = -payoff
            nashconv_lst = []

            # tmp
            prog_br_lst = []
            adver_br_lst = []

            # 
            # 存payoff表，两agent的strategy
            save_payoff_pth = logger[0].save_dir+'/psro/'
            if not os.path.exists(save_payoff_pth):
                os.mkdir(save_payoff_pth)

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

                bs_protagonist, bs_protagonist_critic, adver_bs_reward = adversary.get_best_response(protagonist, cfg, callbacks, logger)
                adversary.add_policy(bs_protagonist, bs_protagonist_critic)
                utility_2_br = -adver_bs_reward
                adver_br_lst.append(adver_bs_reward)

                # 判断是否达到平衡
                if abs(prog_bs_reward - adver_bs_reward) < epsilon:
                    print(f"get equalibium in {e} epoch !!! prog reward:{prog_bs_reward}, adver reward:{adver_bs_reward}")
                    # break
                else:
                    print(f"curr prog reward: {prog_bs_reward}, curr adver reward:{adver_bs_reward} in epoch {e}")

                # compute nashconv
                nashconv = utility_1_br - utility_1 + utility_2_br - utility_2
                nashconv_lst.append(nashconv)

                # 更新新加入policy的payoff矩阵
                row_range = [protagonist.policy_number - 1]
                col_range = range(adversary.policy_number)
                update_payoff(cfg, env, val_data_pth, protagonist, adversary, payoff_prot, row_range, col_range)
                
                row_range = range(protagonist.policy_number -1)
                col_range = [adversary.policy_number - 1]
                update_payoff(cfg, env, val_data_pth, protagonist, adversary, payoff_prot, row_range, col_range)

                

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

                if (e % 5 == 0) or (e == iterations - 1):
                    np.savez(save_payoff_pth+ 'info.npz', 
                        payoffs=payoff_prot,       # key=value
                        iter_reward=iter_reward,
                        nashconv_lst=nashconv_lst,     # nashconv
                        prog_br_lst=prog_br_lst,
                        adver_br_lst=adver_br_lst,
                        adver_strategy=adversary.strategy,
                        prog_strategy=protagonist.strategy)  # 保存的文件名，array_name是随便起的，相当于字典的key

            
        
        except Exception as e:
            print(f"error :{e}")

        finally:        # 
            protagonist.save_model_weights(logger[0].save_dir+"/models_weights/")   # logger是list
            adversary.save_model_weights(logger[0].save_dir+"/models_weights")
            
            print("adver strategy: ", adversary.strategy)
            print("prog strategy: ", protagonist.strategy)
            print("final payoff", payoff_prot)
            print("iteration reward", iter_reward)

        

    if cfg.get("evaluate"):
        # 加载psro prog和adver
        
        protagonist_tmp = Protagonist(AttentionModel, AttentionModelPolicy, env)
        protagonist_tmp.load_model_weights(cfg.evaluate_savedir+"/models_weights/")
        adversary_tmp = Adversary(PPOContiAdvModel, PPOContiAdvPolicy, CriticNetwork, env)
        adversary_tmp.load_model_weights(cfg.evaluate_savedir+"/models_weights")

        data = np.load(cfg.npz_pth)  # 加载
        payoff_tmp = data['payoffs']  # 引用保存好的数组，他的格式默认是numpy.array
        adver_strategy = data['adver_strategy']
        prog_strategy = data['prog_strategy']

        # load 对应环境的test数据
        test_data_pth = cfg.env.data_dir+"/"+cfg.env.test_file
        test_data = env.load_data(test_data_pth)
        # td_init = env.reset(test_data.clone()).to(device)        # 同样数据会进行多次play game，所以val_data需要保持原样，每次game：td_init重新加载
        
        payoff_eval = []
        if cfg.get("eval_withadv"):
            # 有adver的eval
            print("eval with adversary")
            for r in range(protagonist_tmp.policy_number):
                protagonist_model.policy = protagonist_tmp.get_policy_i(r)
                payoff_diff_adv = []
                for c in range(adversary_tmp.policy_number):

                    adversary_model.policy, adversary_model.critic = adversary_tmp.get_policy_i(c)
                    td_init = env.reset(test_data.clone()).to(device)
                    payoff = play_game(env, td_init, protagonist_model, adversary_model)
                    payoff_diff_adv.append(payoff)
                    print(f"r,c :{r},{c}")
                payoff_eval.append(payoff_diff_adv)
                
            reward_eval = eval(payoff_eval, prog_strategy, adver_strategy)
            save_eval_pth = "eval_withadv.npz"
        else:
            # 无adver的eval: 写一个payoff表，存每个prog的policy 在test数据下的结果，然后strategy来得到最后结果
            print("eval without adversary")
            for i in range(protagonist_tmp.policy_number):
                protagonist_model.policy = protagonist_tmp.get_policy_i(i)
                td_init = env.reset(test_data.clone()).to(device)
                payoff = play_game(env, td_init, protagonist_model, None)
                payoff_eval.append(payoff)
            reward_eval = eval_noadver(payoff_eval, prog_strategy)
            save_eval_pth = "eval_withoutadv.npz"
        print(f"eval reward: {reward_eval}")
        np.savez(cfg.evaluate_savedir+ '/'+save_eval_pth, 
                    eval_reward=reward_eval,
                    eval_payoffs=payoff_eval,       # key=value
                    eval_data=test_data_pth,
                    eval_adver_strategy=adver_strategy,
                    eval_prog_strategy=prog_strategy)  # 保
    return None, None
    
def eval_noadver(payoff, prog_strategy):
    '''
    根据strategy和payoff表得到, 无adver下
    '''
    A = np.array(payoff)
    rps = nash.Game(A)
    result = rps[prog_strategy, 1]
    return result[0]


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
