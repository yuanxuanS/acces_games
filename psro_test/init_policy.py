import nashpy as nash
import numpy as np
import torch
import collections
from rl4co.envs import SVRPEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.model_adversary.zoo.ppo.policy_conti import PPOContiAdvPolicy

from rl4co.model_adversary import PPOContiAdvModel
from rl4co.model_MA.psro_amppo import PSRO_AM_PPO
import hydra
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from lightning import LightningModule
from lightning import Callback, LightningModule

from rl4co.utils import RL4COTrainer
from rl4co import utils
from typing import List, Optional, Tuple
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) # 把当下project路径放到环境变量里面。
log = utils.get_pylogger(__name__)


def play_game(env, prog, adver):
    '''
        加载batch数据, 返回一次evaluation的reward: prog-adver
        prog: AM model
        adver: PPOContin model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(batch_size=[10]).to(device)
    prog_model = prog.to(device)
    adver_model = adver.to(device)
    
    out_adv = adver_model(td_init.clone(), phase="test", return_actions=True)
    td = env.reset_stochastic_demand(td_init, out_adv["action_adv"][..., None])    # env transition: get new real demand
    ret = prog_model(td)
    mean_reward = ret["reward"].mean().item()   # return scalar
    # print(mean_reward)
    reward = mean_reward
    return reward

def update_payoff(cfg, env, protagonist, adversary, payoff_prot, row_range, col_range):
    '''
        row 和col的policy 进行 play_game 得到这个位置的 payoff
    '''
    log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
    protagonist_model: LightningModule = hydra.utils.instantiate(cfg.model, env)
    log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
    adversary_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, env)
    
    orig_r = len(payoff_prot)
    orig_c = len(payoff_prot[0])

    
    for r in row_range:
        if r > orig_r - 1:
            new_row_payoff = []
        protagonist_model.policy = protagonist.get_policy_i(r)
        for c in col_range:

            adversary_model.policy, adversary_model.critic = adversary.get_policy_i(c)
            payoff = play_game(env, protagonist_model, adversary_model)
            if r > orig_r - 1:
                new_row_payoff.append(payoff)
            if c > orig_c -1 and r < orig_r -1:     # row新增行，包括c新增的一列
                payoff_prot[r].append(payoff)

        if r > orig_r - 1:
            payoff_prot.append(new_row_payoff)
    return payoff_prot
    
    


def nash_solver(payoff):
    """ given payoff matrix for a zero-sum normal-form game, numpy arrays
    return first mixed equilibrium (may be multiple)
    returns a tuple of numpy arrays """
    game = nash.Game(payoff)
    equilibria = game.lemke_howson_enumeration()
    equilibrium = next(equilibria, None)

    # Lemke-Howson couldn't find equilibrium OR
    # Lemke-Howson return error - game may be degenerate. try other approaches
    if equilibrium is None or (equilibrium[0].shape != (payoff.shape[0],) and equilibrium[1].shape != (payoffs.shape[0],)):
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
        return policy of strategy
        '''
        curr_policy = self.get_a_policy()
        worker_state_dict = [x.state_dict() for x in self.policies]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(self.strategy_length):
                key_sum = key_sum + worker_state_dict[i][key] * self.strategy[i]
            fed_state_dict[key] = key_sum
        #### update fed weights to fl model
        curr_policy.load_state_dict(fed_state_dict)
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
    
    def get_best_response(self, adversary, cfg, callbacks, logger):
        '''
        fix adversary and update Protagonist
        '''
        print("===== in protagonist bs ====")
    

        log.info("Instantiating trainer...")
        trainer: RL4COTrainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
        )
        # get protagonist's policy from strategy: params add
        cur_policy = self.get_curr_policy()     # AttentionModel's policy
        log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
        cur_model: LightningModule = hydra.utils.instantiate(cfg.model, self.env)
    
        # get adver's policy from its' strategy: params add
        adver_curr_policy, adver_curr_critic = adversary.get_curr_policy()
        log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
        adver_cur_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, self.env, 
                                                             policy=adver_curr_policy, critic=adver_curr_critic)


        # fix adver's params / not update
        # run until can't get more reward / reward converge
        psro_model: LightningModule = hydra.utils.instantiate(cfg.model_psro, self.env, cur_model, adver_cur_model, 
                                                              fix_protagonist=False,
                                                              fix_adversary=True)

        

        if cfg.get("train"):
            log.info("Starting training!")
            # if cfg.train_with_pretrain:
            #     model = model.load_from_checkpoint(cfg.train_with_pretrain)
            #     print("load psro pretrained")
            # else:
            #     print("no load psro pretrained")
            trainer.fit(model=psro_model, ckpt_path=cfg.get("ckpt_path"))
            # 取训练完的val reward（最后一次）
            curr_reward = psro_model.last_val_reward.to("cpu")
            # print("wait")

        return cur_policy, curr_reward

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
        return policy of strategy
        '''
        curr_policy, curr_critic = self.get_a_policy()

        worker_state_dict = [x.state_dict() for x in self.policies]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(self.strategy_length):     # 在best response后，strategy维度还没有改变。
                key_sum = key_sum + worker_state_dict[i][key] * self.strategy[i]
            fed_state_dict[key] = key_sum
        #### update fed weights to fl model
        curr_policy.load_state_dict(fed_state_dict)

        worker_state_dict = [x.state_dict() for x in self.correspond_critic]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(self.strategy_length):
                key_sum = key_sum + worker_state_dict[i][key] * self.strategy[i]
            fed_state_dict[key] = key_sum
        #### update fed weights to fl model
        curr_critic.load_state_dict(fed_state_dict)
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
    
    def get_best_response(self, protagonist, cfg, callbacks, logger):
        '''
        fix Protagonist and update adversary
        '''
        print("===== in adversary bs ====")
        

        log.info("Instantiating trainer...")
        trainer: RL4COTrainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
        )
        # get protagonist's policy from strategy: params add
        # get adver's policy from its' strategy: params add
        # fix adver's params / not update
        # run until can't get more reward / reward converge
        # get protagonist's policy from strategy: params add
        prog_policy = protagonist.get_curr_policy()
        log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
        prog_model: LightningModule = hydra.utils.instantiate(cfg.model, self.env, policy=prog_policy)
    
        # get adver's policy from its' strategy: params add
        cur_policy, cur_critic = self.get_curr_policy()     # PPOContinous's policy
        log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
        cur_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, self.env, 
                                                             policy=cur_policy, critic=cur_critic)
    

        # fix adver's params / not update
        # run until can't get more reward / reward converge
        psro_model: LightningModule = hydra.utils.instantiate(cfg.model_psro, self.env, prog_model,
                                                              cur_model, 
                                                              fix_protagonist=True,
                                                              fix_adversary=False)


        
        if cfg.get("train"):
            log.info("Starting training!")
            # if cfg.train_with_pretrain:
            #     model = model.load_from_checkpoint(cfg.train_with_pretrain)
            #     print("load psro pretrained")
            # else:
            #     print("no load psro pretrained")
            trainer.fit(model=psro_model, ckpt_path=cfg.get("ckpt_path"))
            curr_reward = psro_model.last_val_reward.to("cpu")
        return cur_policy, cur_critic, curr_reward

@hydra.main(version_base="1.3", config_path="../configs", config_name="main_psro_frame.yaml")
def run_psro(cfg: DictConfig):
    
    # trainer.logger = logger
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    epochs = 5
    epsilon = 0.01      # prog, adver的bs差距阈值，小于判断为 均衡
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    # 各自初始化1个policy
    protagonist = Protagonist(AttentionModel, AttentionModelPolicy, env)
    adversary = Adversary(PPOContiAdvModel, PPOContiAdvPolicy, CriticNetwork, env)
   
    protagonist.add_policy(protagonist.get_a_policy())
    policy_, critic_ = adversary.get_a_policy()
    adversary.add_policy(policy_, critic_)
    # 分别训练两个agent的初始policy
    protagonist.strategy = [1.]
    adversary.strategy = [1.]
    # 计算出来的bs的reward
    
    protagonist.policies[0], prog_bs_reward = protagonist.get_best_response(adversary, cfg, callbacks, logger)
    adversary.policies[0], adversary.correspond_critic[0], adver_bs_reward = adversary.get_best_response(protagonist,
                                                                                                          cfg, callbacks, logger)

    ## 计算初始payoff矩阵: row-prota, col-adver
    # 改：分别构建 prog, adv的算法model，for r c 遍历只需要重新load网络model
    log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
    protagonist_model: LightningModule = hydra.utils.instantiate(cfg.model, env)
    log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
    adversary_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, env)
    
    
    payoff_prot = []
    row_payoff = []
    protagonist_model.policy = protagonist.get_policy_i(0)
    adversary_model.policy, adversary_model.critic = adversary.get_policy_i(0)
    payoff = play_game(env, protagonist_model, adversary_model)
    row_payoff.append(payoff)
    payoff_prot.append(row_payoff)

    print("init payoff:", payoff_prot)
    print(protagonist.policy_number)
    for _ in range(epochs):

        bs_adversary, prog_bs_reward = protagonist.get_best_response(adversary, cfg, callbacks, logger)
        protagonist.add_policy(bs_adversary)

        bs_protagonist, bs_protagonist_critic, adver_bs_reward = adversary.get_best_response(protagonist, cfg, callbacks, logger)
        adversary.add_policy(bs_protagonist, bs_protagonist_critic)
        
        # 判断是否达到平衡
        if abs(prog_bs_reward - adver_bs_reward) < epsilon:
            print("get equalibium")
            break
        # 更新新加入policy的payoff矩阵
        row_range = [protagonist.policy_number - 1]
        col_range = range(adversary.policy_number)
        update_payoff(cfg, env, protagonist, adversary, payoff_prot, row_range, col_range)
        
        row_range = range(protagonist.policy_number -1)
        col_range = [adversary.policy_number - 1]
        update_payoff(cfg, env, protagonist, adversary, payoff_prot, row_range, col_range)


        ## 根据payoff, 求解现在的nash eq,得到player’s strategies
        # payoff_prot = [[0.5, 0.6], [0.1, 0.9]]
        eq = nash_solver(np.array(payoff_prot))
        # print(eq)
        protagonist_strategy, adversary_strategy = eq
        # print(protagonist_strategy)
        protagonist.update_strategy(protagonist_strategy)
        adversary.update_strategy(adversary_strategy)

    print("adver strategy: ", adversary.strategy)
    print("prog strategy: ", protagonist.strategy)
    print("final payoff", payoff_prot)
if __name__ == "__main__":
    run_psro()
