import numpy as np
import os
import torch
from rl4co import utils
log = utils.get_pylogger(__name__)
from rl4co.model_MA.am_amppo import AM_PPO
from rl4co.envs import SVRPEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.model_adversary.zoo.ppo.policy_conti import PPOContiAdvPolicy

from rl4co.model_adversary import PPOContiAdvModel
class DoubleOracleEval:
    def __init__(self, env, prog_model, prog_policy, adver_model, adver_policy, adver_critic) -> None:
        self.env = env
        # class
        self.prog_model = prog_model
        self.prog_policy = prog_policy
        self.adver_model = adver_model
        self.adver_policy = adver_policy
        self.adver_critic = adver_critic

        self.psro_payoff = []
        self.psro_val_data = ""
        self.psro_prog_strategy = []
        self.psro_adver_strategy = []
        self.psro_prog_policies = []
        self.psro_adver_policies = []
        self.psro_adver_critics = []
        
        self.rarl_prog_policy = None
        self.rarl_adver_policy = None
        self.rarl_adver_critic = None
        # rarl直接load model不是policy: load rarl的model后也存policy，方便和psro的进行play game 
        self.whole_prog_policies = []
        self.whole_adver_policies = []
        self.whole_adver_critics = []
        pass
    
    def _load_psro_payoffs(self, payoff_pth):
        # 加载psro的 payoff, strategy
        if not payoff_pth:
            log.warn("no payoff to be loaded")
            return
        data = np.load(payoff_pth+ 'info.npz')  
        self.psro_payoff = data['payoffs']  # 引用保存好的数组，他的格式默认是numpy.array
        self.psro_adver_strategy = data['adver_strategy']
        self.psro_prog_strategy = data['prog_strategy']

        self.psro_val_data = str(data['val_data'])  # npz加载为np array数据类型

    def _load_psro_prog_policies(self, psro_prog_dir):
        
        models = os.listdir(psro_prog_dir)
        print(f"{len(models)} policies to be loaded now.")
        assert len(self.psro_prog_policies) == 0, "polices are not empty but load more polices!"
        for model in models:
            model_w = torch.load(psro_prog_dir+"/"+model)
            tmp_policy = self.prog_policy(self.env.name)
            tmp_policy.load_state_dict(model_w)
            
            self.psro_prog_policies.append(tmp_policy)
    
    def _load_psro_adver(self, psro_adver_dir):
        load_dir_policy = psro_adver_dir + "_policy/"
        policies = os.listdir(load_dir_policy)
        print(f"{len(policies)} policies to be loaded now.")
        assert len(self.psro_adver_policies) == 0 and len(self.psro_adver_critics) == 0, "adver polices are not empty but load more polices!"
        for policy in policies:
            policy_w = torch.load(load_dir_policy+policy)
            tmp_policy = self.adver_policy(self.env.name)
            tmp_policy.load_state_dict(policy_w)
            self.psro_adver_policies.append(tmp_policy)

        load_dir_critic = psro_adver_dir + "_critic/"
        critics = os.listdir(load_dir_critic)
        for critic in critics:
            critic_w = torch.load(load_dir_critic+critic)
            tmp_critic = self.adver_critic(self.env.name)
            tmp_critic.load_state_dict(critic_w)
            self.psro_adver_critics.append(tmp_critic)

    def load_psro_results(self, psro_payoff_dir, psro_prog_dir, psro_adver_dir):
        if psro_prog_dir != psro_adver_dir:
            log.warn("psro prog and adver from different training! Mustn't load payoff")
            psro_payoff_pth = None
        
        assert len(self.psro_prog_policies) == len(self.psro_adver_policies), "curr prog and adver policies not equal!"

        self._load_psro_payoffs(psro_payoff_dir)
        self._load_psro_prog_policies(psro_prog_dir)
        self._load_psro_adver(psro_adver_dir)

    def load_rarl_prog(self, rarl_prog_pth, rarl_adver_pth):
        # rarl把AM_ppo class整个都save，包括prog和adver, 所以需要加载该class来提取
        tmp_prog = self.prog_model(self.env)
        tmp_adver = self.adver_model(self.env, opponent=tmp_prog)
        model_tmp = AM_PPO(self.env, tmp_prog, tmp_adver)
        model_tmp = model_tmp.load_from_checkpoint(rarl_prog_pth)
        self.rarl_prog_policy = model_tmp.protagonist.policy

        if rarl_adver_pth == rarl_prog_pth:     # 如果同一个rarl训练出来的，直接加载
            self.rarl_adver_policy = model_tmp.adversary.policy
            self.rarl_adver_critic = model_tmp.adversary.critic
        else:
            model_tmp2 = AM_PPO(self.env, tmp_prog, tmp_adver)
            model_tmp2 = model_tmp.load_from_checkpoint(rarl_adver_pth)
            self.rarl_adver_policy = model_tmp2.adversary.policy
            self.rarl_adver_critic = model_tmp2.adversary.critic
    
    
    def get_whole_payoff(self):
        '''
        得到psro和rarl加载到一起的payoff表
        '''
        
        # get whole prog: psro, rarl
        self.whole_prog_policies = [policy for policy in self.psro_prog_policies]
        self.whole_prog_policies.append(self.rarl_prog_policy)      # 默认rarl只有一个
        self.psro_range = list(range(len(self.psro_prog_policies)))     # psro police的索引

        self.whole_adver_policies = [policy for policy in self.psro_adver_policies]
        self.whole_adver_critics = [critic for critic in self.psro_adver_critics]
        self.whole_adver_policies.append(self.rarl_adver_policy)
        self.whole_adver_critics.append(self.rarl_adver_critic)
        self.rarl_range = [len(self.whole_adver_policies) - 1]

        ## 更新payoff
        self.whole_payoff = [list(row.copy()) for row in self.psro_payoff]
        # 补全rarl的payoff
        self.whole_payoff = self.update_payoff(self.whole_payoff, self.rarl_range, self.psro_range)
        self.whole_payoff = self.update_payoff(self.whole_payoff, list(range(len(self.whole_prog_policies))), self.rarl_range)
        print("final payoff:", self.whole_payoff)
        return self.whole_payoff

    
    def update_payoff(self, payoff_prot, row_range, col_range):
        '''
            row 和col的policy 进行 play_game 填充所有pair的 payoff:
            -      col      -
            |xxxx | 
            |-----  
        row |fill fill fill
        '''
        # 加载好数据，用于后面的eval
        val_data = self.env.load_data(self.psro_val_data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        td_init = self.env.reset(val_data.clone()).to(device)        # 同样数据会进行多次play game，所以val_data需要保持原样，每次game：td_init重新加载
        # 初始化 两agent的 model class
        protagonist_model = self.prog_model(self.env)
        adversary_model = self.adver_model(self.env, opponent=protagonist_model)
        
        orig_r = len(payoff_prot)
        orig_c = len(payoff_prot[0])

        
        for r in row_range:
            if r > orig_r - 1:
                new_row_payoff = []
            protagonist_model.policy = self.whole_prog_policies[r]
            for c in col_range:

                adversary_model.policy, adversary_model.critic = self.whole_adver_policies[c], self.whole_adver_critics[c]
                
                td_init = self.env.reset(val_data.clone()).to(device)        # 
                payoff = DoubleOracleEval.play_game(self.env, td_init, protagonist_model, adversary_model)
                if r > orig_r - 1:
                    new_row_payoff.append(payoff)
                if c > orig_c -1 and r < orig_r -1:     # row新增行，包括c新增的一列
                    payoff_prot[r].append(payoff)

            if r > orig_r - 1:
                payoff_prot.append(new_row_payoff)
        return payoff_prot
    
    @staticmethod
    def play_game(env, td_init, prog, adver):
        '''
            加载batch数据, 返回一次evaluation的reward: prog-adver
            prog: AM model
            adver: PPOContin model
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prog_model = prog.to(device)
        adver_model = adver.to(device)
        
        out_adv = adver_model(td_init.clone(), phase="test", return_actions=True)
        td = env.reset_stochastic_demand(td_init, out_adv["action_adv"][..., None])    # env transition: get new real demand
        ret = prog_model(td)
        mean_reward = ret["reward"].mean().item()   # return scalar
        # print(mean_reward)
        reward = mean_reward
        return reward
    


env = SVRPEnv(num_loc=20)
doEval = DoubleOracleEval(env, prog_model=AttentionModel, prog_policy=AttentionModelPolicy,
                          adver_model=PPOContiAdvModel, adver_policy=PPOContiAdvPolicy, adver_critic=CriticNetwork)
psro_payoff_dir = "/home/panpan/rl4co/logs/train_psro/runs/svrp20/am-svrp20/2024-03-26_14-10-29/psro/"
psro_prog_dir = "/home/panpan/rl4co/logs/train_psro/runs/svrp20/am-svrp20/2024-03-26_14-10-29/models_weights"
psro_adver_dir = "/home/panpan/rl4co/logs/train_psro/runs/svrp20/am-svrp20/2024-03-26_14-10-29/models_weights"
doEval.load_psro_results(psro_payoff_dir=psro_payoff_dir,
                         psro_prog_dir=psro_prog_dir,
                         psro_adver_dir=psro_adver_dir)      # 加载psro
rarl_pth = "/home/panpan/rl4co/logs/train_rarl/runs/svrp20/am-svrp20/2024-03-12_14-14-24/rl4co-adv/049hb7g5/checkpoints/epoch=99-step=625000.ckpt"
doEval.load_rarl_prog(rarl_pth, rarl_pth)     # 加载rarl

doEval.get_whole_payoff()       # 补全最后的payoffs

