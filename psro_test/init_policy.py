import nashpy as nash
import numpy as np
import torch
import collections
from rl4co.envs import SVRPEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.model_adversary import PPOContiAdvModel
def play_game(env, prog, adver):
    '''
        加载batch数据, 返回一次evaluation的reward
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(batch_size=[10]).to(device)
    prog_policy = prog.to(device)
    adver_policy = adver.to(device)
    
    out_adv = adver_policy(td_init.clone(), phase="test", return_actions=True)
    td = env.reset_stochastic_demand(td_init, out_adv["action_adv"][..., None])    # env transition: get new real demand
    ret = prog_policy(td)
    mean_reward = ret["reward"].mean().item()   # return scalar
    # print(mean_reward)
    reward = mean_reward
    return reward

def update_payoff(row_range, col_range):
    return 


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
    def __init__(self, model, env) -> None:
        self.model = model
        self.env = env
        self.policies = []
        self.strategy = []
        
    def get_a_policy(self):
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
            for i in range(self.policy_number):
                key_sum = key_sum + worker_state_dict[i][key] * self.strategy[i]
            fed_state_dict[key] = key_sum
        #### update fed weights to fl model
        curr_policy.load_state_dict(fed_state_dict)
        # print("get curr policy done!")
        return curr_policy
    
    @property
    def policy_number(self):
        return len(self.policies)
    
    def update_strategy(self, strategy):
        assert sum(strategy) < 1. + 1e-5, "strategy prob is wrong!"
        if not isinstance(strategy, list):
            strategy = strategy.tolist()
        self.strategy = strategy
    
    def get_best_response(self, env, adversary):
        '''
        fix adversary and update Protagonist
        '''
        # get protagonist's policy from strategy: params add
        cur_policy = self.get_curr_policy()
        # get adver's policy from its' strategy: params add
        adver_curr_policy = adversary.get_curr_policy()
        # fix adver's params / not update
        # run until can't get more reward / reward converge
        return cur_policy

class Adversary:
    def __init__(self, model, env) -> None:
        self.model = model
        self.env = env
        self.policies = []
        self.strategy = []
        
    def get_a_policy(self, opponent):
        return self.model(self.env, opponent=opponent)
    
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
        curr_policy = self.get_a_policy(None)
        worker_state_dict = [x.state_dict() for x in self.policies]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(self.policy_number):
                key_sum = key_sum + worker_state_dict[i][key] * self.strategy[i]
            fed_state_dict[key] = key_sum
        #### update fed weights to fl model
        curr_policy.load_state_dict(fed_state_dict)
        return curr_policy
    
    @property
    def policy_number(self):
        return len(self.policies)
    
    def update_strategy(self, strategy):
        assert sum(strategy) < 1. + 1e-5, "strategy prob is wrong!"
        assert len(strategy) == self.policy_number, "strategy size not equal to policies"
        if not isinstance(strategy, list):
            strategy = strategy.tolist()
        self.strategy = strategy
    
    def get_best_response(self, env, protagonist):
        '''
        fix Protagonist and update adversary
        '''
        # get protagonist's policy from strategy: params add
        cur_policy = None
        # get adver's policy from its' strategy: params add
        # fix adver's params / not update
        # run until can't get more reward / reward converge
        return cur_policy
    
epochs = 1
env = SVRPEnv(num_loc=20) 
# 各自初始化n个policy
n = 2
protagonist = Protagonist(AttentionModel, env)
adversary = Adversary(PPOContiAdvModel, env)
for _ in range(n):
    protagonist.add_policy(protagonist.get_a_policy())
    adversary.add_policy(adversary.get_a_policy(None))

# 计算初始payoff矩阵: row-prota, col-adver
payoff_prot = []
for r in range(n):
    row_payoff = []
    for c in range(n):
        payoff = play_game(env, protagonist.get_policy_i(r), adversary.get_policy_i(c))
        row_payoff.append(payoff)
    payoff_prot.append(row_payoff)

print("init payoff:", payoff_prot)
print(protagonist.policy_number)
for _ in range(epochs):

    # 根据payoff, 求解现在的nash eq,得到player’s strategies
    # payoff_prot = [[0.5, 0.6], [0.1, 0.9]]
    eq = nash_solver(np.array(payoff_prot))
    print(eq)
    protagonist_strategy, adversary_strategy = eq
    # print(protagonist_strategy)
    protagonist.update_strategy(protagonist_strategy)
    adversary.update_strategy(adversary_strategy)


    bs_adversary = protagonist.get_best_response(env, adversary)
    adversary.add_policy(bs_adversary)

    bs_protagonist = adversary.get_best_response(env, protagonist)
    protagonist.add_policy(bs_protagonist)
    
    # 更新新加入policy的payoff矩阵
    row_range = [protagonist.policy_number - 1]
    col_range = range(adversary.policy_number)
    update_payoff(row_range, col_range)
    
    row_range = range(protagonist.policy_number)
    col_range = [adversary.policy_number - 1]
    update_payoff(row_range, col_range)