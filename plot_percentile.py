import random
import matplotlib.pyplot as plt
import numpy as np
seed_times = 20
rarl = [-8.04, -8.04, -8.05, -8.06, -8.07, -8.07]
rl = [-8.32, -8.33, -8.33, -8.35, -8.36, -8.36]



def plot_reward(reward_lsts, metric="VaR"):
    '''
    reward_lsts: list of lists
    VaR: value at risk, VaR(p) means the prob of value of reward greater than VaR(p) is 1-p. Mostly prob p is under VaR(p)
    CVaR: Conditional value at risk, VaR(p) / sum(p), at current all prob. 
    metric: "VaR" or "CVaR"
    '''
    plt.figure()
    
    for reward_lst in reward_lsts:
        seed_times = len(reward_lst)
        print(f"before sort {reward_lst}")
        reward_lst = sorted(reward_lst)      # x = 其中的一个[reward, seed, 'str']
        print(f"after sort {reward_lst}")

        x = [(i+1)/seed_times for i in range(seed_times)]
        x = np.array(x)
        y = np.percentile(reward_lst, x*100, method="lower")      # 第二个参数：分位数:0-100
        if metric == "VaR":
            pass
        elif metric == "CVaR":
            y_CVaR = []
            for i in range(len(y)):
                y_CVaR.append(np.mean(y[:i+1]))
            y = y_CVaR
        else:
            raise NameError("metric name is wrong!")
        print(f" x is {x} \ny is {y}")
        
        plt.plot(x, y)
    plt.legend(["RARL agent", "RL agent"])
    plt.title(metric+" of reward (negative route length)")
    plt.savefig("./reward_"+metric+"_2.png")

plot_reward([rarl, rl], "CVaR")

