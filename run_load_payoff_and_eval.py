from rl4co.model_MA.utils_psro import eval
import numpy as np
npz_pth = "/home/panpan/rl4co/logs/train_psro/runs/opsa20/am-opsa20/2024-05-20_14-19-12/psro/info.npz"
data = np.load(npz_pth)  # 加载
adver_strategy = data['adver_strategy']
prog_strategy = data['prog_strategy']
payoff = data["payoffs"]

result = eval(payoff, prog_strategy, adver_strategy)
print(result)