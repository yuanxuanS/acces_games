import nashpy as nash
import numpy as np
A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])      # 定义utility矩阵：i,j = player采取i，对手j
B = -A
rps = nash.Game(A, B)   # 创建零和游戏
sigma_r = [0, 0, 1]     # row player的strategy， 代表总出3policy
sigma_c = [0, 1, 0]
# print(rps[sigma_r, sigma_c])       # stategy输入游戏，返回每个player的收益

# 假设c player采用前两个策略随机：
sigma_c = [1 / 2, 1 / 2, 0]     # 每个策略的概率值填入
# print(rps[sigma_r, sigma_c]) 

# r player也改变策略：
sigma_r = [0, 1 / 2, 1 / 2]
# print(rps[sigma_r, sigma_c]) 

# 求解nash均衡
eqs = rps.support_enumeration()
# print(list(eqs)) # 得到两个player的均衡策略

# nash均衡只出现在非合作博弈中
# 定义一个游戏，两个player不断play
iterations = 100
np.random.seed(0)
play_counts = rps.fictitious_play(iterations=iterations)        # 知道各自unitlity的多次博弈，返回两个player采取策略的次数
# 查看policy能看到，两个player的policy次数趋于相近，这是该游戏下接近nash均衡
# for row_play_count, col_play_count in play_counts:
#     print(row_play_count, col_play_count)
    
# 求解nash均衡的解，可能不只有一个，所以返回迭代器。指定的参数initial_dropped_label为解的标号
eq0 = rps.lemke_howson(initial_dropped_label=0)
print(eq0)      # 返回两player的strategy

# equilibria = matching_pennies.lemke_howson_enumeration()
# for eq in equilibria:
    # print(eq)