import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
log_pth = "/home/panpan/rl4co/logs/train_psro/runs/svrp20/am-svrp20/2024-04-06_15-03-36/psro/"
data = np.load(log_pth+ 'info.npz')  # 加载
nashconv = data['nashconv_lst']  # 引用保存好的数组，他的格式默认是numpy.array
payoffs = data['payoffs']

# plot JPC
plt.imshow(payoffs, cmap='coolwarm', norm=PowerNorm(3))
plt.title("utility of PSRO")
plt.xlabel("adversary")
plt.ylabel("protagonist")
plt.colorbar()
# compute JPC

print(payoffs.shape)
total_sum = np.sum(payoffs)

diagonal_sum = 0
size = payoffs.shape[0]
for i in range(size):
    diagonal_sum += payoffs[i][i]

nondiag_sum = total_sum - diagonal_sum

jpc = 1 - (diagonal_sum / size) / (nondiag_sum / (size*size - size))
print(f"total_sum is {total_sum}, diag sum is {diagonal_sum}, nondiag sum is {nondiag_sum}, jpc is {jpc}")
'''
# plot nashconv
plt.plot(nashconv)
plt.title("NashConv of PSRO")
plt.xlabel("iteration")
plt.ylabel("NashConv")
'''

name = "payoffs_log"
plt.savefig("./graphs/"+name+".jpg")