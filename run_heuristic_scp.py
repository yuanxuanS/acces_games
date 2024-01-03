import hydra
from rl4co.envs import SVRPEnv, SPCTSPEnv, SCPEnv
from rl4co.tasks.eval_heuristic import evaluate_baseline
from rl4co.heuristic import CW_svrp, TabuSearch_svrp, Greedy_scp
import numpy as np

env = SCPEnv(num_loc=20, min_cover=0.1, max_cover=0.3) 
# 使用test的数据集做evaluation
dataset_f = "/home/panpan/rl4co/data/svrp/svrp_modelize50_test_seed1234.npz"
# dataset_td = env.dataset(phase="test", filename=dataset_f)     
        
baseline = "tabu"
save_fname = "tabu_results_svrp50.npz"
# evaluate_baseline(env, dataset_f, baseline, save_results=True, save_fname=save_fname)

# # test
td_init = env.reset(batch_size=[3])      # return batch_size datas by generate_data
Greedy_scp(td_init, min_cover=0.1, max_cover=0.3).forward()
# np.savez(save_fname, **retvals)
