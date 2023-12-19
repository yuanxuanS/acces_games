import hydra
from rl4co.envs import SVRPEnv, SPCTSPEnv
from rl4co.tasks.eval_heuristic import evaluate_baseline
from rl4co.heuristic import CW_svrp, TabuSearch_svrp
import numpy as np

env = SVRPEnv(num_loc=20) 
# 使用test的数据集做evaluation
dataset_f = "/home/panpan/rl4co/data/svrp/svrp_modelize20_test_seed1234.npz"
# dataset_td = env.dataset(phase="test", filename=dataset_f)     
        
baseline = "tabu"
save_fname = "tabu_results.npz"
evaluate_baseline(env, dataset_f, baseline, save_results=True, save_fname=save_fname)

# # test
# td_init = env.reset(batch_size=[3])      # return batch_size datas by generate_data
# retvals = TabuSearch_svrp(td_init).forward()
# np.savez(save_fname, **retvals)
