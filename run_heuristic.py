import hydra
from rl4co.envs import SVRPEnv, SPCTSPEnv
from rl4co.tasks.eval_heuristic import evaluate_baseline

env = SVRPEnv(num_loc=20) 
# 使用test的数据集做evaluation
dataset_f = "/home/panpan/rl4co/data/svrp/svrp_modelize20_test_seed1234.npz"
# dataset_td = env.dataset(phase="test", filename=dataset_f)     
        
baseline = "tabu"
save_fname = "tabu_results.npz"
evaluate_baseline(env, dataset_f, baseline, save_results=True, save_fname=save_fname)
