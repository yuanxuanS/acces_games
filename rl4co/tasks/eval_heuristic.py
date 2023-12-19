import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rl4co.heuristic import CW_svrp, TabuSearch_svrp

def check_unused_kwargs(class_, kwargs):
    if len(kwargs) > 0 and not (len(kwargs) == 1 and "progress" in kwargs):
        print(f"Warning: {class_.__class__.__name__} does not use kwargs {kwargs}")



def evaluate_baseline(
    env,
    dataset_filename,
    baseline="greedy",
    save_results=True,
    save_fname="results.npz",
    **kwargs,
):
    num_loc = getattr(env, "num_loc", None)

    baselines_mapping = {
        "cw": {"func": CW_svrp, "kwargs": {}},
        "tabu": {
            "func": TabuSearch_svrp, "kwargs": {}},
        
    }

    assert baseline in baselines_mapping, "baseline {} not found".format(baseline)


    # env td  data
    f = getattr(env, f"test_file") if dataset_filename is None else dataset_filename
    td_load = env.load_data(f)       # this func normalize to [0-1]
    # reset td, use this data-key-value to process in heuristic func
    td_load = env._reset(td_load) 
       
    # Set up the evaluation function
    eval_settings = baselines_mapping[baseline]
    func, kwargs_ = eval_settings["func"], eval_settings["kwargs"]
    # subsitute kwargs with the ones passed in
    kwargs_.update(kwargs)
    kwargs = kwargs_
    eval_fn = func(td_load)


    # Run evaluation
    retvals = eval_fn.forward()

    # Save results
    if save_results:
        print("Saving results to {}".format(save_fname))
        np.savez(save_fname, **retvals)

    return retvals
