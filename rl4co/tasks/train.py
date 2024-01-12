from typing import List, Optional, Tuple
from rl4co.tasks.eval import evaluate_policy

import hydra
import lightning as L
import pyrootutils
import torch

from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from rl4co import utils
from rl4co.utils import RL4COTrainer
from memory_profiler import profile
from guppy import hpy
pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)


@utils.task_wrapper
# @profile(stream=open('log_mem_cvrp50.log', 'w+'))
def run(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # h = hpy().heap()
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # We instantiate the environment separately and then pass it to the model
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    if cfg.fix_graph:
        graph_pool = hydra.utils.instantiate(cfg.graph_pool)
        graph_pool.generate_datas()
        env.get_fix_data(graph_pool)
        
    # Note that the RL environment is instantiated inside the model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, env)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer...")
    trainer: RL4COTrainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile", False):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    
    # 额外增加evaluate 过程
    if cfg.get("evaluate"):
        method = cfg.evaluate_method
        log.info(f"Start evaluation by {method}!")
        
        if cfg.get("mode") == "evaluate":
            ckpt_path = cfg.get("ckpt_path")
            evaluate_model = model.load_from_checkpoint(ckpt_path)
            save_fname = cfg.get("evaluate_loc") + "/evalu_"+method+".npz"
        elif cfg.get("mode") == "train":
            ckpt_path = trainer.checkpoint_callback.best_model_path
            evaluate_model = trainer.model.load_from_checkpoint(ckpt_path)
            save_fname = logger[0].save_dir + "/evalu_"+method+".npz"
        log.info(f"Best ckpt path: {ckpt_path}")
        
        dataset = env.dataset(phase="test")     # 使用test的数据集做evaluation
        
        evaluate_policy(env, evaluate_model.policy, dataset, method, save_results=True, save_fname=save_fname)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main.yaml")
# @hydra.main(version_base="1.3", config_path="../../configs", config_name="experiment/routing/am-ppo.yaml")
# @hydra.main(version_base="1.3", config_path="configs", config_name="experiment/routing/am-ppo.yaml")
def train(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    train()
