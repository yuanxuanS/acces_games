```mermaid
sequenceDiagram
run()->>CVRPEnv:_make_spec()
CVRPEnv->>run():_
run()->>AttentionModel:_
AttentionModel->>AttentionModelPolicy:_init
AttentionModelPolicy->>AutoregressivePolicy:super, encoder & decoder
AutoregressivePolicy->>run():_
run()->>run():instantiate_loggers,instantiate_callbacks
run()->>RL4COTrainer:_init_
RL4COTrainer->>run():_
run()->>run():log_hyperparams
run()->>RL4COTrainer:fit()



```