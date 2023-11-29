```mermaid
sequenceDiagram
Q->>RL4COLitModule:setup(),config_optim(),val_dataloader()
RL4COLitModule->>SpeedMonitor:on_validation_epoch_start()
SpeedMonitor->>RL4COLitModule:validation_step()
```