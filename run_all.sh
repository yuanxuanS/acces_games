#!/bin/bash
conda activate rl4co
# 定义任务列表
tasks=("am-critic" "am-ppo" "am-xl" "am" "pomo" "symnco")

# 循环遍历任务列表
for task in "${tasks[@]}"; do
    echo "Running $task..."
    
    # 在这里添加运行任务的命令，例如：
    python run.py experiment=routing/$task
    
    # 等待任务完成
    wait
    
    echo "$task completed."
done

echo "All tasks completed."
