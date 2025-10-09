  sweagent run-batch \
    --config config/qwen32b_switch_claude_python_tools.yaml \
    --output_dir trajectories/qwen32b_switch_claude \
    --num_workers 10 \
    --random_delay_multiplier 1 \
    --instances.type swesmith \
    --instances.path DOWNLOAD_INSTANCES_FROM_SWE_SMITH_GITHUB.json \
    --instances.shuffle=False \
    --instances.deployment.type=modal \
    --instances.deployment.startup_timeout=600.0 