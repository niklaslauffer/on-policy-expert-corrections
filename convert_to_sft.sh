python -m traj_mgr.collect_trajs \
    --traj_dir sweagent_results/smith32b-pr-2k-removed/all-1 \
    --eval_dir logs/run_evaluation/smith32b-pr-2k-removed-0916-${chunk} \
    --out_dir trajectories_sft/qwen32b-2000imitation-pr-2k-removed \
    --resolved_only


python -m swesmith.train.traj_mgr.combine_trajs --max_per_inst 10000 --output_file trajectories_sft/qwen32b-2000imitation-pr-2k-removed-all-1-0917.jsonl --sft_dir trajectories_sft/qwen32b-2000imitation-pr-2k-removed
