
sweagent merge-preds trajectories/smith32b-pr-2k-removed/all-1

python -m swesmith.harness.eval \
    --dataset_path data/SWE_SMITH_INSTANCES.json \
    --predictions_path trajectories/smith32b-pr-2k-removed/all-1/preds.json \
    --run_id smith32b-pr-2k-removed-0916-all-1 \
    --workers 100