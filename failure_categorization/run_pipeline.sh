#!/bin/bash

# Simple seaborn analysis pipeline
# Usage: ./run_pipeline.sh <results_dir> <exit_statuses_yaml> <evaluation_json>

set -e  # Exit on any error

if [ $# -ne 3 ]; then
    echo "Usage: $0 <results_dir> <exit_statuses_yaml> <evaluation_json>"
    echo "Example: $0 /path/to/trajectories /path/to/run_batch_exit_statuses.yaml /path/to/evaluation.json"
    exit 1
fi

RESULTS_DIR="$1"
EXIT_STATUSES_YAML="$2"
EVALUATION_JSON="$3"

echo "STARTING SEABORN ANALYSIS PIPELINE"
echo "============================================================"
echo "Results directory: $RESULTS_DIR"
echo "Exit statuses YAML: $EXIT_STATUSES_YAML"
echo "Evaluation JSON: $EVALUATION_JSON"
echo "============================================================"

# Step 1: Run qualitative analysis for all categories
echo ""
echo "============================================================"
echo "STEP 1: GENERATING ANALYSIS DATA"
echo "============================================================"
echo "Running qualitative analysis for all categories..."

python scripts/run_all_categories.py \
    --results-dir "$RESULTS_DIR" \
    --exit-statuses-yaml "$EXIT_STATUSES_YAML" \
    --filter-instances unresolved \
    --evaluation-json "$EVALUATION_JSON" \
    --num-actions 20

# Extract the output directory from the analysis
LATEST_OUTPUT_DIR=$(find outputs -name "filtered_unresolved_*_analysis_results_*_20actions" -type d | sort | tail -1)

if [ -z "$LATEST_OUTPUT_DIR" ]; then
    echo "ERROR: Could not find analysis output directory"
    exit 1
fi

echo "Analysis complete! Output directory: $LATEST_OUTPUT_DIR"

# Step 2: Create seaborn visualizations
echo ""
echo "============================================================"
echo "STEP 2: CREATING SEABORN VISUALIZATIONS"
echo "============================================================"
echo "Creating seaborn visualizations..."

python scripts/create_seaborn_visualizations.py --results_dir "$LATEST_OUTPUT_DIR"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo "Analysis results: $LATEST_OUTPUT_DIR"
echo "Visualizations: $LATEST_OUTPUT_DIR/seaborn_figures_*"
