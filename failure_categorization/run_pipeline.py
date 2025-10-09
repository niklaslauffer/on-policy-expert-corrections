#!/usr/bin/env python3
"""
Main orchestration script for the seaborn analysis pipeline.
Runs the complete analysis workflow including data generation and visualization.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def run_command(cmd, description, check=True):
    """Run a command with proper error handling"""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ SUCCESS: {description}")
        if result.stdout.strip():
            print("Output:", result.stdout.strip())
    else:
        print(f"❌ FAILED: {description}")
        if result.stderr.strip():
            print("Error:", result.stderr.strip())
        if check:
            sys.exit(1)
    
    return result

def find_most_recent_analysis_dir(base_name="analysis_results"):
    """Find the most recent analysis results directory"""
    output_dir = Path("outputs")
    if not output_dir.exists():
        return None
    
    # Look for directories with various naming patterns
    patterns = [
        f"*{base_name}*",
        f"filtered_*{base_name}*", 
        f"*filtered*{base_name}*"
    ]
    
    all_dirs = []
    for pattern in patterns:
        matching_dirs = list(output_dir.glob(pattern))
        all_dirs.extend(matching_dirs)
    
    if not all_dirs:
        return None
    
    # Sort by modification time and return most recent
    all_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return all_dirs[0]

def main():
    parser = argparse.ArgumentParser(description="Run the complete seaborn analysis pipeline")
    parser.add_argument("--results-dir", required=True, help="Directory containing trajectory results")
    parser.add_argument("--exit-statuses-yaml", required=True, help="Path to exit statuses YAML file")
    parser.add_argument("--evaluation-json", help="Path to evaluation JSON file for filtering")
    parser.add_argument("--filter-instances", choices=["resolved", "unresolved"], help="Filter to only resolved or unresolved instances")
    parser.add_argument("--output-subdir", help="Custom output subdirectory name")
    parser.add_argument("--num-actions", type=int, default=20, help="Number of past actions to analyze")
    parser.add_argument("--model-name", help="Human-readable model name (used in summaries/visualizations)")
    
    args = parser.parse_args()
    
    print("🚀 STARTING SEABORN ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Exit statuses YAML: {args.exit_statuses_yaml}")
    
    if args.filter_instances:
        print(f"Filtering: {args.filter_instances} instances")
        print(f"Evaluation JSON: {args.evaluation_json}")
    
    print("=" * 60)
    print("STEP 1: GENERATING ANALYSIS DATA")
    print("=" * 60)
    
    # Step 1: Run qualitative analysis
    cmd = [
        "python", "scripts/run_all_categories.py",
        "--results-dir", args.results_dir,
        "--exit-statuses-yaml", args.exit_statuses_yaml,
        "--num-actions", str(args.num_actions)
    ]
    
    if args.filter_instances and args.evaluation_json:
        cmd.extend(["--filter-instances", args.filter_instances])
        cmd.extend(["--evaluation-json", args.evaluation_json])
    
    if args.output_subdir:
        cmd.extend(["--output-subdir", args.output_subdir])
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    
    result = run_command(cmd, "Running qualitative analysis for all categories")
    
    # Extract output directory from the command output
    output_lines = result.stdout.split('\n')
    analysis_dir = None
    for line in output_lines:
        if "Output directory:" in line:
            analysis_dir = line.split("Output directory:")[-1].strip()
            break
    
    if not analysis_dir:
        # Try to find the most recent analysis directory
        analysis_dir = find_most_recent_analysis_dir()
        if analysis_dir:
            analysis_dir = str(analysis_dir)
        else:
            print("❌ Could not determine analysis output directory")
            sys.exit(1)
    
    print("=" * 60)
    print("STEP 2: CREATING SEABORN VISUALIZATIONS")
    print("=" * 60)
    
    # Step 2: Create visualizations
    # Convert to relative path if it's in outputs/
    if "outputs/" in analysis_dir:
        rel_path = analysis_dir.split("outputs/")[-1]
        viz_results_dir = f"outputs/{rel_path}"
    else:
        viz_results_dir = analysis_dir
    
    cmd = [
        "python", "scripts/create_seaborn_visualizations.py",
        "--results_dir", viz_results_dir
    ]
    if args.model_name:
        cmd.extend(["--model_name", args.model_name])
    
    run_command(cmd, "Creating seaborn visualizations")
    
    print("=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"📁 Analysis results: {analysis_dir}")
    
    # Find visualization directory
    if Path(viz_results_dir).exists():
        viz_dirs = list(Path(viz_results_dir).glob("seaborn_figures_*"))
        if viz_dirs:
            latest_viz = sorted(viz_dirs, key=lambda x: x.stat().st_mtime)[-1]
            print(f"📊 Visualizations: {latest_viz}")

if __name__ == "__main__":
    main()