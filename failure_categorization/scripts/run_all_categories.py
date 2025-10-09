#!/usr/bin/env python3
"""
Run qualitative analysis for all exit status categories.

This script runs the qualitative analysis on all categories found in the
run_batch_exit_statuses.yaml file with a configurable number of past actions.
"""

import os
import subprocess
import sys
import yaml
import json
import argparse
from pathlib import Path
import time
from datetime import datetime

def load_categories(yaml_path):
    """Load all categories from the YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Handle both direct and nested structures
    if 'instances_by_exit_status' in data:
        categories_data = data['instances_by_exit_status']
    else:
        categories_data = data
    
    categories = list(categories_data.keys())
    return categories, categories_data

def load_evaluation_results(evaluation_json_path):
    """Load evaluation results to get resolved/unresolved instances.

    Supports multiple formats:
    1) {"resolved_ids": [...], "unresolved_ids": [...]} (preferred)
    2) {"instance_id": true|false, ...} where boolean indicates resolved status
    3) [{"instance_id"|"id": str, "resolved": bool}, ...]
    """
    with open(evaluation_json_path, 'r') as f:
        data = json.load(f)

    # Case 1: Explicit arrays
    if isinstance(data, dict) and ("resolved_ids" in data or "unresolved_ids" in data):
        resolved_ids = set(data.get("resolved_ids", []))
        unresolved_ids = set(data.get("unresolved_ids", []))
        return resolved_ids, unresolved_ids

    # Case 2: Mapping of instance_id -> boolean
    if isinstance(data, dict):
        resolved_ids = {instance_id for instance_id, is_resolved in data.items() if bool(is_resolved) is True}
        unresolved_ids = {instance_id for instance_id, is_resolved in data.items() if bool(is_resolved) is False}
        return resolved_ids, unresolved_ids

    # Case 3: List of records
    if isinstance(data, list):
        resolved_ids = set()
        unresolved_ids = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            instance_id = item.get("instance_id") or item.get("id")
            if instance_id is None:
                continue
            is_resolved = item.get("resolved")
            if is_resolved is True:
                resolved_ids.add(instance_id)
            elif is_resolved is False:
                unresolved_ids.add(instance_id)
        return resolved_ids, unresolved_ids

    # Fallback: empty sets
    return set(), set()

def filter_instances_by_category(categories_data, filter_ids, filter_type):
    """Filter instances in each category based on filter_ids."""
    filtered_data = {}
    filter_stats = {
        "filter_type": filter_type,
        "original_total": 0,
        "filtered_total": 0,
        "categories": {}
    }
    
    print(f"🔽 Filtering instances to keep only {filter_type} cases...")
    
    for category, instances in categories_data.items():
        if isinstance(instances, list):
            original_count = len(instances)
            filter_stats["original_total"] += original_count
            
            # For submitted categories, filter by evaluation results
            if category.startswith("submitted"):
                filtered_instances = [inst for inst in instances if inst in filter_ids]
                filtered_data[category] = filtered_instances
                filtered_count = len(filtered_instances)
                print(f"  {category}: {original_count} → {filtered_count} instances (filtered by evaluation)")
            else:
                # For non-submitted categories, keep all instances (they're all unresolved by definition)
                filtered_data[category] = instances
                filtered_count = original_count
                print(f"  {category}: {original_count} → {filtered_count} instances (kept all - non-submitted)")
            
            filter_stats["filtered_total"] += filtered_count
            filter_stats["categories"][category] = {
                "original": original_count,
                "filtered": filtered_count
            }
        else:
            # Handle integer case
            filtered_data[category] = instances
            filter_stats["original_total"] += instances
            filter_stats["filtered_total"] += instances
            filter_stats["categories"][category] = {
                "original": instances,
                "filtered": instances
            }
    
    return filtered_data, filter_stats

def update_num_actions(script_path, num_actions):
    """Update the NUM_PAST_ACTIONS in the analysis script."""
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace the NUM_PAST_ACTIONS line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('NUM_PAST_ACTIONS ='):
            lines[i] = f"NUM_PAST_ACTIONS = {num_actions}  # Number of past actions/observations to analyze"
            break
    
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Updated {script_path} to use {num_actions} past actions")

def run_analysis_for_category(category, results_dir, exit_statuses_yaml, output_dir):
    """Run analysis for a single category."""
    print(f"\n{'='*60}")
    print(f"Running analysis for category: {category}")
    print(f"{'='*60}")
    
    # Build command - use original argument names
    analysis_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qualitative_analysis_by_category.py")
    
    cmd = [
        "python", analysis_script_path,
        "--category", category,
        "--results_dir", results_dir,
        "--exit_statuses_yaml", os.path.abspath(exit_statuses_yaml),
        "--num-actions", str(NUM_PAST_ACTIONS)
    ]
    
    # Change to output directory before running
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    # Restore original working directory
    os.chdir(original_cwd)
    
    if result.returncode == 0:
        print(f"✅ SUCCESS: Analysis completed in {elapsed_time:.1f}s")
        if result.stdout.strip():
            print("Output:", result.stdout.strip()[:500] + "..." if len(result.stdout.strip()) > 500 else result.stdout.strip())
    else:
        print(f"❌ FAILED: Analysis failed after {elapsed_time:.1f}s")
        print("Error:", result.stderr.strip()[:500] + "..." if len(result.stderr.strip()) > 500 else result.stderr.strip())
        return False
    
    return True

def create_pipeline_summary(output_dir, filter_stats, args):
    """Create a summary of the pipeline execution."""
    summary = {
        "pipeline_info": {
            "execution_timestamp": datetime.now().isoformat(),
            "results_dir": args.results_dir,
            "exit_statuses_yaml": args.exit_statuses_yaml,
            "num_actions": args.num_actions,
            "model_name": args.model_name,
            "output_directory": str(output_dir)
        }
    }
    
    if filter_stats:
        summary["filtering"] = filter_stats
    
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Pipeline summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Run qualitative analysis for all categories")
    parser.add_argument("--results-dir", required=True, help="Directory containing trajectory results")
    parser.add_argument("--exit-statuses-yaml", required=True, help="Path to exit statuses YAML file")
    parser.add_argument("--num-actions", type=int, default=20, help="Number of past actions to analyze")
    parser.add_argument("--output-subdir", help="Custom output subdirectory name")
    parser.add_argument("--filter-instances", choices=["resolved", "unresolved"], help="Filter to only resolved or unresolved instances")
    parser.add_argument("--evaluation-json", help="Path to evaluation JSON file (required for filtering)")
    parser.add_argument("--model-name", help="Human-readable model name (annotated in outputs)")
    
    args = parser.parse_args()
    
    # Validation
    if args.filter_instances and not args.evaluation_json:
        print("❌ Error: --evaluation-json is required when using --filter-instances")
        sys.exit(1)
    
    # Make number of past actions available to helper functions
    global NUM_PAST_ACTIONS
    NUM_PAST_ACTIONS = args.num_actions

    # Set up paths
    results_dir = args.results_dir
    exit_statuses_yaml = args.exit_statuses_yaml
    analysis_script = "scripts/qualitative_analysis_by_category.py"
    
    # Output directory
    if args.output_subdir:
        output_subdir = args.output_subdir
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.filter_instances:
            output_subdir = f"filtered_{args.filter_instances}_{Path(results_dir).name}_analysis_results_{timestamp_str}_{args.num_actions}actions"
        else:
            output_subdir = f"analysis_results_{timestamp_str}_{args.num_actions}actions"
    
    output_dir = Path("outputs") / output_subdir
    
    print(f"Running qualitative analysis for all categories with {args.num_actions} past actions")
    print(f"Sample directory being analyzed: {results_dir}")
    print(f"Exit statuses file: {exit_statuses_yaml}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified output directory: {output_dir}")
    
    # Update the analysis script with the number of actions  
    update_num_actions(analysis_script, args.num_actions)
    
    # Load categories and apply filtering if requested
    categories, categories_data = load_categories(exit_statuses_yaml)
    filter_stats = None
    
    if args.filter_instances:
        print(f"\n🔍 Applying {args.filter_instances} filtering...")
        
        # Load evaluation results
        eval_results = load_evaluation_results(args.evaluation_json)
        resolved_ids, unresolved_ids = eval_results
        print(f"Loaded evaluation results: {len(resolved_ids)} resolved, {len(unresolved_ids)} unresolved")
        
        # Apply filtering
        if args.filter_instances == "resolved":
            filter_ids = resolved_ids
        else:  # unresolved
            filter_ids = unresolved_ids
        
        categories_data, filter_stats = filter_instances_by_category(categories_data, filter_ids, args.filter_instances)

        # Ensure unresolved instances not present in YAML categories are still analyzed
        # by adding an 'uncategorized' bucket containing the remainder.
        try:
            all_ids_in_yaml = set()
            for cat, inst_list in categories_data.items():
                if isinstance(inst_list, list):
                    all_ids_in_yaml.update(inst_list)
            remainder = set(filter_ids) - all_ids_in_yaml
            if remainder:
                categories_data['uncategorized'] = sorted(remainder)
                # Update filter stats accounting
                if filter_stats is not None:
                    filter_stats['categories']['uncategorized'] = {
                        'original': 0,
                        'filtered': len(remainder)
                    }
                    filter_stats['filtered_total'] += len(remainder)
                print(f"  uncategorized: added {len(remainder)} unresolved instances not present in exit-status YAML")
        except Exception as e:
            print(f"Warning: failed to add uncategorized remainder: {e}")

        categories = list(categories_data.keys())
        
        # Create filtered YAML for analysis scripts
        filtered_yaml_data = {'instances_by_exit_status': categories_data}
        filtered_yaml_path = output_dir / "filtered_exit_statuses.yaml"
        with open(filtered_yaml_path, 'w') as f:
            yaml.dump(filtered_yaml_data, f)
        
        # Use filtered YAML for analysis
        exit_statuses_yaml = str(filtered_yaml_path)
        print(f"Created filtered YAML: {filtered_yaml_path}")
        
        # Print filtering summary
        print(f"📊 Filtering Summary:")
        print(f"  Filter type: {filter_stats['filter_type']}")
        print(f"  Original total: {filter_stats['original_total']} instances")
        print(f"  Filtered total: {filter_stats['filtered_total']} instances")
        # Guard against division by zero when original_total is 0 (e.g., empty YAML)
        _orig_total = filter_stats.get('original_total', 0)
        _filt_total = filter_stats.get('filtered_total', 0)
        _reduction = _orig_total - _filt_total
        if _orig_total > 0:
            _pct = (_reduction / _orig_total) * 100
            _pct_str = f"{_pct:.1f}%"
        else:
            _pct_str = "N/A"
        print(f"  Reduction: {_reduction} instances ({_pct_str})")
        print(f"  Remaining categories: {len(categories)}")
    
    print(f"\nFound {len(categories)} categories:")
    for i, category in enumerate(categories, 1):
        instance_count = len(categories_data[category]) if isinstance(categories_data[category], list) else categories_data[category]
        print(f"  {i}. {category} ({instance_count} instances)")
    
    print("\nStarting analysis...")
    
    # Run analysis for each category
    successful_categories = []
    failed_categories = []
    
    for category in categories:
        success = run_analysis_for_category(category, results_dir, exit_statuses_yaml, output_dir)
        if success:
            successful_categories.append(category)
        else:
            failed_categories.append(category)
    
    # Create pipeline summary
    create_pipeline_summary(output_dir, filter_stats, args)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total categories: {len(categories)}")
    print(f"Successful: {len(successful_categories)}")
    print(f"Failed: {len(failed_categories)}")
    
    if failed_categories:
        print(f"Failed categories: {', '.join(failed_categories)}")
    
    # Check for generated files
    print(f"Generated analysis files (in {output_subdir}/):")
    for category in categories:
        category_file = category.replace(" ", "_").replace("(", "").replace(")", "")
        expected_file = output_dir / f"{category_file}_analysis_{args.num_actions}actions.json"
        if expected_file.exists():
            file_size = expected_file.stat().st_size / 1024  # KB
            print(f"  ✅ {expected_file.name} ({file_size:.1f} KB)")
        else:
            print(f"  ❌ {expected_file.name} (not found)")
    
    print(f"To view visualizations, run:")
    print(f"python scripts/create_seaborn_visualizations.py --results_dir {output_subdir}")

if __name__ == "__main__":
    main()