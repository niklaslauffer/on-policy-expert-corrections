import os
import openai
import json
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import asyncio
import yaml

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=""  # LiteLLM Proxy
)

# Async client for parallel processing
async_client = openai.AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=""  # LiteLLM Proxy
)

# Initialize with predefined categories
KNOWN_CATEGORIES = {
    "identified_incorrect_file": "The agent incorrectly identified the file that needed to be fixed.",
    "missed_edge_case": "The agent missed an edge case in one of the test cases.",
    "misunderstood_problem_statement": "The agent misunderstood the problem statement.",
    "wrong_solution": "The agent generated a wrong solution.",
    "tool_error": "The agent encountered an error while using a tool (e.g. by calling it incorrectly).",
    "infinite_loop": "The agent entered an infinite loop (e.g. repeating the same sequence of steps).",
    "endless_file_reading": "The agent read the same file multiple times without making any changes.",
    "context_overflow_from_listing": "The agent's file listing operations (ls, find, etc.) caused context overflow.",
    "syntax_error": "The agent generated syntactically incorrect code.",
    "other": "The agent failed to resolve the issue for other reasons.",
}

# Exit status descriptions to help LLM understand the context
EXIT_STATUS_DESCRIPTIONS = {
    "exit_context": "Agent exceeded context window limit.",
    "exit_cost": "Agent exceeded token/cost limits.",  
    "exit_error": "Agent encountered a runtime error that caused termination",
    "submitted": "Agent completed and submitted a solution (may be correct or incorrect)",
    "submitted (exit_context)": "Agent submitted after hitting context limits",
    "submitted (exit_cost)": "Agent submitted after hitting cost limits"
}

# Action descriptions from SWE-agent to help LLM understand what each action does
ACTION_DESCRIPTIONS = """
AVAILABLE ACTIONS:

---- BEGIN FUNCTION #1: bash ----
Description: Execute a bash command in the terminal.
* Can generate very large outputs when listing files (ls, find, grep)
* Output contributes directly to context window usage
* Commands like 'find /repo -name "*.py"' can list thousands of files
* Large outputs can quickly fill the context window

Parameters:
  (1) command (string, required): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: submit ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
* Used when agent thinks task is done (may be correct or incorrect solution)
* Also used when agent is stuck and cannot make progress
* No parameters are required for this function.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* Directory views can generate large outputs contributing to context usage
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
  (2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #3 ----

---- BEGIN FUNCTION #4: file_viewer ----
Description: Interactive file viewer for opening and navigating files in the editor.
* open <path> [<line_number>]: Opens the file at path. If line_number is provided, the view moves to include that line.
* goto <line_number>: Moves the window to show the specified line number.
* scroll_down: Moves the window down 100 lines.
* scroll_up: Moves the window up 100 lines.

Parameters:
  (1) command (string, required): One of `open`, `goto`, `scroll_down`, `scroll_up`.
  (2) path_or_line (string/int, optional): For `open`, a path (and optional line). For `goto`, a line number.
---- END FUNCTION #4 ----

---- BEGIN FUNCTION #5: search_tools ----
Description: Searching utilities for locating text or files within the workspace.
* search_file <search_term> [<file>]: Searches for search_term in file. If file is not provided, searches the current open file.
* search_dir <search_term> [<dir>]: Searches for search_term in all files in dir. If dir is not provided, searches in the current directory.
* find_file <file_name> [<dir>]: Finds all files with the given name in dir. If dir is not provided, searches in the current directory.

Parameters:
  (1) subcommand (string, required): One of `search_file`, `search_dir`, `find_file`.
  (2) arg1 (string, required): The search term or file name, depending on subcommand.
  (3) arg2 (string, optional): Target file (for search_file) or directory (for search_dir/find_file).
---- END FUNCTION #5 ----

---- BEGIN FUNCTION #6: edit_block ----
Description: Block editor for replacing ranges in the current open file and finalizing edits.
* edit <n>:<m> <replacement_text>: Replaces lines n through m (inclusive) with the given text in the open file. Ensure indentation is correct.
* end_of_edit: Applies the pending changes. Python files are syntax-checked after the edit; if an error is found, the edit is rejected.

Parameters:
  (1) command (string, required): `edit` or `end_of_edit`.
  (2) range_and_text (varies): For `edit`, a line range `n:m` and the replacement text.
---- END FUNCTION #6 ----

---- BEGIN FUNCTION #7: create_file ----
Description: Creates and opens a new file with the given name.

Parameters:
  (1) filename (string, required): Absolute or workspace-relative path to create. The file must not already exist.
---- END FUNCTION #7 ----
"""

# Model pricing (per 1M tokens). These are rough estimates and may differ per provider.
GPT41_INPUT_PRICE_PER_1M = 2.00
GPT41_OUTPUT_PRICE_PER_1M = 8.00

# Analysis configuration
NUM_PAST_ACTIONS = 20  # Number of past actions/observations to analyze


async def analyze_failed_trajectory_async(instance_id, trajectory_data, known_categories, exit_status=None):
    """
    Async version of analyze_failed_trajectory for parallel processing
    """
    
    # Extract key information from trajectory
    trajectory_steps = trajectory_data.get('trajectory', [])
    
    # Get the last few steps and overall trajectory summary - UPDATED: Using configurable number of actions
    last_steps = trajectory_steps[-NUM_PAST_ACTIONS:] if len(trajectory_steps) > NUM_PAST_ACTIONS else trajectory_steps
    
    # Format the categories as a comma-separated list
    categories_str = ", ".join([f"{k}: {v}" for k, v in known_categories.items()])
    
    # Extract problem statement if available
    problem_statement = "Not available"
    if trajectory_steps:
        for step in trajectory_steps:
            if isinstance(step, dict) and 'query' in step:
                # Try to extract problem statement from first query
                for query_item in step.get('query', []):
                    if isinstance(query_item, dict) and query_item.get('role') == 'user':
                        content = query_item.get('content', '')
                        if 'PR description' in content or 'problem_statement' in content:
                            problem_statement = content[:1000] + "..." if len(content) > 1000 else content
                            break
                if problem_statement != "Not available":
                    break
    
    # Extract final actions/observations 
    final_actions = []
    final_observations = []
    
    for step in last_steps:
        if isinstance(step, dict):
            if 'action' in step:
                final_actions.append(step['action'])
            if 'observation' in step:
                final_observations.append(step['observation'][:200])
    
    # Get exit status description if available
    exit_status_desc = ""
    if exit_status and exit_status in EXIT_STATUS_DESCRIPTIONS:
        exit_status_desc = f"\nEXIT STATUS: {exit_status}\nEXIT STATUS MEANING: {EXIT_STATUS_DESCRIPTIONS[exit_status]}\n"
    
    # Construct the prompt with all relevant information
    prompt = f"""
You are an expert software engineer analyzing why a software engineering agent failed to resolve an issue.

INSTANCE ID: {instance_id}
{exit_status_desc}
{ACTION_DESCRIPTIONS}

PROBLEM STATEMENT:
{problem_statement}

FINAL ACTIONS TAKEN (Last {NUM_PAST_ACTIONS}):
{chr(10).join(final_actions[-NUM_PAST_ACTIONS:]) if final_actions else "No actions recorded"}

FINAL OBSERVATIONS (Last {NUM_PAST_ACTIONS}):
{chr(10).join(final_observations[-NUM_PAST_ACTIONS:]) if final_observations else "No observations recorded"}

TRAJECTORY SUMMARY:
- Total steps: {len(trajectory_steps)}
- Final state: Failed (no successful patch generated)

ANALYSIS INSTRUCTIONS:
The exit status indicates WHY the agent terminated. Consider how the final actions contributed to this specific exit condition.
Pay special attention to:
- If exit_context: Look for excessive output from bash commands (ls, find, etc.) that filled the context window
- If exit_cost: Look for repeated or expensive operations that exceeded token limits  
- If exit_error: Look for runtime errors or tool misuse
- If submitted: Determine if the solution was wrong, incomplete, or correct

Based on the information above, provide an error analysis in two parts:
First, an explanation of the issue and why the trajectory failed.
Second, a category for the error.

Wrap your explanation in <description></description> tags.

        For the category, choose EXACTLY one from the following set: {categories_str}
        Do NOT invent or propose new categories. If none fits, use "other".

        Place the category at the end, separated by two newlines. Category must be 
        all lowercase and only list the category name.

        Remember to write two new lines before the category.

"""

    # Call the API for analysis
    try:
        response = await async_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system",
                 "content": "You are a software engineering assistant "
                            "specializing in error analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=1
        )

        analysis_text = response.choices[0].message.content
        
        # Extract token usage for cost tracking
        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "input_cost": (usage.prompt_tokens / 1_000_000) * GPT41_INPUT_PRICE_PER_1M,
            "output_cost": (usage.completion_tokens / 1_000_000) * GPT41_OUTPUT_PRICE_PER_1M
        }
        token_info["total_cost"] = token_info["input_cost"] + token_info["output_cost"]

        # Parse the description
        description = None
        if ("<description>" in analysis_text and 
                "</description>" in analysis_text):
            start = analysis_text.find("<description>") + len("<description>")
            end = analysis_text.find("</description>")
            description = analysis_text[start:end].strip()

        # Get the category (at the end after two newlines)
        parts = analysis_text.split("\n")
        category = parts[-1].strip().lower() if parts else None
        if not category or category not in known_categories:
            category = "other"

        return instance_id, description, category, token_info

    except Exception as e:
        print(f"Error during analysis for {instance_id}: {e}")
        return instance_id, f"Analysis failed: {str(e)}", "analysis_error", None


def load_exit_statuses(yaml_file):
    """Load exit status data from YAML file"""
    with open(yaml_file, 'r') as f:
        exit_data = yaml.safe_load(f)
    return exit_data


def load_trajectory_data(trajectory_file):
    """Load trajectory data from .traj file"""
    try:
        with open(trajectory_file, 'r') as f:
            trajectory_data = json.load(f)
        return trajectory_data
    except Exception as e:
        print(f"Error loading trajectory file {trajectory_file}: {e}")
        return None


def find_trajectory_files(results_dir, instance_ids):
    """Find trajectory files for given instance IDs"""
    trajectory_files = {}
    results_path = Path(results_dir)
    
    print(f"Scanning directory: {results_path}")
    
    # Convert instance_ids to a set for faster lookup
    instance_id_set = set(instance_ids)
    
    # Single scan of all .traj files - much faster than individual rglob calls
    print("Scanning for .traj files...")
    all_traj_files = list(results_path.rglob("*.traj"))
    print(f"Found {len(all_traj_files)} .traj files total")
    
    # Build lookup table from the found files
    for traj_file in all_traj_files:
        # Extract instance_id from the file path
        # Expecting pattern: .../instance_id/instance_id.traj
        if traj_file.stem in instance_id_set:
            trajectory_files[traj_file.stem] = traj_file
    
    # Report missing files
    found_ids = set(trajectory_files.keys())
    missing_ids = instance_id_set - found_ids
    
    if missing_ids:
        print(f"Warning: No trajectory files found for {len(missing_ids)} instances")
        if len(missing_ids) <= 10:  # Only show first 10 to avoid spam
            for missing_id in list(missing_ids)[:10]:
                print(f"  Missing: {missing_id}")
        else:
            print(f"  (showing first 10) Missing: {list(missing_ids)[:10]}")
    
    return trajectory_files


async def process_batch_async(batch_data, categories, semaphore, pbar, cost_tracker, exit_status=None):
    """Process a batch of trajectories in parallel with rate limiting"""
    async with semaphore:  # Rate limiting
        tasks = []
        for instance_id, trajectory_data in batch_data:
            task = analyze_failed_trajectory_async(instance_id, trajectory_data, categories, exit_status)
            tasks.append(task)
        
        # Process all tasks in this batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update progress bar and cost tracking
        for result in results:
            if isinstance(result, Exception):
                cost_tracker['failed_requests'] += 1
                print(f"Error in batch processing: {result}")
            else:
                instance_id, description, category, token_info = result
                cost_tracker['total_requests'] += 1
                if token_info:
                    cost_tracker['total_cost'] += token_info["total_cost"]
                    cost_tracker['total_tokens'] += token_info["total_tokens"]
                else:
                    cost_tracker['failed_requests'] += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Cost': f'${cost_tracker["total_cost"]:.3f}',
                    'Tokens': f'{cost_tracker["total_tokens"]:,}',
                    'Fails': cost_tracker['failed_requests']
                })
        
        return results


async def analyze_trajectories_parallel(trajectory_files, categories, target_category, max_concurrent=10, batch_size=20):
    """Analyze trajectories in parallel with batching and rate limiting"""
    
    # Load all trajectory data first
    print("Loading trajectory data...")
    trajectory_data_list = []
    
    for instance_id, trajectory_file in tqdm(trajectory_files.items(), desc="Loading trajectories"):
        trajectory_data = load_trajectory_data(trajectory_file)
        if trajectory_data is not None:
            trajectory_data_list.append((instance_id, trajectory_data))
    
    print(f"Loaded {len(trajectory_data_list)} trajectories")
    
    # Cost tracking
    cost_tracker = {
        'total_cost': 0.0,
        'total_tokens': 0,
        'total_requests': 0,
        'failed_requests': 0
    }
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create progress bar
    pbar = tqdm(total=len(trajectory_data_list), desc="Analyzing trajectories")
    
    # Process in batches
    all_results = []
    for i in range(0, len(trajectory_data_list), batch_size):
        batch = trajectory_data_list[i:i + batch_size]
        batch_results = await process_batch_async(batch, categories, semaphore, pbar, cost_tracker, target_category)
        all_results.extend(batch_results)
    
    pbar.close()
    
    # Process results
    results = []
    for result in all_results:
        if isinstance(result, Exception):
            continue
        
        instance_id, description, category, token_info = result
        repo = instance_id.split('__')[0] if '__' in instance_id else 'unknown'
        
        if category and category not in categories:
            categories[category] = f"Auto-discovered category: {category}"
        
        results.append({
            'instance_id': instance_id,
            'repo': repo,
            'category': category,
            'description': description,
            'trajectory_file': str(trajectory_files.get(instance_id, 'unknown')),
            'token_usage': token_info
        })
    
    return results, cost_tracker


def analyze_category_trajectories(results_dir, exit_statuses_yaml, target_category, output_file):
    """Main function to analyze trajectories for a specific category"""
    print(f"Loading exit status data from {exit_statuses_yaml}...")
    exit_data = load_exit_statuses(exit_statuses_yaml)
    
    # Get instances for the target category
    instances_by_exit_status = exit_data.get('instances_by_exit_status', {})
    
    if target_category not in instances_by_exit_status:
        print(f"Error: Category '{target_category}' not found in exit statuses")
        print(f"Available categories: {list(instances_by_exit_status.keys())}")
        return
    
    category_instances = instances_by_exit_status[target_category]
    print(f"Found {len(category_instances)} instances for category '{target_category}'")
    
    print("Finding trajectory files...")
    trajectory_files = find_trajectory_files(results_dir, category_instances)
    print(f"Found trajectory files for {len(trajectory_files)} instances")
    
    if not trajectory_files:
        print("No trajectory files found. Exiting.")
        return
    
    categories = KNOWN_CATEGORIES.copy()
    
    # Run parallel analysis
    print(f"Starting analysis for category: {target_category}")
    results, cost_tracker = asyncio.run(
        analyze_trajectories_parallel(
            trajectory_files, 
            categories, 
            target_category,
            max_concurrent=50,  # Adjust based on rate limits
            batch_size=50  # Process 50 at a time
        )
    )
    
    # Extract cost tracking values
    total_cost = cost_tracker['total_cost']
    total_tokens = cost_tracker['total_tokens']
    total_requests = cost_tracker['total_requests']
    failed_requests = cost_tracker['failed_requests']
    
    # Create summary statistics
    df = pd.DataFrame(results)
    category_counts = {}
    repo_counts = {}
    
    if not df.empty:
        category_counts = df['category'].value_counts().to_dict()
        repo_counts = df['repo'].value_counts().to_dict()
    
    # Create structured output
    output_data = {
        "summary": {
            "target_category": target_category,
            "total_instances_analyzed": len(results),
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "category_distribution": category_counts,
            "repository_distribution": repo_counts,
            "known_categories": KNOWN_CATEGORIES,
            "cost_analysis": {
                "total_requests": total_requests,
                "failed_requests": failed_requests,
                "success_rate": round((total_requests - failed_requests) / max(total_requests, 1), 3),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 4),
                "avg_cost_per_instance_usd": round(total_cost / max(len(results), 1), 4),
                "pricing_model": "GPT-5",
                "input_price_per_1m_tokens": GPT41_INPUT_PRICE_PER_1M,
                "output_price_per_1m_tokens": GPT41_OUTPUT_PRICE_PER_1M
            }
        },
        "failed_instances": results
    }
    
    # Save results as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Target category: {target_category}")
    print(f"Total instances analyzed: {len(results)}")
    print("\nCost Summary:")
    print(f"  Total API requests: {total_requests}")
    print(f"  Failed requests: {failed_requests}")
    print(f"  Success rate: {((total_requests - failed_requests) / max(total_requests, 1)) * 100:.1f}%")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Average cost per instance: ${total_cost / max(len(results), 1):.4f}")
    print("\nFailure category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze failed trajectories for a specific exit status category")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing trajectory results")
    parser.add_argument("--exit_statuses_yaml", type=str, required=True,
                        help="Path to run_batch_exit_statuses.yaml file")
    parser.add_argument("--category", type=str, required=True,
                        help="Specific exit status category to analyze (e.g., 'exit_context')")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Output file for analysis results (default: {category}_analysis_{NUM_PAST_ACTIONS}actions.json)")
    parser.add_argument("--num-actions", type=int, default=20,
                        help="Number of past actions/observations to analyze")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.exit_statuses_yaml):
        print(f"Error: Exit statuses YAML file not found: {args.exit_statuses_yaml}")
        return
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Apply configurable number of past actions
    global NUM_PAST_ACTIONS
    NUM_PAST_ACTIONS = args.num_actions
    
    # Set default output filename if not provided - include number of actions to avoid overwriting
    output_file = args.output or f"{args.category.replace(' ', '_')}_analysis_{NUM_PAST_ACTIONS}actions.json"
    
    analyze_category_trajectories(
        args.results_dir, args.exit_statuses_yaml, args.category, output_file)


if __name__ == "__main__":
    main()