"""
Convert SWE-agent trajectory rollouts to SFT (Supervised Fine-Tuning) data format.

This script processes trajectory files that contain regenerated responses and converts
them into prompt-response pairs suitable for supervised fine-tuning.

The output format is:
[
  {
    "prompt": "system prompt + full conversation history up to current turn",
    "response": "the regenerated response for this turn",
    "instance_id": "unique identifier",
    "traj_id": "trajectory hash identifier",
    "turn_id": "turn number within trajectory",
    "resolved": "whether the instance was resolved"
  },
  ...
]

Usage:
python -m swesmith.train.traj_mgr.convert_to_sft --traj_dir <path> \
    --eval_dir <path> \
    --output <path>
"""

import argparse
import json
from pathlib import Path
from traj_mgr.utils import generate_hash
from tqdm.auto import tqdm
from typing import Optional, Tuple, List
from transformers import AutoTokenizer


def build_prompt_from_query(entry: dict, tokenizer) -> str:
    """Extract the query field and format it using the tokenizer's chat template."""
    if "query" not in entry or not entry["query"]:
        return ""
    
    # The query field already contains the entire conversation history up to this point
    if isinstance(entry["query"], list):
        # New format: list of messages - use directly
        messages = entry["query"]
    else:
        # Old format: single query string - wrap in messages format
        messages = [{"role": "user", "content": entry["query"]}]
    
    # Use the tokenizer's chat template to format the messages
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return prompt


def extract_sft_pairs_from_trajectory(traj: dict, tokenizer) -> List[dict]:
    """Extract prompt-response pairs from a single trajectory."""
    if "trajectory" not in traj:
        return []
    
    trajectory = traj["trajectory"]
    instance_id = traj.get("instance_id", "unknown")
    resolved = traj.get("resolved", False)
    
    # Generate a trajectory ID based on the content
    traj_content = json.dumps(trajectory, sort_keys=True)
    hash_id = generate_hash(traj_content)
    traj_id = f"{instance_id}.{hash_id}"
    
    sft_pairs = []
    
    for i, entry in enumerate(trajectory):
        # Only process entries that have a regenerated response
        if "response" not in entry or not entry["response"]:
            continue
            
        # Extract the prompt from the query field using chat template
        prompt = build_prompt_from_query(entry, tokenizer)
        
        # Use the regenerated response
        response = entry["response"]
        
        sft_pair = {
            "prompt": prompt,
            "response": response,
            "instance_id": instance_id,
            "traj_id": traj_id,
            "turn_id": i,
            "resolved": resolved
        }
        
        # Add model information if available
        if "replay_config" in traj:
            try:
                config = json.loads(traj["replay_config"])
                sft_pair["model"] = config["agent"]["model"]["name"]
            except (json.JSONDecodeError, KeyError):
                pass
        
        sft_pairs.append(sft_pair)
    
    return sft_pairs


def process_single_trajectory_file(
    traj_file: Path,
    eval_dir: Path,
    tokenizer,
) -> Optional[Tuple[str, List[dict]]]:
    """Process a single trajectory file and return SFT pairs."""
    try:
        # Extract instance_id from filename
        instance_id = traj_file.stem.replace(".traj", "")
        
        # Check if we have evaluation results for this instance
        if eval_dir and (eval_dir / instance_id).exists():
            report_path = eval_dir / instance_id / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                is_resolved = (
                    report.get("resolved", False)
                    if instance_id not in report
                    else report[instance_id].get("resolved", False)
                )
            else:
                is_resolved = False
        else:
            is_resolved = False
        
        # Load trajectory
        traj = json.loads(traj_file.read_text())
        traj["instance_id"] = instance_id
        traj["resolved"] = is_resolved
        
        # Extract SFT pairs
        sft_pairs = extract_sft_pairs_from_trajectory(traj, tokenizer)
        
        return (instance_id, sft_pairs)
        
    except Exception as e:
        print(f"Error processing file {traj_file}: {e}")
        return None


def main(
    traj_dir: Path,
    eval_dir: Optional[Path],
    output: Path,
):
    """Main function to convert trajectory rollouts to SFT data."""
    
    # Initialize the Qwen tokenizer
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    
    # Find all trajectory files
    traj_files = list(traj_dir.glob("*.traj"))
    if not traj_files:
        # Try looking in subdirectories
        traj_files = list(traj_dir.glob("*/*.traj"))
    
    if not traj_files:
        print(f"No .traj files found in {traj_dir}")
        return
    
    print(f"Found {len(traj_files)} trajectory files in {traj_dir}")
    
    # Process trajectories sequentially
    all_sft_pairs = []
    processed_files = 0
    
    for traj_file in tqdm(traj_files, desc="Processing trajectory files"):
        result = process_single_trajectory_file(traj_file, eval_dir, tokenizer)
        if result is not None:
            instance_id, sft_pairs = result
            all_sft_pairs.extend(sft_pairs)
            processed_files += 1
    
    # Write results to output file
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        for sft_pair in all_sft_pairs:
            f.write(json.dumps(sft_pair) + "\n")
    
    print(f"Processed {processed_files} trajectory files")
    print(f"Generated {len(all_sft_pairs)} SFT prompt-response pairs")
    print(f"Wrote results to {output.absolute()}")
    
    # Print some statistics
    if all_sft_pairs:
        resolved_count = sum(1 for pair in all_sft_pairs if pair["resolved"])
        unique_instances = len(set(pair["instance_id"] for pair in all_sft_pairs))
        
        print("\nStatistics:")
        print(f"  Unique instances: {unique_instances}")
        print(f"  Resolved instances: {resolved_count} / {len(all_sft_pairs)} pairs")
        print(f"  Average turns per instance: {len(all_sft_pairs) / unique_instances:.1f}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        "-t",
        "--traj_dir",
        type=Path,
        required=True,
        help="Path to folder containing trajectory files with regenerated responses",
    )
    arg_parser.add_argument(
        "-e",
        "--eval_dir",
        type=Path,
        required=False,
        help="Path to folder containing evaluation results (optional)",
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to output JSONL file for SFT data",
    )
    args = arg_parser.parse_args()
    main(**vars(args)) 