# Seaborn Analysis Pipeline

A comprehensive pipeline for analyzing SWE agent trajectories using qualitative analysis and generating research-quality visualizations.

## Features

- **Qualitative Analysis**: Uses GPT-4.1 to analyze agent failures with detailed categorization
- **Flexible Filtering**: Filter analysis by resolved/unresolved cases based on evaluation results
- **Professional Visualizations**: Generate research-paper-ready charts with consistent color schemes
- **Specialized Submitted Analysis**: Separate pipeline for submitted cases using solution-oriented failure categories
- **Timestamped Outputs**: Prevents overwriting previous results
- **Comprehensive Documentation**: Detailed summaries and metadata for all analyses

## Directory Structure

```
seaborn_pipeline/
├── README.md                              # This file
├── run_pipeline.py                        # Main orchestration script
├── scripts/                               # Analysis and visualization scripts
│   ├── run_all_categories.py             # General analysis runner with filtering
│   ├── qualitative_analysis_by_category.py # Core GPT-4 analysis script
│   ├── create_seaborn_visualizations.py   # General visualization generator
│   ├── run_submitted_analysis.py          # Submitted cases analysis runner
│   ├── submitted_analysis_by_category.py  # Submitted cases analysis script
│   └── create_submitted_visualizations.py # Submitted cases visualization generator
└── outputs/                               # Generated analysis results and visualizations
```

## Quick Start

### 1. General Analysis (All Categories)

Run the complete pipeline on trajectory data:

```bash
cd seaborn_pipeline
conda activate swe-env
export OPENAI_API_KEY=your_key_here

# Run complete pipeline with filtering for unresolved cases
python run_pipeline.py \
  --results-dir /path/to/trajectory/data \
  --exit-statuses-yaml /path/to/exit_statuses.yaml \
  --evaluation-json /path/to/evaluation.json \
  --filter-instances unresolved
```

### 2. Submitted Cases Analysis (Solution-Oriented)

Run the specialized analysis for submitted cases:

```bash
# Run submitted analysis on unresolved cases
python scripts/run_submitted_analysis.py \
  --results-dir /path/to/trajectory/data \
  --exit-statuses-yaml /path/to/exit_statuses.yaml \
  --evaluation-json /path/to/evaluation.json \
  --filter-instances unresolved

# Generate visualizations for submitted analysis
python scripts/create_submitted_visualizations.py \
  --results_dir outputs/submitted_filtered_unresolved_dataset_analysis_results_*
```

## Analysis Types

### General Analysis Categories

The general analysis uses these failure categories:
- `wrong_solution`: Incorrect solution approach
- `identified_incorrect_file`: Wrong files targeted
- `tool_error`: Tool usage errors
- `infinite_loop`: Repetitive behavior
- `misunderstood_problem_statement`: Misinterpreted requirements
- `syntax_error`: Code syntax issues
- `context_overflow_from_listing`: Context overflow from file operations
- `endless_file_reading`: Excessive file reading
- `missed_edge_case`: Incomplete solution coverage
- `exit_context`: Context limit issues
- `other`: Unclassified failures

### Submitted Analysis Categories (Solution-Oriented)

The submitted analysis focuses on solution process failures:
- `issue_replication`: Failed to understand the issue before solving
- `solution_confirmation`: Failed to verify solution works / premature submission
- `logical_clarity`: Misinterpreted problem or made incorrect assumptions
- `hallucinations`: Imagined details not present in data
- `consistency`: Discrepancies between intentions and actions
- `efficiency`: Unnecessary steps or repeated actions

## Filtering Options

The pipeline supports several filtering modes:

- `--filter-instances unresolved`: Only analyze cases marked as unresolved in evaluation
- `--filter-instances resolved`: Only analyze cases marked as resolved
- `--filter-instances completed`: Analyze all evaluated cases (resolved + unresolved)
- No filtering: Analyze all available cases

**Important**: When filtering for "unresolved", the pipeline:
- Keeps ALL instances from non-submitted categories (they are inherently unresolved)
- Only filters submitted categories based on the evaluation JSON

## Output Structure

### Analysis Results

Each analysis run creates a timestamped directory:
```
outputs/
└── filtered_unresolved_dataset_analysis_results_20250808_20actions/
    ├── pipeline_summary.json              # Analysis metadata
    ├── filtered_exit_statuses.yaml        # Filtered instance data
    ├── category1_analysis_20actions.json   # Individual category results
    ├── category2_analysis_20actions.json
    └── seaborn_figures_20250808_123456/   # Timestamped visualizations
        ├── root_cause_by_exit_category.png
        ├── proportional_root_cause_by_category.png
        ├── failure_type_heatmap.png
        ├── failure_type_overview.png
        └── analysis_summary.txt
```

### Visualizations Generated

1. **Root Cause by Exit Category** - Primary stacked bar chart
2. **Proportional Root Cause Chart** - Maintains proportional heights while showing percentages
3. **Failure Type Heatmap** - Distribution across categories
4. **Failure Type Overview** - Overall distribution
5. **Analysis Summary** - Text report with statistics

## Advanced Usage

### Custom Output Directory

```bash
python scripts/run_all_categories.py \
  --results-dir /path/to/data \
  --exit-statuses-yaml /path/to/yaml \
  --output-subdir custom_analysis_name
```

### Skip Data Generation (Visualization Only)

```bash
python run_pipeline.py \
  --results-dir /path/to/data \
  --exit-statuses-yaml /path/to/yaml \
  --skip-data-generation
```

### Individual Steps

Run analysis steps separately:

```bash
# 1. Generate analysis data
python scripts/run_all_categories.py \
  --results-dir /path/to/data \
  --exit-statuses-yaml /path/to/yaml \
  --filter-instances unresolved \
  --evaluation-json /path/to/eval.json

# 2. Create visualizations
python scripts/create_seaborn_visualizations.py \
  --results_dir ../outputs/analysis_results_directory
```

## API Configuration

The pipeline uses GPT-4.1 via LiteLLM proxy with conservative rate limiting:
- `max_concurrent=5`: Maximum concurrent API calls
- `batch_size=10`: Batch processing size
- Automatic retry logic for failed requests

Set your API key:
```bash
export OPENAI_API_KEY=your_litellm_key_here
```

## Examples

### Multi-Dataset Analysis

```bash
# Analyze best_DAgger dataset (unresolved cases)
python run_pipeline.py \
  --results-dir /path/to/best_DAgger \
  --exit-statuses-yaml /path/to/best_DAgger/run_batch_exit_statuses.yaml \
  --evaluation-json /path/to/best_DAgger/0723.swebench_evaluation.json \
  --filter-instances unresolved

# Analyze smith7b dataset (unresolved cases)
python run_pipeline.py \
  --results-dir /path/to/smith7b \
  --exit-statuses-yaml /path/to/smith7b/run_batch_exit_statuses.yaml \
  --evaluation-json /path/to/smith7b/0717.swebench_evaluation.json \
  --filter-instances unresolved

# Run submitted analysis on best_DAgger
python scripts/run_submitted_analysis.py \
  --results-dir /path/to/best_DAgger \
  --exit-statuses-yaml /path/to/best_DAgger/run_batch_exit_statuses.yaml \
  --evaluation-json /path/to/best_DAgger/0723.swebench_evaluation.json \
  --filter-instances unresolved
```

## Color Schemes

The visualizations use professional, research-paper-appropriate color schemes:

**General Analysis Colors:**
- Deep blue, orange, green, red, purple, brown, pink, gray, olive, cyan

**Submitted Analysis Colors:**
- Consistent mapping for solution-oriented categories with distinct, deep colors

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

2. **File Not Found Errors**
   - Ensure you're in the `seaborn_pipeline` directory
   - Use absolute paths for input files

3. **Empty Filter Results**
   - Check that the evaluation JSON contains the expected IDs
   - Verify the filter type matches your needs

4. **Memory Issues**
   - Reduce `batch_size` in analysis scripts
   - Process smaller datasets

### Error Recovery

If analysis is interrupted:
1. Check the outputs directory for partial results
2. Use `--skip-data-generation` to only regenerate visualizations
3. Check API key validity and quota

## Dependencies

- Python 3.8+
- openai
- pandas
- matplotlib
- seaborn
- pyyaml
- tqdm
- aiofiles
- pathlib

## Contributing

When adding new features:
1. Maintain backward compatibility
2. Update this README
3. Add appropriate error handling
4. Test with multiple datasets
