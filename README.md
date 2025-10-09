A fork of SWE-agent for generating OEC trajectories. 

Student models will have to be hosted (e.g., with vLLM) and the correct names and ports with have to be specified in "config/qwen32b_switch_claude_python_tools.yaml". Then run ```generate_oec.sh``` to generate OEC trajectories. Then use ```eval_swesmith.sh```, ```convert_to_sft.sh```, and ```trajectories/prep_for_sft.sh``` in order to generate data for SFT.

Training problem instances can be sourced from the SWE-smith Github: https://github.com/SWE-bench/SWE-smith.

```failure_categorization``` contains code for the LLM-as-judge categorization of trajectories into buckets.

```covariate_shift_analysis``` contains code for embedding SWE-agent trajectories and computing the divergence.
