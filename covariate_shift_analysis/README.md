Run compute:

```bash
python compute_ic_metrics.py \
  --student-7b-pkl <path/to/student7b.pkl> \
  --student-32b-pkl <path/to/student32b.pkl> \
  --expert-pkl <path/to/expert.pkl> \
  --output-dir <path/to/output_dir> \
  [--num-instances 10] \
  [--replicates 500] \
  [--max-rollouts-per-instance 50] \
  [--drop-first-turn] \
  [--role-only {assistant|user}] \
  [--pca-dim <int>]
```

Args:
- --student-7b-pkl: student 7B embeddings PKL
- --student-32b-pkl: student 32B embeddings PKL
- --expert-pkl: expert embeddings PKL
- --output-dir: directory for timestamped run outputs
- --num-instances: number of problem instances (default 10)
- --replicates: paired bootstrap replicates (default 500)
- --max-rollouts-per-instance: cap per-instance rollouts (default 50)
- --drop-first-turn: drop turn 0
- --role-only: filter turns by role
- --pca-dim: optional PCA dimensionality

Run plots (three views: median, median+1sd, full):

```bash
# Either with manifest.json
python plot_ic_views.py --manifest <output_dir>/<timestamp>/manifest.json --dpi 300 --metric fid
python plot_ic_views.py --manifest <output_dir>/<timestamp>/manifest.json --dpi 300 --metric kl

# Or simply with --run-dir (auto-discover files)
python plot_ic_views.py --run-dir <output_dir>/<timestamp> --dpi 300 --metric fid
python plot_ic_views.py --run-dir <output_dir>/<timestamp> --dpi 300 --metric kl
```

Outputs:
- overlay_{fid,kl}_{median,median_p1sd,full}.pdf in <output_dir>/<timestamp>/


