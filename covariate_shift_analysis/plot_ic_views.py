#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _import_overlay_plotter():
    try:
        # Local import when run from repo root
        import plot_ic_overlay_with_ci as overlay  # type: ignore
        return overlay
    except Exception:
        # Relative import when run from within OOD_Validation_pipeline
        import sys
        # Ensure the repository src directory is on sys.path
        p = Path(__file__).resolve()
        # .../src/models/agent/evals/swe/OOD_validation/OOD_Validation_pipeline/plot_ic_views.py
        # parents[6] should be the 'src' directory
        try:
            src_dir = p.parents[6]
        except Exception:
            src_dir = p.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        import plot_ic_overlay_with_ci as overlay  # type: ignore
        return overlay


overlay = _import_overlay_plotter()


def main() -> None:
    ap = argparse.ArgumentParser(description='Plot IC overlays (median, median+1sd, full)')
    ap.add_argument('--manifest', required=False, help='Path to manifest.json from compute step')
    ap.add_argument('--run-dir', required=False, help='Path to compute run directory (auto-discover files)')
    ap.add_argument('--out-dir', required=False, default=None, help='Directory to write plots; defaults to manifest run_dir')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--no-legend', action='store_true')
    ap.add_argument('--no-baseline-on-full', action='store_true')
    ap.add_argument('--metric', choices=['fid','kl'], default='fid')
    args = ap.parse_args()

    if not args.manifest and not args.run_dir:
        raise SystemExit('Provide either --manifest or --run-dir')
    if args.manifest:
        manifest_path = Path(args.manifest)
        data = json.loads(manifest_path.read_text())
        run_dir = Path(data['run_dir'])
    else:
        run_dir = Path(args.run_dir)
        # Auto-discover files
        # Pick two student metrics files (prefers 7B then 32B by name)
        metrics = sorted(run_dir.glob('turnwise_metrics_ic__*.csv'))
        cis_fid = sorted(run_dir.glob('turnwise_ci_ic_fid__*.csv'))
        cis_kl = sorted(run_dir.glob('turnwise_ci_ic_kl__*.csv'))
        exp_txt = run_dir / 'experiment.txt'
        if not (metrics and cis_fid and cis_kl and exp_txt.exists()):
            raise SystemExit('Could not auto-discover required files in run-dir')
        # Heuristic order: 7B before 32B
        def _prefer_7b_32b(paths):
            s7 = [p for p in paths if '7B' in p.name or 'smith7b' in p.name]
            s32 = [p for p in paths if '32B' in p.name or 'smith32b' in p.name]
            rest = [p for p in paths if p not in s7 and p not in s32]
            out = []
            out.extend(s7[:1])
            out.extend(s32[:1])
            if len(out) < 2 and rest:
                out.extend(rest[: 2 - len(out)])
            return out[:2]
        metrics = _prefer_7b_32b(metrics)
        cis_fid = _prefer_7b_32b(cis_fid)
        cis_kl = _prefer_7b_32b(cis_kl)
        labels = []
        for p in metrics:
            # Extract label from filename tail
            name = p.stem.split('__')[-1]
            labels.append(name)
        data = {
            'run_dir': str(run_dir),
            'student_labels': labels if len(labels) == 2 else ['student-1','student-2'],
            'metrics': [str(p) for p in metrics],
            'cis_fid': [str(p) for p in cis_fid],
            'cis_kl': [str(p) for p in cis_kl],
            'experiment_txt': str(exp_txt),
        }
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = data.get('student_labels', ['SWE-smith-LM-7B','SWE-smith-LM-32B'])
    metrics = data['metrics']
    cis = data['cis_fid'] if args.metric == 'fid' else data['cis_kl']
    exp = data['experiment_txt']
    baseline = data.get('baseline')  # May be absent if baseline omitted

    # Common args
    common = dict(
        labels=','.join(labels),
        metrics=','.join(metrics),
        cis=','.join(cis),
        experiments=','.join([exp, exp]),
        drop_first_turn=True,
        metric=args.metric,
        no_title=True,
        dpi=int(args.dpi),
        x_divisor=2.0,
        line_width=2.0,
    )

    # Median
    overlay.main.__wrapped__ if hasattr(overlay.main, '__wrapped__') else None
    out_pdf = out_dir / f"overlay_{args.metric}_median.pdf"
    overlay.main = overlay.main  # silence linters: we're just calling main via CLI-style args
    overlay.sys = __import__('sys')
    argv = [
        'prog',
        '--labels', common['labels'],
        '--metrics', common['metrics'],
        '--cis', common['cis'],
        '--experiments', common['experiments'],
        '--cutoff', 'median',
        '--drop-first-turn',
        '--metric', args.metric,
        '--no-title',
        '--dpi', str(common['dpi']),
        '--x-divisor', '2.0',
        '--line-width', '2.0',
        '--out', str(out_pdf),
    ]
    if args.no_legend:
        argv.append('--no-legend')
    overlay.sys.argv = argv
    overlay.main()

    # Median + 1 SD
    out_pdf = out_dir / f"overlay_{args.metric}_median_p1sd.pdf"
    argv = [
        'prog',
        '--labels', common['labels'],
        '--metrics', common['metrics'],
        '--cis', common['cis'],
        '--experiments', common['experiments'],
        '--cutoff', 'median_plus1sd',
        '--drop-first-turn',
        '--metric', args.metric,
        '--no-title',
        '--dpi', str(common['dpi']),
        '--x-divisor', '2.0',
        '--line-width', '3.0',
        '--out', str(out_pdf),
    ]
    if args.no_legend:
        argv.append('--no-legend')
    overlay.sys.argv = argv
    overlay.main()

    # Full (no baseline by default if none provided)
    out_pdf = out_dir / f"overlay_{args.metric}_full.pdf"
    argv = [
        'prog',
        '--labels', common['labels'],
        '--metrics', common['metrics'],
        '--cis', common['cis'],
        '--experiments', common['experiments'],
        '--cutoff', 'full',
        '--drop-first-turn',
        '--metric', args.metric,
        '--no-title',
        '--dpi', str(common['dpi']),
        '--x-divisor', '2.0',
        '--line-width', '2.0',
        '--out', str(out_pdf),
        '--trim-tail', '6',
    ]
    if args.no_legend:
        argv.append('--no-legend')
    if args.no_baseline_on_full or not baseline:
        argv.append('--no-baseline')
        argv.append('--no-vline')
    else:
        # pass the baseline path explicitly
        argv.extend(['--baseline', str(baseline)])
        argv.append('--no-vline')
    overlay.sys.argv = argv
    overlay.main()

    print(f"Saved IC overlays to {out_dir}")


if __name__ == '__main__':
    main()


