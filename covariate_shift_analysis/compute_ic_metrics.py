#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def _import_metrics_module():
    """Robustly import compute_turnwise_metrics utilities from the parent directory."""
    try:
        # When executed as a package module
        from ..compute_turnwise_metrics import _load_embeddings, compute_turnwise_metrics  # type: ignore
        return _load_embeddings, compute_turnwise_metrics
    except Exception:
        # Fallback to direct path manipulation
        import sys
        base = Path(__file__).resolve().parents[1]
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        from compute_turnwise_metrics import _load_embeddings, compute_turnwise_metrics  # type: ignore
        return _load_embeddings, compute_turnwise_metrics


_load_embeddings, compute_turnwise_metrics = _import_metrics_module()


def _filter_nonfinite(emb_map: Dict[str, dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for tid, rec in emb_map.items():
        embs = rec.get('embeddings')
        try:
            if embs is not None and np.isfinite(embs).all():
                out[tid] = rec
        except Exception:
            continue
    return out


def _index_by_instance(emb_map: Dict[str, dict]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for tid, rec in emb_map.items():
        inst = rec.get('instance_id')
        if inst is None:
            continue
        idx.setdefault(str(inst), []).append(tid)
    return idx


def _filter_to_instances(emb_map: Dict[str, dict], allowed: set[str]) -> Dict[str, dict]:
    return {tid: rec for tid, rec in emb_map.items() if str(rec.get('instance_id')) in allowed}


def _choose_fixed_instances(shared_instances: List[str], s1: Dict[str, dict], s2: Dict[str, dict], e: Dict[str, dict], n: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    s1_idx = _index_by_instance(s1)
    s2_idx = _index_by_instance(s2)
    e_idx = _index_by_instance(e)
    eligible = [inst for inst in shared_instances if len(e_idx.get(inst, [])) >= 2 and len(s1_idx.get(inst, [])) >= 1 and len(s2_idx.get(inst, [])) >= 1]
    if not eligible:
        return []
    if len(eligible) <= n:
        return list(eligible)
    return list(rng.choice(eligible, size=n, replace=False))


def _cap_rollouts_per_instance(src: Dict[str, dict], max_per_instance: int, seed: int) -> Dict[str, dict]:
    if max_per_instance is None or max_per_instance <= 0:
        return src
    rng = np.random.default_rng(seed)
    idx = _index_by_instance(src)
    out: Dict[str, dict] = {}
    for inst, tids in idx.items():
        if not tids:
            continue
        if len(tids) <= max_per_instance:
            for tid in tids:
                out[tid] = src[tid]
        else:
            picks = list(rng.choice(tids, size=max_per_instance, replace=False))
            for tid in picks:
                out[tid] = src[tid]
    return out


def _split_expert_half_per_instance(expert_map: Dict[str, dict], seed: int) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Deprecated: retained for backward compatibility but unused (baseline omitted)."""
    return {}, {}


def _sample_one_per_instance(src: Dict[str, dict], inst_ids: List[str], rng: np.random.Generator) -> Dict[str, dict]:
    idx = _index_by_instance(src)
    out: Dict[str, dict] = {}
    dup: Dict[str, int] = {}
    for inst in inst_ids:
        tids = idx.get(inst, [])
        if not tids:
            continue
        tid = str(rng.choice(tids))
        key = tid
        if key in out:
            dup[tid] = dup.get(tid, 0) + 1
            key = f"{tid}##rep{dup[tid]}"
        out[key] = src[tid]
    return out


def _eff_last_turn(rec: dict, drop_first_turn: bool, role_filter: Optional[str]) -> int:
    embs = rec.get('embeddings')
    n = int(embs.shape[0]) if embs is not None else 0
    off = 1 if drop_first_turn else 0
    # Role filtering may reduce effective length; approximate by alternating pattern if roles absent
    if role_filter is None:
        return max(0, n - off - 1)
    roles = rec.get('roles')
    count = 0
    for gi in range(off, n):
        if isinstance(roles, (list, tuple)) and len(roles) == n:
            if str(roles[gi]).lower() == role_filter.lower():
                count += 1
        else:
            rf = role_filter.lower()
            if rf == 'assistant' and gi >= 2 and (gi % 2 == 0):
                count += 1
            elif rf == 'user' and gi >= 1 and (gi % 2 == 1):
                count += 1
    return max(0, count - 1)


def _model_from_pkl_path(p: str) -> str:
    m = re.search(r'__([A-Za-z0-9_.\-]+)__', p)
    return m.group(1) if m else Path(p).stem


def _percentile_band(mat: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(mat, axis=0)
    lo_v = np.nanpercentile(mat, lo, axis=0)
    hi_v = np.nanpercentile(mat, hi, axis=0)
    return mu, lo_v, hi_v


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute IC FD/KL metrics with paired bootstrap CIs (paper-aligned)')
    ap.add_argument('--student-7b-pkl', required=True)
    ap.add_argument('--student-32b-pkl', required=True)
    ap.add_argument('--expert-pkl', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--num-instances', type=int, default=10)
    ap.add_argument('--replicates', type=int, default=500)
    ap.add_argument('--max-rollouts-per-instance', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eps', type=float, default=1e-6)
    ap.add_argument('--drop-first-turn', action='store_true')
    ap.add_argument('--role-only', choices=['assistant','user'], default=None)
    ap.add_argument('--pca-dim', type=int, default=None, help='Optional PCA dim; if set, uses PCA whitening=off')
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = out_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load and sanitize
    s7_path = args.student_7b_pkl
    s32_path = args.student_32b_pkl
    e_path = args.expert_pkl
    s7_map = _filter_nonfinite(_load_embeddings(s7_path))
    s32_map = _filter_nonfinite(_load_embeddings(s32_path))
    e_map = _filter_nonfinite(_load_embeddings(e_path))

    # Restrict to shared instances
    s7_idx = _index_by_instance(s7_map)
    s32_idx = _index_by_instance(s32_map)
    e_idx = _index_by_instance(e_map)
    shared = set(s7_idx.keys()) & set(s32_idx.keys()) & set(e_idx.keys())
    if not shared:
        raise SystemExit('No shared instance_ids across 7B, 32B, and expert.')
    s7_map = _filter_to_instances(s7_map, shared)
    s32_map = _filter_to_instances(s32_map, shared)
    e_map = _filter_to_instances(e_map, shared)

    # Cap per-instance rollouts for stability and parity with paper setup
    s7_map = _cap_rollouts_per_instance(s7_map, args.max_rollouts_per_instance, seed=args.seed + 1)
    s32_map = _cap_rollouts_per_instance(s32_map, args.max_rollouts_per_instance, seed=args.seed + 2)
    e_map = _cap_rollouts_per_instance(e_map, args.max_rollouts_per_instance, seed=args.seed + 3)

    # Choose fixed instances (require expert>=1; baseline omitted)
    shared_now = sorted(set(_index_by_instance(e_map).keys()) & set(_index_by_instance(s7_map).keys()) & set(_index_by_instance(s32_map).keys()))
    chosen = _choose_fixed_instances(shared_now, s7_map, s32_map, e_map, n=int(max(1, args.num_instances)), seed=args.seed + 10)
    if not chosen:
        raise SystemExit('No eligible instances after filtering (need expert>=2 and students>=1 each).')
    if len(chosen) < args.num_instances:
        print(f"[warn] Eligible instances fewer than requested: using {len(chosen)}")

    # Filter maps to chosen instances
    chosen_set = set(chosen)
    s7_map = _filter_to_instances(s7_map, chosen_set)
    s32_map = _filter_to_instances(s32_map, chosen_set)
    e_map = _filter_to_instances(e_map, chosen_set)

    # Baseline omitted; use full expert set as comparator for student-vs-expert

    # Compute expert effective turn distribution for cutoffs
    last_turns = np.array([
        _eff_last_turn(rec, args.drop_first_turn, args.role_only)
        for rec in e_map.values()
        if rec.get('embeddings') is not None
    ], dtype=int)
    med = int(np.median(last_turns)) if last_turns.size else 0
    sd = int(round(np.std(last_turns))) if last_turns.size else 0

    # Optional PCA
    pca_model = None
    if args.pca_dim is not None:
        try:
            from sklearn.decomposition import PCA  # type: ignore
        except Exception as _e:
            raise RuntimeError('scikit-learn is required for --pca-dim')
        # Fit PCA on combined sample of student+expert
        def _sample_vectors(src_map: Dict[str, dict], max_samples: int, rng: np.random.Generator) -> np.ndarray:
            samples: List[np.ndarray] = []
            for _, rec in src_map.items():
                embs = rec['embeddings']
                T = embs.shape[0]
                idxs = np.arange(T)
                rng.shuffle(idxs)
                for t in idxs[: min(4, len(idxs))]:
                    samples.append(embs[t].astype(np.float32, copy=False))
                    if len(samples) >= max_samples:
                        return np.stack(samples, axis=0)
            return np.stack(samples, axis=0)
        rng_p = np.random.default_rng(args.seed + 77)
        pool = []
        pool.append(_sample_vectors(s7_map, 10000, rng_p))
        pool.append(_sample_vectors(s32_map, 10000, rng_p))
        pool.append(_sample_vectors(e_map, 10000, rng_p))
        fit_mat = np.concatenate(pool, axis=0)
        pca_model = PCA(n_components=int(args.pca_dim), whiten=False, svd_solver='randomized', random_state=args.seed)
        pca_model.fit(fit_mat)
        print(f"[pca] Fitted PCA to {fit_mat.shape[0]} vectors; out dim={pca_model.n_components_}")

    # Bootstrap replicates (paired over instances)
    R = int(max(1, args.replicates))
    N = len(chosen)
    rng = np.random.default_rng(args.seed)

    # Collect replicate per-turn arrays
    s7_fid_reps: List[List[float]] = []
    s7_kl_reps: List[List[float]] = []
    s32_fid_reps: List[List[float]] = []
    s32_kl_reps: List[List[float]] = []

    for r in range(R):
        rng_r = np.random.default_rng(args.seed + 1000 + r)
        inst_boot = list(rng.choice(chosen, size=max(1, N), replace=True))
        # Build maps with one rollout per instance for each group
        s7_boot = _sample_one_per_instance(s7_map, inst_boot, rng_r)
        s32_boot = _sample_one_per_instance(s32_map, inst_boot, rng_r)
        eB_boot = _sample_one_per_instance(e_map, inst_boot, rng_r)

        # Students vs expert-B
        res_7 = compute_turnwise_metrics(
            student_map=s7_boot,
            expert_map=eB_boot,
            max_turns=None,
            eps=float(args.eps),
            subsample_student=None,
            subsample_expert=None,
            seed=args.seed + r,
            drop_first_turn=args.drop_first_turn,
            pca_model=pca_model,
            role_filter=args.role_only,
        )
        res_32 = compute_turnwise_metrics(
            student_map=s32_boot,
            expert_map=eB_boot,
            max_turns=None,
            eps=float(args.eps),
            subsample_student=None,
            subsample_expert=None,
            seed=args.seed + r + 17,
            drop_first_turn=args.drop_first_turn,
            pca_model=pca_model,
            role_filter=args.role_only,
        )

        s7_fid_reps.append(list(res_7['fid']))
        s7_kl_reps.append(list(res_7['kl_student_expert']))
        s32_fid_reps.append(list(res_32['fid']))
        s32_kl_reps.append(list(res_32['kl_student_expert']))

    # Harmonize lengths (pad/truncate with NaNs to common T)
    def _to_mat(reps: List[List[float]]) -> np.ndarray:
        max_len = max((len(s) for s in reps), default=0)
        mat = np.full((len(reps), max_len), np.nan, dtype=float)
        for i, s in enumerate(reps):
            if not s:
                continue
            n = min(len(s), max_len)
            mat[i, :n] = np.asarray(s[:n], dtype=float)
        return mat

    s7_fid_mat = _to_mat(s7_fid_reps)
    s7_kl_mat = _to_mat(s7_kl_reps)
    s32_fid_mat = _to_mat(s32_fid_reps)
    s32_kl_mat = _to_mat(s32_kl_reps)
    # Baseline omitted

    # Aggregate
    s7_fid_mu, s7_fid_lo, s7_fid_hi = _percentile_band(s7_fid_mat)
    s7_kl_mu, s7_kl_lo, s7_kl_hi = _percentile_band(s7_kl_mat)
    s32_fid_mu, s32_fid_lo, s32_fid_hi = _percentile_band(s32_fid_mat)
    s32_kl_mu, s32_kl_lo, s32_kl_hi = _percentile_band(s32_kl_mat)
    # Baseline omitted

    # Output helpers
    def _save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            with open(path, 'w') as f:
                f.write('')
            return
        cols = list(rows[0].keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Build identifiers
    tag = f"ic__N{len(chosen)}__R{R}__drop1={int(args.drop_first_turn)}__role={args.role_only or 'all'}"
    student7_label = _model_from_pkl_path(s7_path)
    student32_label = _model_from_pkl_path(s32_path)

    # Write student metrics and CI CSVs
    T7 = s7_fid_mu.size
    turns7 = list(range(T7))
    rows_metrics_7 = [
        {
            'turn': int(t),
            'fid': float(s7_fid_mu[t]),
            'kl_student_expert': float(s7_kl_mu[t]),
        }
        for t in range(T7)
    ]
    rows_ci_fid_7 = [
        {'turn': int(t), 'lo': float(s7_fid_lo[t]), 'hi': float(s7_fid_hi[t])}
        for t in range(T7)
    ]
    rows_ci_kl_7 = [
        {'turn': int(t), 'lo': float(s7_kl_lo[t]), 'hi': float(s7_kl_hi[t])}
        for t in range(T7)
    ]
    _save_csv(run_dir / f"turnwise_metrics_ic__{tag}__{student7_label}.csv", rows_metrics_7)
    _save_csv(run_dir / f"turnwise_ci_ic_fid__{tag}__{student7_label}.csv", rows_ci_fid_7)
    _save_csv(run_dir / f"turnwise_ci_ic_kl__{tag}__{student7_label}.csv", rows_ci_kl_7)

    # Student 32B
    T32 = s32_fid_mu.size
    rows_metrics_32 = [
        {
            'turn': int(t),
            'fid': float(s32_fid_mu[t]),
            'kl_student_expert': float(s32_kl_mu[t]),
        }
        for t in range(T32)
    ]
    rows_ci_fid_32 = [
        {'turn': int(t), 'lo': float(s32_fid_lo[t]), 'hi': float(s32_fid_hi[t])}
        for t in range(T32)
    ]
    rows_ci_kl_32 = [
        {'turn': int(t), 'lo': float(s32_kl_lo[t]), 'hi': float(s32_kl_hi[t])}
        for t in range(T32)
    ]
    _save_csv(run_dir / f"turnwise_metrics_ic__{tag}__{student32_label}.csv", rows_metrics_32)
    _save_csv(run_dir / f"turnwise_ci_ic_fid__{tag}__{student32_label}.csv", rows_ci_fid_32)
    _save_csv(run_dir / f"turnwise_ci_ic_kl__{tag}__{student32_label}.csv", rows_ci_kl_32)

    # Baseline omitted

    # Experiment metadata for plotting
    exp_txt = []
    exp_txt.append(f"student_pkl: {s7_path}")
    exp_txt.append(f"student_pkl: {s32_path}")
    exp_txt.append(f"expert_pkl: {e_path}")
    exp_txt.append(f"num_instances: {len(chosen)}")
    exp_txt.append(f"replicates: {R}")
    exp_txt.append(f"drop_first_turn: {int(args.drop_first_turn)}")
    exp_txt.append(f"role_only: {args.role_only or 'all'}")
    exp_txt.append(f"median_expert_length_all: {med}")
    exp_txt.append(f"std_expert_length_all: {sd}")
    exp_txt.append(f"median_plus1sd_all: {med + sd}")
    exp_txt.append(f"chosen_instances: {json.dumps(chosen)}")
    exp_txt.append(f"eps: {args.eps}")
    (run_dir / 'experiment.txt').write_text('\n'.join(exp_txt) + '\n')

    # Save a small manifest to ease plotting discovery
    manifest = {
        'run_dir': str(run_dir),
        'tag': tag,
        'student_labels': [student7_label, student32_label],
        'metrics': [
            str(run_dir / f"turnwise_metrics_ic__{tag}__{student7_label}.csv"),
            str(run_dir / f"turnwise_metrics_ic__{tag}__{student32_label}.csv"),
        ],
        'cis_fid': [
            str(run_dir / f"turnwise_ci_ic_fid__{tag}__{student7_label}.csv"),
            str(run_dir / f"turnwise_ci_ic_fid__{tag}__{student32_label}.csv"),
        ],
        'cis_kl': [
            str(run_dir / f"turnwise_ci_ic_kl__{tag}__{student7_label}.csv"),
            str(run_dir / f"turnwise_ci_ic_kl__{tag}__{student32_label}.csv"),
        ],
        'experiment_txt': str(run_dir / 'experiment.txt'),
    }
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    print(f"[done] Wrote outputs to {run_dir}")


if __name__ == '__main__':
    main()


