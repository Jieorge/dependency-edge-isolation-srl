"""
evaluate.py — Evaluate ablation/isolation models against baseline; generate plots.

Usage:
    python evaluate.py --mode ablation  --group core_args
    python evaluate.py --mode isolation --group core_args
    python evaluate.py --mode ablation  --group all
    python evaluate.py --mode isolation --group all
"""

import os, json, argparse
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import DataLoader

from shared import (
    read_conll09, SRLDataset, collate_fn,
    load_model_from_checkpoint, dep_distance,
)

DEPREL_GROUPS = {
    'core_args':     'Core predicate-argument relations',
    'clausal':       'Clausal and verb-chain relations',
    'adjuncts':      'Adjunct and modifier relations',
    'noun_internal': 'Noun-phrase internal relations',
    'coordination':  'Coordination and conjunction relations',
    'functional':    'Functional, punctuation, and vocative relations',
    'gap':           'Gap and long-distance dependency relations',
}


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ablation/isolation models vs baseline')
    p.add_argument('--mode', default='ablation', choices=['ablation', 'isolation'])
    p.add_argument('--group', required=True,
                   choices=list(DEPREL_GROUPS.keys()) + ['all'])
    p.add_argument('--baseline_ckpt', default='checkpoints/best_baseline.pt')
    p.add_argument('--ablation_dir',  default='checkpoints_ablation')
    p.add_argument('--isolation_dir', default='checkpoints_isolation')
    p.add_argument('--dev_file',
                   default='data/conll2009/CoNLL2009-ST-English-development.txt')
    p.add_argument('--results_dir',   default='results_ablation')
    p.add_argument('--batch_size',    type=int, default=64)
    p.add_argument('--num_workers',   type=int, default=4)
    return p.parse_args()


# ── F1 helpers ────────────────────────────────────────────────────────────────

def example_f1(gold, pred, null_id):
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        if g not in (0, null_id):
            tp += (p == g); fn += (p != g)
        if p not in (0, null_id) and p != g:
            fp += 1
    pr = tp / (tp + fp + 1e-9)
    re = tp / (tp + fn + 1e-9)
    return 2 * pr * re / (pr + re + 1e-9)


def corpus_f1(results, null_id):
    tp = fp = fn = 0
    for r in results:
        for g, p in zip(r['gold'], r['pred']):
            if g not in (0, null_id):
                tp += (p == g); fn += (p != g)
            if p not in (0, null_id) and p != g:
                fp += 1
    pr = tp / (tp + fp + 1e-9)
    re = tp / (tp + fn + 1e-9)
    return 2 * pr * re / (pr + re + 1e-9)


def per_role_f1(results, rv_inv, null_id):
    stats_dict = defaultdict(lambda: [0, 0, 0])
    for r in results:
        for g, p in zip(r['gold'], r['pred']):
            if g not in (0, null_id):
                label = rv_inv.get(g, str(g))
                stats_dict[label][0] += (p == g)
                stats_dict[label][2] += (p != g)
            if p not in (0, null_id) and p != g:
                label = rv_inv.get(p, str(p))
                stats_dict[label][1] += 1
    out = {}
    for label, (tp, fp, fn) in stats_dict.items():
        pr = tp / (tp + fp + 1e-9)
        re = tp / (tp + fn + 1e-9)
        out[label] = 2 * pr * re / (pr + re + 1e-9)
    return out


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, loader, device, null_id):
    model.eval()
    raw = []
    with torch.no_grad():
        for batch in loader:
            cw_cpu = batch.pop('cw')
            batch  = {k: v.to(device) for k, v in batch.items()}
            cw     = cw_cpu.to(device) if cw_cpu is not None else None
            logits = model(
                batch['word_ids'], batch['lemma_ids'], batch['pos_ids'],
                batch['pred_flag'], batch['pred_idxs'],
                batch['heads'],     batch['deprel_ids'],
                batch['mask'].float(), cw)
            preds = logits.argmax(-1).cpu()
            for b in range(preds.shape[0]):
                L = batch['lengths'][b].item()
                raw.append({
                    'sent_idx': batch['sent_idxs'][b].item(),
                    'gold':     batch['role_ids'][b, :L].tolist(),
                    'pred':     preds[b, :L].tolist(),
                    'dep_dist': dep_distance(batch['heads'][b, :L].tolist()),
                    'ndr':      batch['ndrs'][b].item(),
                })

    sent_groups = defaultdict(list)
    for r in raw:
        sent_groups[r['sent_idx']].append(r)

    results = []
    for sent_idx, grp in sorted(sent_groups.items()):
        all_gold = [g for r in grp for g in r['gold']]
        all_pred = [p for r in grp for p in r['pred']]
        results.append({
            'sent_idx': sent_idx,
            'gold':     all_gold,
            'pred':     all_pred,
            'dep_dist': grp[0]['dep_dist'],
            'ndr':      grp[0]['ndr'],
            'f1':       example_f1(all_gold, all_pred, null_id),
        })
    return results


# ── Scatter + regression plot helper ─────────────────────────────────────────

def scatter_regression(ax, x, y, color, label, zero_hline=False):
    ax.scatter(x, y, alpha=0.2, s=8, color=color)
    if len(set(x)) < 2:
        ax.plot([], [], color=color, linewidth=2,
                label=f'{label}  (insufficient x variance)')
        if zero_hline:
            ax.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
        return 0.0, 0.0, 1.0
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 100)
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2,
            label=f'{label}  slope={slope:.4f}, r={r:.3f}, p={p:.4f}')
    if zero_hline:
        ax.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    return slope, r, p


# ── Per-group scatter plots (MDD and NDR) ─────────────────────────────────────

def make_scatter_plots(res_ref, res_exp, group, mode, results_dir, dep_dists, ndrs):
    """
    dep_dists and ndrs are always computed from the full-graph dev set
    so they reflect true sentence complexity regardless of edge masking.
    """
    f1_base  = np.array([r['f1'] for r in res_ref])
    f1_exp   = np.array([r['f1'] for r in res_exp])
    delta_f1 = f1_exp - f1_base

    for metric, x_vals, xlabel in [
        ('mdd', dep_dists, 'Mean Dependency Distance (Liu 2008)'),
        ('ndr', ndrs,      'Mean New Discourse Referents (Gibson 1998)'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        scatter_regression(axes[0], x_vals, f1_base, 'steelblue',
                           'zero-edge' if mode == 'isolation' else 'Baseline')
        scatter_regression(axes[0], x_vals, f1_exp,  'tomato', f'{mode}/{group}')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('F1 — sentence level')
        axes[0].set_title(f'F1 vs {metric.upper()} — {mode}/{group}')
        axes[0].legend(fontsize=8)

        slope, r, p = scatter_regression(
            axes[1], x_vals, delta_f1, 'mediumpurple',
            f'slope={0:.4f}', zero_hline=True)
        axes[1].clear()
        scatter_regression(axes[1], x_vals, delta_f1, 'mediumpurple',
                           f'slope={slope:.4f}, r={r:.3f}, p={p:.4f}',
                           zero_hline=True)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('ΔF1 (experiment − reference) — sentence level')
        axes[1].set_title(f'ΔF1 vs {metric.upper()} — {mode}/{group}')
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        out = os.path.join(results_dir, f'{mode}_scatter_{metric}_{group}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'Saved: {out}')

    return f1_base, f1_exp, delta_f1


# ── Summary plots (run after all groups evaluated) ────────────────────────────

def plot_f1_drop(results_dir, mode):
    path = os.path.join(results_dir, f'{mode}_summary.json')
    if not os.path.exists(path):
        print(f'Not found: {path}'); return
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows).sort_values('delta_f1')

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.7)))
    colors = ['tomato' if d < 0 else 'steelblue' for d in df['delta_f1']]
    bars = ax.barh(df['group'], df['delta_f1'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('ΔF1 (experiment − baseline)')
    ax.set_title(f'F1 Change by Dependency Group ({mode})')
    for bar, val in zip(bars, df['delta_f1']):
        ax.text(val - 0.0003 if val < 0 else val + 0.0003,
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.4f}', va='center',
                ha='right' if val < 0 else 'left', fontsize=9)
    plt.tight_layout()
    out = os.path.join(results_dir, f'{mode}_f1_drop.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')


def plot_role_heatmap(results_dir, mode):
    path = os.path.join(results_dir, f'{mode}_roles.csv')
    if not os.path.exists(path):
        print(f'Not found: {path}'); return
    df    = pd.read_csv(path)
    pivot = df.pivot(index='group', columns='role', values='delta_f1')

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.7),
                                    max(4,  len(pivot.index)   * 0.6)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label='ΔF1')
    ax.set_title(f'ΔF1 by Group and Argument Role — {mode}')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.3f}', ha='center', va='center', fontsize=7,
                        color='black' if abs(val) < 0.07 else 'white')
    plt.tight_layout()
    out = os.path.join(results_dir, f'{mode}_role_heatmap.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_iso      = (args.mode == 'isolation')
    results_dir = (args.results_dir.replace('ablation', 'isolation')
                   if is_iso else args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if args.group == 'all':
        print(f'Generating summary plots ({args.mode})...')
        plot_f1_drop(results_dir, args.mode)
        plot_role_heatmap(results_dir, args.mode)
        return

    group     = args.group
    ckpt_dir  = args.isolation_dir if is_iso else args.ablation_dir
    ckpt_name = f'best_isolation_{group}.pt' if is_iso else f'best_ablation_{group}.pt'
    exp_ckpt  = os.path.join(ckpt_dir, ckpt_name)

    print(f'Loading baseline:  {args.baseline_ckpt}')
    m_base, vv, _ = load_model_from_checkpoint(args.baseline_ckpt, device)

    # Isolation experiments compare against the zero-edge baseline (no_edge),
    # not the full-graph baseline, to measure each group's independent contribution.
    if is_iso:
        no_edge_ckpt = os.path.join(args.isolation_dir, 'best_isolation_no_edge.pt')
        print(f'Loading zero-edge baseline: {no_edge_ckpt}')
        m_ref, _, _ = load_model_from_checkpoint(no_edge_ckpt, device)
    else:
        m_ref = m_base

    print(f'Loading {args.mode} ({group}): {exp_ckpt}')
    m_exp, _, _ = load_model_from_checkpoint(exp_ckpt, device)

    wv, lv, pv, dv, rv = vv['word'], vv['lemma'], vv['pos'], vv['deprel'], vv['role']
    null_id = rv.get('NULL', 1)
    rv_inv  = {v: k for k, v in rv.items()}

    print('Loading dev data...')
    dev_sents = read_conll09(args.dev_file)

    exp_ck       = torch.load(exp_ckpt, map_location='cpu', weights_only=False)
    ablated_ids  = set(exp_ck.get('ablated_ids',  [])) or None
    isolated_ids = set(exp_ck.get('isolated_ids', [])) or None

    # Full-graph dataset — used only to extract dep_dist/ndr for scatter plots
    dev_ds_full = SRLDataset(dev_sents, wv, lv, pv, dv, rv)
    loader_full = DataLoader(dev_ds_full, args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    # Reference loader: zero-edge for isolation, full graph for ablation
    if is_iso:
        dev_ds_ref = SRLDataset(dev_sents, wv, lv, pv, dv, rv,
                                isolated_ids=set())
    else:
        dev_ds_ref = dev_ds_full

    dev_ds_exp  = SRLDataset(dev_sents, wv, lv, pv, dv, rv,
                              ablated_ids=ablated_ids, isolated_ids=isolated_ids)
    loader_ref  = DataLoader(dev_ds_ref, args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)
    loader_exp  = DataLoader(dev_ds_exp, args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    # Complexity metrics always from full-graph inference
    print('Computing sentence complexity metrics from full graph...')
    res_full  = run_inference(m_base, loader_full, device, null_id)
    dep_dists = np.array([r['dep_dist'] for r in res_full])
    ndrs      = np.array([r['ndr']      for r in res_full])

    ref_label = 'zero-edge' if is_iso else 'baseline'
    print(f'Running {ref_label} inference...')
    res_base = run_inference(m_ref, loader_ref, device, null_id)
    print(f'Running {args.mode} ({group}) inference...')
    res_exp  = run_inference(m_exp, loader_exp,  device, null_id)

    f1_base = corpus_f1(res_base, null_id)
    f1_exp  = corpus_f1(res_exp,  null_id)
    delta   = f1_exp - f1_base
    print(f'\nOverall F1 — {ref_label}={f1_base:.4f}  {args.mode}/{group}={f1_exp:.4f}  Δ={delta:+.4f}')

    # ── Scatter plots + raw sentence-level data ───────────────────────────────
    f1_b_arr, f1_e_arr, delta_arr = make_scatter_plots(
        res_base, res_exp, group, args.mode, results_dir, dep_dists, ndrs)

    # Save sentence-level CSV
    sent_csv = os.path.join(results_dir, f'sent_results_{args.mode}_{group}.csv')
    pd.DataFrame({
        'sent_idx':              [r['sent_idx'] for r in res_base],
        'dep_dist':              dep_dists,
        'ndr':                   ndrs,
        f'f1_{ref_label}':       f1_b_arr,
        f'f1_{args.mode}_{group}': f1_e_arr,
        'delta_f1':              delta_arr,
    }).to_csv(sent_csv, index=False)
    print(f'Saved: {sent_csv}')

    # ── Role F1 ───────────────────────────────────────────────────────────────
    roles_base = per_role_f1(res_base, rv_inv, null_id)
    roles_exp  = per_role_f1(res_exp,  rv_inv, null_id)
    role_rows  = [{'group': group, 'role': r,
                   'delta_f1': roles_exp.get(r, 0.0) - roles_base[r]}
                  for r in sorted(roles_base)]

    # ── Persist aggregate results ─────────────────────────────────────────────
    summary_path = os.path.join(results_dir, f'{args.mode}_summary.json')
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
    existing = [r for r in existing if r['group'] != group]
    existing.append({
        'group': group, 'description': DEPREL_GROUPS[group],
        'reference': ref_label, 'f1_reference': f1_base,
        f'f1_{args.mode}': f1_exp, 'delta_f1': delta,
    })
    with open(summary_path, 'w') as f:
        json.dump(existing, f, indent=2)

    role_fpath = os.path.join(results_dir, f'{args.mode}_roles.csv')
    new_df = pd.DataFrame(role_rows)
    if os.path.exists(role_fpath):
        old_df = pd.read_csv(role_fpath)
        old_df = old_df[old_df['group'] != group]
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(role_fpath, index=False)

    print(f'\nResults saved to: {results_dir}')
    print(f'Run with --mode {args.mode} --group all once all groups are done.')


if __name__ == '__main__':
    main()
