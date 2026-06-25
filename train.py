"""
train.py — Train baseline, ablation, and isolation SRL models.

Modes:
    baseline  — Train on full dependency tree.
    ablation  — Train with one dependency relation group masked out.
    isolation — Train with ONLY one dependency relation group; all others masked.
                Use --group no_edge for the zero-edge self-loop baseline.

Usage:
    python train.py --mode baseline
    python train.py --mode ablation  --group core_args
    python train.py --mode isolation --group core_args
    python train.py --mode isolation --group no_edge
"""

import os, sys, json, time, argparse
from collections import Counter

import torch
import torch.nn as nn

from shared import read_conll09, build_vocabs, load_glove, make_loaders, SRLModel


DEPREL_GROUPS = {
    'core_args': {
        'labels': ['SBJ', 'OBJ', 'OPRD', 'PRD', 'PUT', 'DTV', 'LGS'],
        'description': 'Core predicate-argument relations',
    },
    'clausal': {
        'labels': ['VC', 'SUB', 'EXTR', 'IM'],
        'description': 'Clausal and verb-chain relations',
    },
    'adjuncts': {
        'labels': ['ADV', 'TMP', 'LOC', 'DIR', 'MNR', 'EXT', 'BNF', 'PRP',
                   'PRD-PRP', 'PRD-TMP', 'LOC-OPRD', 'LOC-PRD', 'MNR-TMP'],
        'description': 'Adjunct and modifier relations (AM-* roles)',
    },
    'noun_internal': {
        'labels': ['NMOD', 'AMOD', 'APPO', 'HMOD', 'NAME', 'TITLE',
                   'POSTHON', 'SUFFIX', 'PMOD'],
        'description': 'Noun-phrase internal relations',
    },
    'coordination': {
        'labels': ['COORD', 'CONJ'],
        'description': 'Coordination and conjunction relations',
    },
    'functional': {
        'labels': ['DEP', 'P', 'HYPH', 'PRT', 'PRN', 'ADV-GAP', 'VOC'],
        'description': 'Functional, punctuation, and vocative relations',
    },
    'gap': {
        'labels': ['GAP-SBJ', 'GAP-OBJ', 'GAP-LOC', 'GAP-TMP', 'GAP-NMOD',
                   'GAP-OPRD', 'GAP-PMOD', 'GAP-PRD', 'GAP-VC', 'GAP-LGS',
                   'DEP-GAP', 'DIR-GAP', 'EXT-GAP'],
        'description': 'Gap and long-distance dependency relations',
    },
    'no_edge': {
        'labels': [],
        'description': 'No dependency edges — BiLSTM only (zero-edge baseline)',
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='Train SRL model (baseline, ablation, or isolation)')
    p.add_argument('--mode', required=True, choices=['baseline', 'ablation', 'isolation'])
    p.add_argument('--group', choices=list(DEPREL_GROUPS.keys()),
                   help='Dependency group (required for ablation/isolation; no_edge only for isolation)')
    p.add_argument('--train_file',
                   default='data/conll2009/CoNLL2009-ST-English-train.txt')
    p.add_argument('--dev_file',
                   default='data/conll2009/CoNLL2009-ST-English-development.txt')
    p.add_argument('--glove_file',
                   default='data/glove.2024.wikigiga.100d.txt')
    p.add_argument('--baseline_dir', default='checkpoints',
                   help='Output dir for baseline checkpoint')
    p.add_argument('--ablation_dir', default='checkpoints_ablation',
                   help='Output dir for ablation checkpoints')
    p.add_argument('--isolation_dir', default='checkpoints_isolation',
                   help='Output dir for isolation checkpoints')
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=0.001)
    p.add_argument('--lstm_hidden', type=int,   default=512)
    p.add_argument('--lstm_layers', type=int,   default=4)
    p.add_argument('--gcn_layers',  type=int,   default=1)
    p.add_argument('--dropout',     type=float, default=0.3)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--force',       action='store_true',
                   help='Overwrite existing checkpoint and retrain')
    return p.parse_args()


def run_epoch(model, loader, opt, loss_fn, rv, device, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    null_id    = rv.get('NULL', 1)
    tp = fp = fn = 0

    freq_arr = getattr(run_epoch, '_freq_arr', None) if training else None

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            cw_cpu = batch.pop('cw')
            batch  = {k: v.to(device) for k, v in batch.items()}
            cw     = cw_cpu.to(device) if cw_cpu is not None else None

            if training and freq_arr is not None:
                wids  = batch['word_ids']
                safe  = wids.clamp(0, len(freq_arr)-1).cpu()
                probs = 0.25 / (freq_arr[safe].to(device) + 0.25)
                batch['word_ids'] = wids.masked_fill(torch.bernoulli(probs).bool(), 1)

            logits = model(
                batch['word_ids'], batch['lemma_ids'], batch['pos_ids'],
                batch['pred_flag'], batch['pred_idxs'],
                batch['heads'],     batch['deprel_ids'],
                batch['mask'].float(), cw)

            B, L, R = logits.shape
            loss = loss_fn(logits.view(B*L, R), batch['role_ids'].view(B*L))

            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item()
            preds = logits.argmax(-1)
            gold  = batch['role_ids']
            for b in range(B):
                for i in range(batch['lengths'][b].item()):
                    p_i, g_i = preds[b,i].item(), gold[b,i].item()
                    if g_i not in (0, null_id):
                        tp += (p_i == g_i)
                        fn += (p_i != g_i)
                    if p_i not in (0, null_id) and p_i != g_i:
                        fp += 1

    pr = tp / (tp + fp + 1e-9)
    re = tp / (tp + fn + 1e-9)
    f1 = 2 * pr * re / (pr + re + 1e-9)
    return total_loss / len(loader), pr, re, f1


def main():
    args = parse_args()

    if args.mode in ('ablation', 'isolation') and not args.group:
        print(f'Error: --group is required when --mode {args.mode}')
        sys.exit(1)
    if args.mode == 'ablation' and args.group == 'no_edge':
        print('Error: --group no_edge is only valid for --mode isolation')
        sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine output paths and edge masking mode
    if args.mode == 'baseline':
        output_dir  = args.baseline_dir
        ckpt_name   = 'best_baseline.pt'
        log_name    = 'train_log_baseline.txt'
        run_label   = 'baseline'
        group_info  = None
        ablated_ids = None
        isolated_ids = None
    elif args.mode == 'ablation':
        output_dir  = args.ablation_dir
        ckpt_name   = f'best_ablation_{args.group}.pt'
        log_name    = f'train_log_ablation_{args.group}.txt'
        run_label   = f'ablation/{args.group}'
        group_info  = DEPREL_GROUPS[args.group]
        ablated_ids = None   # resolved below
        isolated_ids = None
    else:  # isolation
        output_dir  = args.isolation_dir
        ckpt_name   = f'best_isolation_{args.group}.pt'
        log_name    = f'train_log_isolation_{args.group}.txt'
        run_label   = f'isolation/{args.group}'
        group_info  = DEPREL_GROUPS[args.group]
        ablated_ids = None
        isolated_ids = None  # resolved below

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, ckpt_name)

    if os.path.exists(ckpt_path) and not args.force:
        print(f'Checkpoint already exists: {ckpt_path}')
        print('Use --force to overwrite.')
        sys.exit(0)

    print(f'=== Training: {run_label} | device={device} ===')
    if group_info:
        print(f'    {group_info["description"]}')
        print(f'    Labels: {group_info["labels"]}')

    # ── Data ──────────────────────────────────────────────────────────────────
    print('Loading data...')
    train_sents = read_conll09(args.train_file)
    dev_sents   = read_conll09(args.dev_file)
    wv, lv, pv, dv, rv = build_vocabs(train_sents)

    vocab_path = os.path.join(output_dir, 'vocabs.json')
    with open(vocab_path, 'w') as f:
        json.dump(dict(word=wv, lemma=lv, pos=pv, deprel=dv, role=rv), f)

    # Resolve label strings -> deprel IDs based on mode
    ablated_ids  = None
    isolated_ids = None
    if group_info:
        group_ids = set()
        for label in group_info['labels']:
            if label in dv:
                group_ids.add(dv[label])
            else:
                print(f'  Warning: "{label}" not in deprel vocab, skipping.')

        total_edges  = sum(len(s.tokens) for s in train_sents)
        group_edges  = sum(1 for s in train_sents for t in s.tokens
                           if t.deprel in group_info['labels'])
        pct = 100 * group_edges / total_edges

        if args.mode == 'ablation':
            ablated_ids = group_ids
            print(f'  Edges masked (ablation): {group_edges}/{total_edges} ({pct:.1f}%)')
        else:  # isolation
            isolated_ids = group_ids
            print(f'  Edges kept (isolation): {group_edges}/{total_edges} ({pct:.1f}%)'
                  f' — all others masked')

    # ── GloVe ─────────────────────────────────────────────────────────────────
    pretrained = None
    if args.glove_file and os.path.exists(args.glove_file):
        print('Loading GloVe embeddings...')
        pretrained = load_glove(args.glove_file, wv, dim=100)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    _, _, train_ld, dev_ld = make_loaders(
        train_sents, dev_sents, wv, lv, pv, dv, rv,
        ablated_ids=ablated_ids,
        isolated_ids=isolated_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # Word dropout frequency table
    freq     = Counter(t.form.lower() for s in train_sents for t in s.tokens)
    freq_arr = torch.zeros(len(wv))
    for w, i in wv.items():
        freq_arr[i] = freq.get(w, 0)
    run_epoch._freq_arr = freq_arr

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SRLModel(
        len(wv), len(lv), len(pv), len(dv), len(rv),
        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
        gcn_layers=args.gcn_layers,   dropout=args.dropout,
        pretrained=pretrained,
    ).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    best_f1 = 0.0

    # ── Training loop ──────────────────────────────────────────────────────────
    log_path = os.path.join(output_dir, log_name)
    print(f'Training — epochs: {args.epochs} | log: {log_path}\n')

    with open(log_path, 'w') as log_f:
        header = 'epoch\ttrain_loss\ttrain_f1\tdev_p\tdev_r\tdev_f1\tseconds'
        log_f.write(header + '\n')
        print(header)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, _, _, tr_f1 = run_epoch(
                model, train_ld, opt, loss_fn, rv, device, training=True)
            _, p, r, f1 = run_epoch(
                model, dev_ld,   opt, loss_fn, rv, device, training=False)
            elapsed = time.time() - t0

            row = f'{epoch}\t{tr_loss:.4f}\t{tr_f1:.4f}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{elapsed:.1f}'
            log_f.write(row + '\n')
            log_f.flush()
            print(row)

            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    'model':          model.state_dict(),
                    'vocabs':         dict(word=wv, lemma=lv, pos=pv, deprel=dv, role=rv),
                    'mode':           args.mode,
                    'group':          args.group if args.mode != 'baseline' else None,
                    'group_labels':   group_info['labels'] if group_info else [],
                    'ablated_ids':    list(ablated_ids)  if ablated_ids  else [],
                    'isolated_ids':   list(isolated_ids) if isolated_ids else [],
                    'hparams':        dict(lstm_hidden=args.lstm_hidden,
                                          lstm_layers=args.lstm_layers,
                                          gcn_layers=args.gcn_layers,
                                          dropout=args.dropout),
                }, ckpt_path)
                print(f'  -> Saved best model (F1={f1:.4f})')

    print(f'\nBest dev F1: {best_f1:.4f}')
    print(f'Checkpoint:  {ckpt_path}')


if __name__ == '__main__':
    main()
