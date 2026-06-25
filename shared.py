"""
shared.py — Common data structures, readers, dataset, and model definitions.
Imported by train.py and evaluate.py.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networkx as nx


# ── CoNLL-2009 Data Structures ────────────────────────────────────────────────

@dataclass
class Token:
    id: int
    form: str
    lemma: str
    pos: str
    head: int
    deprel: str
    is_predicate: bool
    pred_sense: Optional[str]
    arg_roles: List[str] = field(default_factory=list)


@dataclass
class Sentence:
    tokens: List[Token]

    @property
    def heads_0based(self):
        """0-based heads; root token points to itself."""
        result = []
        for i, t in enumerate(self.tokens):
            h = t.head - 1
            result.append(i if h < 0 else h)
        return result

    @property
    def predicate_indices(self):
        return [i for i, t in enumerate(self.tokens) if t.is_predicate]

    def get_roles(self, pred_col: int) -> Dict[int, str]:
        return {i: t.arg_roles[pred_col]
                for i, t in enumerate(self.tokens)
                if pred_col < len(t.arg_roles) and t.arg_roles[pred_col] != '_'}


# ── CoNLL-2009 Reader ─────────────────────────────────────────────────────────

def read_conll09(filepath: str) -> List[Sentence]:
    sentences, current = [], []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '':
                if current:
                    sentences.append(Sentence(tokens=current))
                    current = []
                continue
            if line.startswith('#'):
                continue
            cols = line.split('\t')
            if len(cols) < 14:
                continue
            try:
                head = int(cols[9]) if cols[9] not in ('_', '') else 0
            except ValueError:
                head = 0
            current.append(Token(
                id=int(cols[0]), form=cols[1], lemma=cols[2],
                pos=cols[5], head=head,
                deprel=cols[11] if cols[11] != '_' else 'dep',
                is_predicate=(cols[12] == 'Y'),
                pred_sense=cols[13] if cols[13] != '_' else None,
                arg_roles=cols[14:],
            ))
    if current:
        sentences.append(Sentence(tokens=current))
    return sentences


def build_vocabs(sentences):
    def vocab(items, specials):
        v = {s: i for i, s in enumerate(specials)}
        for x in sorted(set(items)):
            if x not in v:
                v[x] = len(v)
        return v

    words, lemmas, pos_tags, deprels, roles = [], [], [], [], []
    for s in sentences:
        for t in s.tokens:
            words.append(t.form.lower())
            lemmas.append(t.lemma.lower())
            pos_tags.append(t.pos)
            deprels.append(t.deprel)
            roles.extend([r for r in t.arg_roles if r != '_'])
    return (vocab(words,    ('<PAD>', '<UNK>')),
            vocab(lemmas,   ('<PAD>', '<UNK>')),
            vocab(pos_tags, ('<PAD>', '<UNK>')),
            vocab(deprels,  ('<PAD>', '<UNK>')),
            vocab(roles,    ('<PAD>', 'NULL')))


# ── GloVe Loader ──────────────────────────────────────────────────────────────

def load_glove(filepath, vocab, dim=100):
    rng    = np.random.default_rng(42)
    matrix = rng.normal(scale=0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0  # PAD = zero vector
    found = 0
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) != dim + 1:
                continue
            word = parts[0].lower()
            if word in vocab:
                matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f'  GloVe: {found}/{len(vocab)} vectors loaded')
    return matrix


# ── Centrality & DLT Metrics ──────────────────────────────────────────────────

def build_graph(heads):
    G = nx.DiGraph()
    n = len(heads)
    G.add_nodes_from(range(n))
    for dep, head in enumerate(heads):
        if head != dep and 0 <= head < n:
            G.add_edge(head, dep)
            G.add_edge(dep, head)
    return G


def node_centrality(heads, ctype):
    """
    Compute per-node centrality scores.
    ctype suffix '_inv' triggers inverted weighting (1/score),
    giving higher weight to low-centrality (peripheral) nodes.
    """
    invert = ctype.endswith('_inv')
    base   = ctype[:-4] if invert else ctype

    G = build_graph(heads)
    n = len(heads)
    if base == 'degree':
        d = nx.degree_centrality(G)
    elif base == 'betweenness':
        d = nx.betweenness_centrality(G, normalized=True)
    elif base == 'closeness':
        d = nx.closeness_centrality(G)
    else:
        raise ValueError(f'Unknown centrality type: {ctype}')

    scores = np.array([d.get(i, 0.0) for i in range(n)], dtype=np.float32) + 1e-6
    if invert:
        scores = 1.0 / scores
    return scores / scores.sum()


def centrality_matrix(heads, ctype):
    """(n,n) weight matrix. w[i,j] = weight of arc j->i."""
    n      = len(heads)
    scores = node_centrality(heads, ctype)
    w      = np.zeros((n, n), dtype=np.float32)
    for dep, head in enumerate(heads):
        if 0 <= head < n and head != dep:
            w[dep,  head] = scores[head]
            w[head, dep]  = scores[dep]
        w[dep, dep] = scores[dep]
    return w


def batch_centrality(heads_batch, ctype, max_len):
    B   = len(heads_batch)
    out = torch.zeros(B, max_len, max_len)
    for b, heads in enumerate(heads_batch):
        n = len(heads)
        out[b, :n, :n] = torch.from_numpy(centrality_matrix(heads, ctype))
    return out


def dep_distance(heads):
    dists = [abs(i-h) for i, h in enumerate(heads) if h != i and 0 <= h < len(heads)]
    return float(np.mean(dists)) if dists else 0.0


# ── Dataset ───────────────────────────────────────────────────────────────────

NP_POS      = {'NN', 'NNS', 'NNP', 'NNPS'}
VP_POS      = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
NEW_REF_POS = NP_POS | VP_POS


def count_ndr(tokens, pred_idx, arg_idx):
    """Count new discourse referents intervening between predicate and argument (Gibson 1998)."""
    lo = min(pred_idx, arg_idx) + 1
    hi = max(pred_idx, arg_idx)
    return sum(1 for k in range(lo, hi) if tokens[k].pos in NEW_REF_POS)


class SRLDataset(Dataset):
    def __init__(self, sentences, wv, lv, pv, dv, rv,
                 centrality='none', ablated_ids: set = None,
                 isolated_ids: set = None):
        self.wv, self.lv, self.pv = wv, lv, pv
        self.dv, self.rv = dv, rv
        self.centrality  = centrality
        self.ablated_ids = ablated_ids  or set()
        self.isolated_ids = isolated_ids  # if set, keep ONLY these deprel IDs
        self.examples    = []
        self.sent_ndr    = {}
        self.sent_cw     = {}

        mode_str = ('isolation' if isolated_ids
                    else f'ablation({sorted(ablated_ids)})' if ablated_ids
                    else 'none')
        print(f'  Building dataset (centrality={centrality}, edge_mode={mode_str})...')
        for sent_idx, sent in enumerate(sentences):
            all_ndr = []
            for col, pred_idx in enumerate(sent.predicate_indices):
                roles = sent.get_roles(col)
                for ai in roles.keys():
                    all_ndr.append(count_ndr(sent.tokens, pred_idx, ai))
            self.sent_ndr[sent_idx] = float(np.mean(all_ndr)) if all_ndr else 0.0

            if centrality != 'none':
                heads = sent.heads_0based
                self.sent_cw[sent_idx] = torch.from_numpy(
                    centrality_matrix(heads, centrality))  # tensor (n, n)

            for col, pred_idx in enumerate(sent.predicate_indices):
                self.examples.append((sent, pred_idx, sent.get_roles(col), sent_idx))

        print(f'  {len(self.examples)} predicate-argument examples ready.')

    def __len__(self):
        return len(self.examples)

    def _id(self, s, vocab, lowercase=True):
        key = s.lower() if lowercase else s
        return vocab.get(key, vocab.get('<UNK>', 1))

    def __getitem__(self, idx):
        sent, pred_idx, roles, sent_idx = self.examples[idx]
        n    = len(sent.tokens)
        wids = torch.tensor([self._id(t.form,   self.wv) for t in sent.tokens], dtype=torch.long)
        lids = torch.tensor([self._id(t.lemma,  self.lv) for t in sent.tokens], dtype=torch.long)
        pids = torch.tensor([self._id(t.pos,    self.pv) for t in sent.tokens], dtype=torch.long)
        dids = torch.tensor([self._id(t.deprel, self.dv, lowercase=False) for t in sent.tokens], dtype=torch.long)
        flag = torch.zeros(n, dtype=torch.long)
        flag[pred_idx] = 1
        heads   = torch.tensor(sent.heads_0based, dtype=torch.long)

        # Ablation: redirect ablated edges to self-loop
        if self.ablated_ids:
            for i in range(n):
                if dids[i].item() in self.ablated_ids:
                    heads[i] = i
        # Isolation: redirect ALL edges except the kept group to self-loop
        if self.isolated_ids is not None:
            for i in range(n):
                if dids[i].item() not in self.isolated_ids:
                    heads[i] = i
        null_id = self.rv.get('NULL', 1)
        rids    = torch.full((n,), null_id, dtype=torch.long)
        for tok_idx, role_str in roles.items():
            rids[tok_idx] = self.rv.get(role_str, null_id)
        ndr_val = self.sent_ndr[sent_idx]
        cw = self.sent_cw.get(sent_idx)  # None for baseline, tensor (n,n) otherwise

        return dict(word_ids=wids, lemma_ids=lids, pos_ids=pids,
                    deprel_ids=dids, pred_flag=flag, pred_idx=pred_idx,
                    heads=heads, role_ids=rids, length=n,
                    ndr=ndr_val, sent_idx=sent_idx, cw=cw)


def collate_fn(batch):
    max_len = max(ex['length'] for ex in batch)
    B = len(batch)

    def pad(key, val=0):
        out = torch.full((B, max_len), val, dtype=torch.long)
        for i, ex in enumerate(batch):
            out[i, :ex['length']] = ex[key]
        return out

    lengths = torch.tensor([ex['length'] for ex in batch], dtype=torch.long)

    # Pad pre-computed centrality matrices if present
    cw_batch = None
    if batch[0]['cw'] is not None:
        cw_batch = torch.zeros(B, max_len, max_len)
        for i, ex in enumerate(batch):
            n = ex['length']
            cw_batch[i, :n, :n] = ex['cw']

    return dict(
        word_ids=pad('word_ids'), lemma_ids=pad('lemma_ids'),
        pos_ids=pad('pos_ids'),   deprel_ids=pad('deprel_ids'),
        pred_flag=pad('pred_flag'), heads=pad('heads'),
        role_ids=pad('role_ids'),
        pred_idxs=torch.tensor([ex['pred_idx']  for ex in batch], dtype=torch.long),
        sent_idxs=torch.tensor([ex['sent_idx']  for ex in batch], dtype=torch.long),
        lengths=lengths,
        mask=(torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)),
        ndrs=torch.tensor([ex['ndr'] for ex in batch], dtype=torch.float),
        cw=cw_batch,  # (B, max_len, max_len) or None
    )


def make_loaders(train_sents, dev_sents, wv, lv, pv, dv, rv,
                 centrality='none', ablated_ids: set = None,
                 isolated_ids: set = None,
                 batch_size=64, num_workers=4):
    train_ds = SRLDataset(train_sents, wv, lv, pv, dv, rv, centrality, ablated_ids, isolated_ids)
    dev_ds   = SRLDataset(dev_sents,   wv, lv, pv, dv, rv, centrality, ablated_ids, isolated_ids)
    train_ld = DataLoader(train_ds, batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=num_workers,
                          pin_memory=True)
    dev_ld   = DataLoader(dev_ds,   batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=num_workers,
                          pin_memory=True)
    return train_ds, dev_ds, train_ld, dev_ld


# ── Model ─────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    One syntactic GCN layer (Marcheggiani & Titov 2017).
    Three directions: along arc, opposite arc, self-loop.
    Supports optional centrality weight matrix (cw).
    """
    def __init__(self, dim, deprel_vocab_size, dropout):
        super().__init__()
        self.W    = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(3)])
        self.gv   = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(3)])
        self.b_l  = nn.Embedding(deprel_vocab_size, dim, padding_idx=0)
        self.b_lg = nn.Embedding(deprel_vocab_size, 1,   padding_idx=0)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, heads, deprel_ids, mask, cw=None):
        B, L, D = h.shape
        out = torch.zeros_like(h)

        # Self-loop
        sg  = torch.sigmoid((h * self.gv[2]).sum(-1, keepdim=True) + self.b_lg(deprel_ids))
        out = out + sg * (self.W[2](h) + self.b_l(deprel_ids))

        # Along arc: token i <- head[i]
        hi   = heads.clamp(0).unsqueeze(-1).expand(-1, -1, D)
        h_h  = h.gather(1, hi)
        ag   = torch.sigmoid((h_h * self.gv[0]).sum(-1, keepdim=True) + self.b_lg(deprel_ids))
        ac   = ag * (self.W[0](h_h) + self.b_l(deprel_ids))
        if cw is not None:
            ac = ac * cw.gather(2, heads.clamp(0).unsqueeze(-1))
        out  = out + ac

        # Opposite arc: scatter dep contributions to head positions
        og   = torch.sigmoid((h * self.gv[1]).sum(-1, keepdim=True) + self.b_lg(deprel_ids))
        oc   = og * (self.W[1](h) + self.b_l(deprel_ids))
        if cw is not None:
            oc = oc * cw.transpose(1, 2).gather(2, heads.clamp(0).unsqueeze(-1))
        out.scatter_add_(1, heads.clamp(0).unsqueeze(-1).expand_as(oc), oc)

        out = out * mask.unsqueeze(-1)
        return F.relu(self.drop(out))


class SRLModel(nn.Module):
    def __init__(self, word_vsz, lemma_vsz, pos_vsz, dep_vsz, role_vsz,
                 word_dim=100, lemma_dim=100, pos_dim=16, flag_dim=16,
                 lstm_hidden=512, lstm_layers=4, gcn_layers=1,
                 dropout=0.1, pretrained=None):
        super().__init__()
        self.word_emb  = nn.Embedding(word_vsz,  word_dim,  padding_idx=0)
        self.lemma_emb = nn.Embedding(lemma_vsz, lemma_dim, padding_idx=0)
        self.pos_emb   = nn.Embedding(pos_vsz,   pos_dim,   padding_idx=0)
        self.flag_emb  = nn.Embedding(2, flag_dim)
        if pretrained is not None:
            self.pre_emb = nn.Embedding.from_pretrained(
                torch.tensor(pretrained), freeze=True, padding_idx=0)
            in_dim = word_dim + lemma_dim + pos_dim + flag_dim + pretrained.shape[1]
        else:
            self.pre_emb = None
            in_dim = word_dim + lemma_dim + pos_dim + flag_dim

        self.bilstm = nn.LSTM(in_dim, lstm_hidden, lstm_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if lstm_layers > 1 else 0.0)
        hd = lstm_hidden * 2
        self.gcns = nn.ModuleList(
            [GCNLayer(hd, dep_vsz, dropout) for _ in range(gcn_layers)])

        self.role_lemma = nn.Embedding(lemma_vsz, 128, padding_idx=0)
        self.role_emb   = nn.Embedding(role_vsz,  128, padding_idx=0)
        self.role_U     = nn.Linear(256, hd * 2, bias=False)
        self.drop       = nn.Dropout(dropout)

    def forward(self, word_ids, lemma_ids, pos_ids, pred_flag,
                pred_idxs, heads, deprel_ids, mask, cw=None):
        parts = [self.word_emb(word_ids), self.lemma_emb(lemma_ids),
                 self.pos_emb(pos_ids),   self.flag_emb(pred_flag)]
        if self.pre_emb is not None:
            parts.append(self.pre_emb(word_ids))
        x = self.drop(torch.cat(parts, -1))

        lengths = mask.sum(1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        h, _    = self.bilstm(packed)
        h, _    = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h       = self.drop(h)

        for gcn in self.gcns:
            h = gcn(h, heads, deprel_ids, mask.float(), cw)

        B, L, D = h.shape
        h_pred  = h.gather(1, pred_idxs.view(B,1,1).expand(B,1,D)).squeeze(1)
        h_pair  = torch.cat([h, h_pred.unsqueeze(1).expand(B,L,D)], -1)

        pred_lemma = lemma_ids.gather(1, pred_idxs.unsqueeze(1)).squeeze(1)
        q_l = self.role_lemma(pred_lemma)
        q_r = self.role_emb.weight
        R   = q_r.shape[0]
        W   = F.relu(self.role_U(torch.cat([
            q_l.unsqueeze(1).expand(B, R, 128),
            q_r.unsqueeze(0).expand(B, R, 128)], -1)))
        return torch.bmm(h_pair, W.transpose(1, 2))


def load_model_from_checkpoint(path, device='cpu'):
    """Load a saved model checkpoint. Returns (model, vocabs, centrality)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'No checkpoint at {path}')
    ck  = torch.load(path, map_location=device, weights_only=False)
    vv  = ck['vocabs']
    hp  = ck['hparams']
    has_pretrained = 'pre_emb.weight' in ck['model']
    pre_weight = ck['model']['pre_emb.weight'].cpu().numpy() if has_pretrained else None
    m = SRLModel(len(vv['word']), len(vv['lemma']), len(vv['pos']),
                 len(vv['deprel']), len(vv['role']),
                 pretrained=pre_weight, **hp)
    m.load_state_dict(ck['model'])
    m.to(device)
    m.eval()
    return m, vv, ck.get('centrality', 'unknown')
