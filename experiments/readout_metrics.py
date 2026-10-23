"""Readout quality at identical prefixes, not speculative acceptance or speed."""
from collections import defaultdict


def select_positions(length, limit):
    if length < 0 or limit < 0:
        raise ValueError('length and limit must be nonnegative')
    count = min(length, limit)
    if count < 2:
        return [0] if count else []
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def measure_readouts(early, final, references):
    """Accept [positions,vocab] logits; return compact CPU records.

    Rank uses 1 + number of strictly greater logits (ties share a rank).
    Top-k membership uses the actual torch.topk indices, not a rank shortcut.
    """
    early, final = early.float(), final.float()
    if early.shape != final.shape or early.ndim != 2 or len(references) != early.shape[0]:
        raise ValueError('Expected matching [positions,vocab] logits and reference IDs')
    if not early.isfinite().all().item() or not final.isfinite().all().item():
        raise ValueError('Non-finite readout logits')
    p, q = early.softmax(-1), final.softmax(-1)
    k = min(10, early.shape[-1])
    values, indices = p.topk(k, dim=-1)
    final_ids = q.argmax(-1)
    ranks = []
    for row, token in zip(early, final_ids):
        ranks.append(int((row > row[token]).sum().item()) + 1)
    rows = []
    for i, ref in enumerate(references):
        ids = indices[i].tolist()
        target = int(final_ids[i].item())
        rows.append({
            'reference_id': int(ref), 'early_id': ids[0], 'final_id': target,
            'early_confidence': float(values[i, 0].item()),
            'early_probability_margin': float((values[i, 0]-values[i, 1]).item()) if k > 1 else 1.0,
            'final_confidence': float(q[i, target].item()),
            'final_probability_in_early': float(p[i, target].item()),
            'final_rank_in_early': ranks[i],
            'final_in_early_top5': target in ids[:5],
            'final_in_early_top10': target in ids,
            'early_top10_ids': ids,
        })
    return rows


def summarize_readouts(rows):
    groups = defaultdict(list)
    for row in rows:
        stage = str(row['stage'])
        scopes = ['all', 'eos' if row['is_eos'] else 'content']
        if row['position'] == 0:
            scopes.append('first')
        for scope in scopes:
            groups[(stage, scope)].append(row)
    result = {}
    for (stage, scope), selected in sorted(groups.items()):
        n = len(selected)
        result.setdefault(stage, {})[scope] = {
            'samples': len({r['id'] for r in selected}), 'compared_tokens': n,
            'top1_agreement': sum(r['early_id'] == r['final_id'] for r in selected)/n,
            'top5_hit_rate': sum(r['final_in_early_top5'] for r in selected)/n,
            'top10_hit_rate': sum(r['final_in_early_top10'] for r in selected)/n,
            'mean_final_rank_in_early': sum(r['final_rank_in_early'] for r in selected)/n,
            'mean_early_confidence': sum(r['early_confidence'] for r in selected)/n,
            'mean_final_confidence': sum(r['final_confidence'] for r in selected)/n,
            'final_reference_mismatches': sum(r['final_id'] != r['reference_id'] for r in selected),
        }
    return result
