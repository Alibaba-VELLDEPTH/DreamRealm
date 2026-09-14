"""Recover whole families from R4 pair records without sampling image members."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def split_query(query):
    if not isinstance(query, list):
        raise ValueError('Expected one-image multimodal query')
    images = [x['image'] for x in query if x['type'] == 'image']
    rest = [x for x in query if x['type'] != 'image']
    if len(images) != 1 or any(x['type'] != 'text' for x in rest):
        raise ValueError('Expected one image plus text only')
    return images[0], '\n'.join(x['text'] for x in rest)


def canonical_answer(s):
    return ' '.join(s.split())


def validate_family(f):
    required = ('family_id', 'question', 'original_image', 'original_answer', 'edited_pairs', 'original_negative_answers')
    if any(k not in f for k in required):
        raise ValueError('Missing family field')
    if any(not isinstance(f[k], str) or not f[k].strip() for k in required[:4]):
        raise ValueError('Empty family identity, prompt, image or answer')
    images = [f['original_image']] + [p['image'] for p in f['edited_pairs']]
    answers = [f['original_answer']] + [p['answer'] for p in f['edited_pairs']]
    if len(images) != len(set(images)):
        raise ValueError('duplicate_images')
    if any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError('Empty answer')
    if len({canonical_answer(a) for a in answers}) != len(answers):
        raise ValueError('duplicate_answers')
    neg = f['original_negative_answers']
    if any(not isinstance(a, str) or not a.strip() for a in neg):
        raise ValueError('Empty negative')
    if canonical_answer(f['original_answer']) in {canonical_answer(a) for a in neg}:
        raise ValueError('Negative equals correct answer')
    if len({canonical_answer(a) for a in neg}) != len(neg):
        raise ValueError('Duplicate negatives')
    if f['edited_pairs'] and neg:
        raise ValueError('This protocol excludes unpaired answer columns for edited families')
    return f


def convert(rows, negatives=None):
    groups, singles = {}, {}
    for d in rows:
        if d['loss_type'] == 'sft':
            image, q = split_query(d['query'])
            fid = d['metadata']['pair_group']
            f = dict(family_id=fid, question=q, original_image=image, original_answer=d['response'],
                     edited_pairs=[], original_negative_answers=list(d.get('original_negative_answers', [])),
                     source_ids=[d['id']], source='r4_singleton_sft')
            if fid in singles:
                raise ValueError(f'Duplicate SFT family {fid}')
            singles[fid] = f
        elif d['loss_type'] == 'dpo':
            if d.get('dpo_objective_route') != 'cycle_pair':
                raise ValueError('Only cycle_pair DPO is convertible')
            if d['chosen'] != d['aux_dpo_rejected'] or d['rejected'] != d['aux_dpo_chosen']:
                raise ValueError('Non-reversed pair answers')
            cp = d['cycle_pair']; fid = d['pair_group']
            g = groups.setdefault(fid, dict(members={}, size=cp['group_size'], source_ids=[]))
            if g['size'] != cp['group_size']:
                raise ValueError('Inconsistent declared family size')
            g['source_ids'].append(d['id'])
            for side, query, answer in [('left', d['query'], d['chosen']), ('right', d['aux_dpo_query'], d['aux_dpo_chosen'])]:
                image, q = split_query(query)
                position = cp[side+'_position']
                member = dict(image=image, answer=answer, question=q, source_id=cp[side+'_source_id'])
                if position in g['members'] and g['members'][position] != member:
                    raise ValueError(f'Conflicting member {fid}/{position}')
                g['members'][position] = member
        else:
            raise ValueError('Unknown R4 loss_type')
    if set(groups) & set(singles):
        raise ValueError('Family occurs in both branches')
    accepted, rejected = [], []
    for fid, g in groups.items():
        members = [g['members'][k] for k in sorted(g['members'])]
        if len(members) != g['size'] or len({p['question'] for p in members}) != 1:
            raise ValueError(f'Incomplete family or question mismatch: {fid}')
        originals = [p for p in members if p['source_id'].startswith('orig_')]
        if len(originals) != 1:
            raise ValueError(f'Expected one original: {fid}')
        original = originals[0]
        f = dict(family_id=fid, question=original['question'], original_image=original['image'],
                 original_answer=original['answer'], edited_pairs=[{k:p[k] for k in ('image','answer','source_id')} for p in members if p is not original],
                 original_negative_answers=[], source_ids=g['source_ids'], source='r4_recovered_whole_family')
        try:
            validate_family(f)
        except ValueError as e:
            rejected.append(dict(reason=str(e), family=f))
        else:
            accepted.append(f)
    negatives = negatives or {}
    if set(negatives) - set(singles):
        raise ValueError('Negative file includes unknown/non-singleton family IDs')
    for fid, f in singles.items():
        if fid in negatives:
            f['original_negative_answers'] = negatives[fid]
        accepted.append(validate_family(f))
    return accepted, rejected


def write_jsonl(path, rows):
    with open(path, 'x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--negative-file', help='JSONL: family_id, original_negative_answers; singleton families only')
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    negatives = {}
    if a.negative_file:
        for row in read_jsonl(a.negative_file):
            if row['family_id'] in negatives: raise ValueError('Duplicate negative family ID')
            negatives[row['family_id']] = row['original_negative_answers']
    accepted, rejected = convert(read_jsonl(a.input), negatives)
    write_jsonl(out/'families.jsonl', accepted); write_jsonl(out/'quarantined_families.jsonl', rejected)
    stats = dict(source_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(), total_families=len(accepted)+len(rejected),
                 accepted=len(accepted), quarantined=len(rejected), reasons=dict(collections.Counter(r['reason'] for r in rejected)),
                 edited_family_sizes=dict(collections.Counter(len(f['edited_pairs'])+1 for f in accepted if f['edited_pairs'])),
                 noedit_negative_counts=dict(collections.Counter(len(f['original_negative_answers']) for f in accepted if not f['edited_pairs'])))
    (out/'preparation_report.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2)+'\n')
    # Branch verification fixtures. Plain negatives are genuine mismatches from an
    # edited family, with edits intentionally withheld ONLY in this diagnostic file.
    picks = [next(f for f in accepted if len(f['edited_pairs'])==n) for n in (1,2)]
    plain = dict(next(f for f in accepted if len(f['edited_pairs'])>=2))
    plain['original_negative_answers'] = [v['answer'] for v in plain['edited_pairs'][:2]]
    plain['edited_pairs'] = []; plain['family_id'] += '__diagnostic_withheld_edits'
    plain['source'] = 'diagnostic_only_edits_withheld_not_production_singleton'
    picks += [plain, next(f for f in accepted if not f['edited_pairs'])]
    write_jsonl(out/'smoke_families.jsonl', picks)
    print(json.dumps(stats, ensure_ascii=False))

if __name__ == '__main__': main()
