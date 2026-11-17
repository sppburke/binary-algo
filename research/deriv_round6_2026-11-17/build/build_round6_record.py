#!/usr/bin/env python3
"""Round-6 record builder: check the ranking against the clusters and write the plan and the matrix.

From the repository root:

    python3 -I research/deriv_round6_2026-11-17/build/build_round6_record.py research/deriv_round6_2026-11-17 <out dir> [--check]

Inputs: <record dir>/clusters.jsonl.gz (the clusters as the clustering agents wrote them) and, beside this script,
RANK_result.txt (the ranking agent's output, JSON lines), gaps.jsonl (the gap-finder's proposals) and primary.json (the
primary agent's dispositions: gap decisions, its own gap proposals, claim dispositions, the program decisions each
test's own specification needs, and notes on single tests). The stage gates and the Gates column of the decision table
are derived from the per-test lists, and what a stage needs through the stages it builds on from the lists of those
stages. The build stops if a list contradicts the test's needs field, or if a specification names a test of a stage
that its own stage does not build on, or a test that waits for a decision the naming test neither lists nor needs
through earlier stages, and the naming test carries no note. These checks are structural: no note is read.
Outputs (out dir): matrix.jsonl.gz (one line per cluster and gap, with its stage), plan.json and plan_tables.md. With the
record dir as out dir the three files are rewritten in place; the gzip stream carries no timestamp, so equal inputs
give equal bytes. --check prints the partition check and the counts and writes nothing.
"""
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

record, out = Path(sys.argv[1]), Path(sys.argv[2])
src = Path(__file__).resolve().parent
check_only = '--check' in sys.argv


def jl(text):
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith('{'):
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                print('unparsed line', ln[:80])
    return rows


clusters = {c['cid']: c for c in jl(gzip.open(record / 'clusters.jsonl.gz', 'rt').read())}
rank = jl((src / 'RANK_result.txt').read_text())
one = lambda k: next((r[k] for r in rank if k in r and 'stage' not in r), None)
stages = sorted([r for r in rank if 'stage' in r], key=lambda r: r['stage'])
merged, gapdec, first25 = one('merged') or [], one('gaps') or [], one('first_25') or []
not_run, owner, claims = one('not_run') or [], one('owner_decisions') or [], one('claims_to_verify') or []
disagree, counts, summary = one('disagreements') or [], one('counts') or {}, one('summary') or ''

where = {}                                           # cid -> list of placements
for s in stages:
    for i in s.get('ids', []):
        where.setdefault(i, []).append(('stage', s['stage']))
for m in merged:
    for i in m.get('absorb', []):
        where.setdefault(i, []).append(('absorbed', m['keep']))
for n in not_run:
    where.setdefault(n['id'], []).append(('not_run', n.get('why', '')))

test = {c for c, v in clusters.items() if v['verdict'] == 'TEST'}
rej = {c for c, v in clusters.items() if v['verdict'] != 'TEST'}
missing = sorted(test - set(where))
twice = sorted(c for c in test if len(where.get(c, [])) > 1)
rej_staged = sorted(c for c in rej if any(k == 'stage' for k, _ in where.get(c, [])))
rej_missing = sorted(rej - set(where))
unknown = sorted(i for i in where if i not in clusters and not i.startswith('R6-GAP'))
gaps_accepted = [g for g in gapdec if g.get('decision') == 'accept']
print(f'clusters {len(clusters)} | TEST {len(test)} | REJECTED {len(rej)} | stages {len(stages)} | first_25 {len(first25)}')
print(f'TEST placed: staged {sum(1 for c in test if where.get(c) and where[c][0][0] == "stage")}, '
      f'absorbed {sum(1 for c in test if where.get(c) and where[c][0][0] == "absorbed")}, '
      f'not run {sum(1 for c in test if where.get(c) and where[c][0][0] == "not_run")}')
print(f'missing {len(missing)} {missing[:12]} | placed twice {len(twice)} {twice[:12]} | rejected but staged {rej_staged[:8]} '
      f'| rejected not listed {len(rej_missing)} | unknown ids {unknown[:8]}')
print(f'gaps: {len(gapdec)} decided, {len(gaps_accepted)} accepted | owner decisions {len(owner)} | claims {len(claims)}')

gaps_in = {}
for r in jl((src / 'gaps.jsonl').read_text()):
    for g in r.get('gaps', []) if isinstance(r.get('gaps'), list) else []:
        if isinstance(g, dict) and g.get('id'):
            gaps_in[g['id']] = g
    if r.get('id', '').startswith('R6-GAP'):
        gaps_in[r['id']] = r
# The primary agent's dispositions replace the ranking agent's (empty) gap list: the ranking agent ran before the gap
# list existed.
primary = json.loads((src / 'primary.json').read_text())
for g in primary.get('own_gaps', []):
    gaps_in[g['id']] = g
gapdec = [{'id': g['id'], 'decision': g['decision'], 'stage_or_covered_by': g.get('stage') or g.get('covered_by', ''),
           'why': g.get('why', ''), 'narrowed_to': g.get('narrowed_to'), 'evidence': g.get('evidence'),
           'prior': g.get('prior')}
          for g in primary['gap_decisions']]
for g in primary['gap_decisions']:
    if g['decision'] == 'accept' and g.get('stage') is not None:
        where.setdefault(g['id'], []).append(('stage', g['stage']))
        for s in stages:
            if s['stage'] == g['stage'] and g['id'] not in s['ids']:
                s['ids'].append(g['id'])
gaps_accepted = [g for g in gapdec if g.get('decision') == 'accept']
print(f'primary.json: {len(gapdec)} gap decisions, {len(gaps_accepted)} accepted, {len(primary.get("own_gaps", []))} own gaps')
for g in gaps_in.values():                          # one schema for the gap-finder's and the primary agent's records
    g.setdefault('side', g.get('sign_or_size'))
    g.setdefault('fast_kill', g.get('fast_kill_falsifier'))
    g.setdefault('axis', g.get('kind'))

# Which open owner decisions of docs/DERIV_SOTA_PROGRAM.md each test's own specification needs (the primary agent's
# reading): the list of its stage unless primary.json names the test. The stage gates and the Gates column of the
# decision table are derived from these lists, so they cannot disagree with them. The ranking agent's can_start_now says
# only that a stage needs no new scope: no venue call, key, closed family or data source.
tdec = primary['test_decisions']
dmap = primary['decision_map']
row_notes = primary.get('row_notes', {})
staged_ids = {i for s in stages for i in s['ids']}
needs_of = {**{c: v.get('needs') for c, v in clusters.items()}, **{g: v.get('needs') for g, v in gaps_in.items()}}
stage_default = {s['stage']: sorted(tdec['by_stage'].get(str(s['stage']), [])) for s in stages}
dec = {i: sorted(tdec['by_test'].get(i, stage_default[s['stage']])) for s in stages for i in s['ids']}
code = lambda ids: ', '.join('`' + i + '`' for i in ids)
nums = lambda ds: ', '.join(str(n) for n in ds)
# What a stage needs through the stages it builds on (the ranking agent's depends_on_stages, followed through): the
# lists of those stages. An earlier stage counts at its stage list: the tests it holds back for a further decision are
# not prerequisites of later stages, and a specification that names one is caught by the rule on named tests below.
deps = {s['stage']: set(s.get('depends_on_stages') or []) for s in stages}


def all_deps(n):
    """Every stage that stage n builds on, directly or through another stage."""
    seen, todo = set(), list(deps[n])
    while todo:
        d = todo.pop()
        if d not in seen:
            seen.add(d)
            todo.extend(deps[d])
    return seen


through = {n: {x for d in all_deps(n) for x in stage_default[d]} for n in deps}


def spans(ns):
    """[1, 2, 3, 5, 6] -> '1-3, 5, 6'."""
    runs = []
    for n in ns:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return ', '.join(f'{r[0]}-{r[-1]}' if len(r) > 2 else nums(r) for r in runs)


for s in stages:
    other = {}                                       # decision list -> the stage's tests that differ from its default
    for i in s['ids']:
        if dec[i] != stage_default[s['stage']]:
            other.setdefault(tuple(dec[i]), []).append(i)
    s['program_decisions'] = sorted({n for i in s['ids'] for n in dec[i]})
    s['waits_for'] = '; '.join([nums(stage_default[s['stage']])] + [f'{code(ids)}: {nums(ds)}' for ds, ids in other.items()])
    s['upstream_decisions'] = sorted(through[s['stage']] - set(stage_default[s['stage']]))
for m in dmap:
    n = m['program_decision']
    full = [s['stage'] for s in stages if all(n in dec[i] for i in s['ids'])]
    part = [(s['stage'], [i for i in s['ids'] if n in dec[i]]) for s in stages if s['stage'] not in full]
    m['gates'] = '; '.join(([('stage ' if len(full) == 1 else 'stages ') + spans(full)] if full else [])
                           + [f'{code(ids)} (stage {st})' for st, ids in part if ids])
mapped = sorted(n for m in dmap for n in m['ranking_decisions'])
known = {m['program_decision'] for m in dmap}
bad_notes = sorted(i for i in row_notes if i not in staged_ids)
bad_named = sorted(i for i in tdec['by_test'] if i not in staged_ids or i not in row_notes
                   or dec[i] == next(stage_default[s['stage']] for s in stages if i in s['ids']))
bad_lists = sorted(i for i in staged_ids if not dec[i] or not set(dec[i]) <= known
                   or (needs_of.get(i) == 'reopen_closed_family' and 11 not in dec[i])
                   or (needs_of.get(i) == 'owner_data_decision' and not {9, 12, 13} & set(dec[i])))
# A specification that names another test, by its id or by the id of a lens proposal merged into it, leans on that
# test. If the named test sits in a stage that this test's stage does not build on (the ranking agent's
# depends_on_stages), the test must carry a note saying how it runs without it.
member_of = {m: c for c, v in clusters.items() for m in v.get('members') or []}
kept_as = {i: m['keep'] for m in merged for i in m.get('absorb', [])}
stage_of = {i: s['stage'] for s in stages for i in s['ids']}
builds_on = {s['stage']: {s['stage'], *(s.get('depends_on_stages') or [])} for s in stages}
rank_notes = {i: n for s in stages for i, n in (s.get('notes') or {}).items()}
narrowed = {g['id']: g.get('narrowed_to') for g in gapdec}


def named_test(token):
    """The staged test a token names: a test id, or a lens proposal id with or without its R6- prefix."""
    for t in (token, 'R6-' + token):
        t = member_of.get(t, t)
        t = kept_as.get(t, t)
        if t in staged_ids:
            return t
    return None


def names(i):
    """The other staged tests that the specification, kill rule, cost and ranking note of test i name."""
    r = clusters.get(i) or gaps_in.get(i) or {}
    parts = (r.get('spec'), r.get('fast_kill'), r.get('cost'), rank_notes.get(i), narrowed.get(i))
    text = ' '.join(str(v or '') for v in parts)
    return sorted({named_test(f) for f in re.findall(r'\b(?:R6-)?(?:GAP-\d+|[A-Z]\d?-\d+)\b', text)} - {None, i})


refs = [(i, t) for i in sorted(staged_ids) for t in names(i)]
far_refs = [(i, t) for i, t in refs if stage_of[t] not in builds_on[stage_of[i]]]
bad_refs = [f'{i} names {t}' for i, t in far_refs if i not in row_notes]
# The named test may also wait for a decision that the naming test neither lists nor needs through the stages it builds
# on, in whatever stage it sits. Then the naming test must carry a note saying which part of the named test it uses.
gap_refs = [(i, t, sorted(set(dec[t]) - set(dec[i]) - through[stage_of[i]])) for i, t in refs]
gap_refs = [(i, t, d) for i, t, d in gap_refs if d]
bad_gaps = [f'{i} names {t}' for i, t, d in gap_refs if i not in row_notes]
print(f'program decisions {[m["program_decision"] for m in dmap]} | ranking decisions mapped once each: '
      f'{mapped == list(range(1, len(owner) + 1))} | notes on tests that are not staged: {bad_notes}')
print(f'tests named in test_decisions without a note, unstaged or equal to their stage: {bad_named} | tests whose list is '
      f'empty, names an unknown decision or omits what their needs field requires: {bad_lists}')
print(f'specifications naming another test: {len(refs)}; naming a test of a stage theirs does not build on: '
      f'{[f"{i} names {t} (stage {stage_of[t]})" for i, t in far_refs]}; of those without a note: {bad_refs}')
print(f'naming a test that waits for a decision they neither list nor need through earlier stages: '
      f'{[f"{i} names {t} (decision {nums(d)})" for i, t, d in gap_refs]}; of those without a note: {bad_gaps}')
for s in stages:
    print(f"  stage {s['stage']:>2} in scope {str(s.get('can_start_now')):5} waits for {s['waits_for']:38} "
          f"through earlier stages {nums(s['upstream_decisions']) or 'none':5} "
          f"n={len(s.get('ids', [])):3} | {s.get('name', '')[:52]} | {str(s.get('budget', ''))[:70]}")
for m in dmap:
    print(f"  decision {m['program_decision']:>2} {m['name']:24} gates {m['gates']}")
ok = (not missing and not twice and not rej_staged and not unknown and not bad_notes and not bad_named and not bad_lists
      and not bad_refs and not bad_gaps and mapped == list(range(1, len(owner) + 1))
      and len(tdec['by_stage']) == len(stages))
print('PARTITION OK' if ok else 'PARTITION NOT OK')
if check_only or not ok:
    sys.exit(0 if ok else 1)

out.mkdir(parents=True, exist_ok=True)
# A staged test's upstream decisions: what its stage needs through earlier stages, less what the test lists itself.
upstream_of = lambda i: sorted(through[stage_of[i]] - set(dec[i])) if i in stage_of else None
sname = {s['stage']: s.get('name', '') for s in stages}
notes = {i: n for s in stages for i, n in (s.get('notes') or {}).items()}
rank25 = {x['id']: x['rank'] for x in first25}
lines = []
for cid, c in clusters.items():
    pl = where.get(cid, [('unplaced', '')])[0]
    lines.append({'id': cid, 'group': c.get('group', cid[0]), 'axis': c.get('axis'), 'title': c.get('title'), 'side': c.get('side'),
                  'needs': c.get('needs'), 'prior': c.get('prior'), 'tier_in_group': c.get('tier'), 'verdict': c.get('verdict'),
                  'members': c.get('members'), 'n_lenses': c.get('n_lenses'),
                  'disposition': pl[0], 'stage': pl[1] if pl[0] == 'stage' else None,
                  'stage_name': sname.get(pl[1]) if pl[0] == 'stage' else None, 'program_decisions': dec.get(cid),
                  'upstream_decisions': upstream_of(cid),
                  'absorbed_into': pl[1] if pl[0] == 'absorbed' else None, 'not_run_why': pl[1] if pl[0] == 'not_run' else None,
                  'first_25_rank': rank25.get(cid), 'rank_note': notes.get(cid), 'primary_agent_note': row_notes.get(cid),
                  'spec': c.get('spec'), 'fast_kill': c.get('fast_kill'), 'cost': c.get('cost'), 'needs_cache': c.get('needs_cache'),
                  'new_vs_existing': c.get('new_vs_existing'), 'duplicate_of': c.get('duplicate_of'), 'prior_why': c.get('prior_why')})
for g in gapdec:
    s0 = gaps_in.get(g['id'], {})
    pl = where.get(g['id'], [('declined' if g.get('decision') != 'accept' else 'unplaced', g.get('stage_or_covered_by', ''))])[0]
    lines.append({'id': g['id'], 'group': 'GAP', 'axis': s0.get('axis'), 'title': s0.get('title'), 'side': s0.get('side'),
                  'needs': s0.get('needs'), 'prior': g.get('prior') or s0.get('prior'), 'prior_as_proposed': s0.get('prior'),
                  'verdict': 'TEST' if g.get('decision') == 'accept' else 'DECLINED',
                  'disposition': pl[0], 'stage': pl[1] if pl[0] == 'stage' else None,
                  'stage_name': sname.get(pl[1]) if pl[0] == 'stage' else None, 'program_decisions': dec.get(g['id']),
                  'upstream_decisions': upstream_of(g['id']),
                  'not_run_why': g.get('why') if g.get('decision') != 'accept' else None,
                  'first_25_rank': rank25.get(g['id']), 'rank_note': g.get('narrowed_to') or notes.get(g['id']),
                  'primary_agent_note': row_notes.get(g['id']),
                  'primary_agent_why': g.get('why'), 'primary_agent_evidence': g.get('evidence'),
                  'mechanism': s0.get('mechanism'), 'nearest_existing': s0.get('nearest_existing'),
                  'prior_why': s0.get('prior_why'),
                  'spec': s0.get('spec'), 'fast_kill': s0.get('fast_kill'), 'cost': s0.get('cost')})
with open(out / 'matrix.jsonl.gz', 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as fh:
    fh.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in lines).encode())
claim_disp = primary.get('claim_dispositions', [])
counts = dict(counts, gaps_accepted=len(gaps_accepted),
              tests_planned_with_gaps=sum(len(s.get('ids', [])) for s in stages))
(out / 'plan.json').write_text(json.dumps({'stages': stages, 'first_25': first25, 'merged': merged, 'gaps': gapdec, 'not_run': not_run,
                                           'owner_decisions': owner, 'decision_map': dmap,
                                           'test_decisions': {k: tdec[k] for k in ('by_stage', 'by_test')},
                                           'claims_to_verify': claims,
                                           'claim_dispositions': claim_disp, 'disagreements': disagree,
                                           'counts': counts, 'summary': summary,
                                           'primary_agent_note': primary.get('note')}, indent=1, ensure_ascii=False) + '\n')
esc = lambda t: str(t).replace('|', '\\|').replace('\n', ' ')
md = ['### Stages', '',
      '"Inside the existing scope" is the ranking agent\'s `can_start_now`: the stage needs no venue call, new key, closed family or',
      'new data source. It is not a release: "Waits for program decisions" lists the open owner decisions of',
      '`docs/DERIV_SOTA_PROGRAM.md` that the specifications of the stage\'s tests need, then each test whose list differs with its',
      'own list (the primary agent\'s reading). Every test\'s list is the matrix field `program_decisions`; this column and the',
      'Gates column of the decision table below are derived from those lists (`build/primary.json` `test_decisions`). "Through',
      'earlier stages" adds the lists of the stages the stage builds on (the ranking agent\'s `depends_on_stages`, followed',
      'through), less its own; per test it is the matrix field `upstream_decisions`.', '',
      '| Stage | Name | Tests | Inside the existing scope | Ranking agent\'s gate | Waits for program decisions | Through earlier stages | Budget | Expectation |',
      '|---:|---|---:|---|---|---|---|---|---|']
for s in stages:
    md.append(f"| {s['stage']} | {esc(s.get('name', ''))} | {len(s.get('ids', []))} | {'yes' if s.get('can_start_now') else 'no'} | "
              f"{esc(s.get('owner_decision', ''))} | {esc(s['waits_for'])} | {nums(s['upstream_decisions']) or 'none'} | "
              f"{esc(s.get('budget', ''))} | {esc(s.get('expected', ''))} |")
md += ['', '### First 25 in run order', '', '| Rank | Id | Test | What runs | Kill rule | Cost |', '|---:|---|---|---|---|---|']
for x in first25:
    md.append(f"| {x['rank']} | `{x['id']}` | {esc(x.get('title', ''))} | {esc(x.get('what_runs', ''))} | {esc(x.get('kill_rule', ''))} | {esc(x.get('cost', ''))} |")
md += ['', '### Ids by stage, in run order', '', '| Stage | Ids in run order |', '|---:|---|']
for s in stages:
    md.append(f"| {s['stage']} | {' '.join('`' + i + '`' for i in s.get('ids', []))} |")
md += ['', '### Gap proposals and their dispositions', '', '| Id | Proposal | Source | Prior | Decision | Stage | Narrowed to, or why |',
       '|---|---|---|---|---|---:|---|']
own_ids = {g['id'] for g in primary.get('own_gaps', [])}
for g in gapdec:
    s0 = gaps_in.get(g['id'], {})
    md.append(f"| `{g['id']}` | {esc(s0.get('title', ''))} | {'primary agent' if g['id'] in own_ids else 'gap-finder'} | {esc(g.get('prior') or s0.get('prior', ''))} | "
              f"{esc(g.get('decision', ''))} | {esc(g.get('stage_or_covered_by', ''))} | {esc(g.get('narrowed_to') or g.get('why', ''))} |")
md += ['', '### Claims the ranking flagged for verification', '', '| Claim | Raised by | Status | Evidence | Consequence, or what is needed |',
       '|---|---|---|---|---|']
for c in claim_disp:
    md.append(f"| {esc(c.get('claim', ''))} | `{esc(c.get('by', ''))}` | {esc(c.get('status', ''))} | {esc(c.get('evidence', ''))} | "
              f"{esc(c.get('consequence') or c.get('needed', ''))} |")
md += ['', '### Owner decisions named by the ranking', '', '| # | Decision | Options | Ranking agent\'s recommendation | Unlocks |',
       '|---:|---|---|---|---|']
for i, o in enumerate(owner, 1):
    md.append(f"| {i} | {esc(o.get('decision', ''))} | {esc(' / '.join(o.get('options', [])))} | {esc(o.get('recommendation', ''))} | "
              f"{esc(', '.join(str(u) for u in o.get('unlocks', [])))} |")
md += ['', '### Open owner decisions 7-13 of the program and what they carry', '',
       '| Program decision | Ranking decisions it carries (# above) | Added by the primary agent | Gates |', '|---|---|---|---|']
for m in dmap:
    md.append(f"| {m['program_decision']}. {esc(m['name'])} | {esc(', '.join(str(n) for n in m['ranking_decisions']) or 'none')} | "
              f"{esc(m.get('added_by_primary_agent') or 'none')} | {esc(m['gates'])} |")
md += ['', '### The primary agent\'s notes on single tests', '', '| Id | Note |', '|---|---|']
for i, n in row_notes.items():
    md.append(f"| `{i}` | {esc(n)} |")
md += ['', '### Merges across groups', '', '| Kept | Absorbed | Why |', '|---|---|---|']
for m in merged:
    md.append(f"| `{m['keep']}` | {' '.join('`' + i + '`' for i in m.get('absorb', []))} | {esc(m.get('why', ''))} |")
md += ['', '### Not run', '', '| Id | Cluster | Why |', '|---|---|---|']
for n in not_run:
    md.append(f"| `{n['id']}` | {esc(clusters.get(n['id'], {}).get('title', ''))} | {esc(n.get('why', ''))} |")
md += ['', '### Where the lenses disagreed', '', '| Topic | Positions | Ranking agent\'s view |', '|---|---|---|']
for d in disagree:
    md.append(f"| {esc(d.get('topic', ''))} | {esc(d.get('positions', ''))} | {esc(d.get('your_view', ''))} |")
(out / 'plan_tables.md').write_text('\n'.join(md) + '\n')
by = Counter((r[0][0]) for r in where.values())
print('written matrix.jsonl.gz, plan.json, plan_tables.md to', out, dict(by))
