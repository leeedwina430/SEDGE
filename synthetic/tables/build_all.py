"""All 13 cases/all methods, retaining every seed and sample standard deviation."""
import csv,json,os,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('SYNTHETIC_RESULTS_ROOT',str(ROOT.parent/'results/reported')))
rows=[]
for suite in ['progressive','multimodal']:
    paths=sorted((SOURCE/suite/'cells').glob('*/*/results.json'))
    assert len(paths)==(21 if suite=='progressive' else 18),(suite,len(paths))
    for path in paths:
        assert (path.parent/'complete.json').exists(),path
        rows += [dict(suite=suite,case=path.parent.parent.name,seed=int(path.parent.name),**r) for r in json.loads(path.read_text())]
(ROOT/'all_results_per_seed.json').write_text(json.dumps(rows,indent=2)+'\n')
metrics=['validity','joint_mmd','x_mmd','conditional_x_mmd','nn_coverage','bridge_overlap','z1_satisfaction','z2_satisfaction','mechanism_mae']
with (ROOT/'all_results_mean_std.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['suite','case','method','seeds',*[v for k in metrics for v in [k+'_mean',k+'_sd']]])
    for suite,case,method in sorted({(r['suite'],r['case'],r['method']) for r in rows}):
        rr=[r for r in rows if (r['suite'],r['case'],r['method'])==(suite,case,method)]
        assert sorted(r['seed'] for r in rr)==[0,1,2]
        vals=[]
        for k in metrics:
            v=[r.get(k) for r in rr]
            vals.extend([statistics.mean(v),statistics.stdev(v)] if all(x is not None for x in v) else ['NA','NA'])
        w.writerow([suite,case,method,3,*vals])
print('Aggregated',len(rows),'per-seed method records across 13 cases (fractions, not percentages).')
