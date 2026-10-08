"""Recompute reported metrics from saved samples without training a model."""
import argparse,json,math
from pathlib import Path
from common import runtime as r
from ours import engine
from datagen import progressive,multimodal
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['progressive','multimodal'],default='progressive')
    p.add_argument('--case',default='linear')
    p.add_argument('--seed',type=int,choices=range(3),default=0)
    p.add_argument('--results-root',type=Path,default=ROOT/'results/reported')
    p.add_argument('--output',type=Path,default=ROOT/'results/recomputed')
    a=p.parse_args();torch.set_num_threads(1)
    module=progressive if a.suite=='progressive' else multimodal
    case=next((v for v in module.CASES if v['name']==a.case),None)
    if case is None:p.error('case does not belong to suite')
    folder=a.results_root/a.suite/'cells'/a.case/str(a.seed)
    dest=a.output/a.suite/a.case/str(a.seed)
    protected=[a.results_root.resolve(),(ROOT/'results/reported').resolve(),(ROOT/'results/published_tables').resolve()]
    if any(v==dest.resolve() or v in dest.resolve().parents for v in protected):p.error('output must be separate from source/archived results')
    dest.mkdir(parents=True,exist_ok=True)
    saved=torch.load(folder/'data.pt',map_location='cpu',weights_only=False)
    outputs=torch.load(folder/'samples.pt',map_location='cpu',weights_only=False)['outputs']
    engine.mechanism=module.mechanism;engine.compatible=module.compatible
    rows=engine.metrics(outputs,saved['truth'],saved['target'],r.Normalizer(**saved['normalizer']),case,saved['train'],dest)
    if a.suite=='multimodal':
        expected=module.audit(case)['posterior_right_mode_probability']
        for row in rows:
            x=outputs[row['method']].numpy();c=x[:,1];one,two=module.masks(c);overlap=one&two
            valid=module.mechanism(x,case).all(1)&(np.abs(x)<=1).all(1)&np.isfinite(x).all(1)
            row['right_mode_fraction_among_overlap']=float((c[overlap]>0).mean()) if overlap.any() else None
            row['right_mode_fraction_among_valid']=float((c[valid]>0).mean()) if valid.any() else None
            row['mode_mass_error']=abs(row['right_mode_fraction_among_overlap']-expected) if overlap.any() else None
            row['left_mode_valid_mass']=float((valid&(c<0)).mean());row['right_mode_valid_mass']=float((valid&(c>0)).mean());row['oracle_right_mode_probability']=expected
    reference={v['method']:v for v in json.loads((folder/'results.json').read_text())}
    max_error=0.
    assert set(reference)=={row['method'] for row in rows}
    for row in rows:
        old=reference[row['method']];assert row.keys()==old.keys()
        for k,value in row.items():
            if isinstance(value,(int,float)):
                assert math.isclose(value,old[k],rel_tol=1e-10,abs_tol=1e-10),(row['method'],k,value,old[k])
                max_error=max(max_error,abs(value-old[k]))
            else:assert value==old[k],(row['method'],k)
    (dest/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(suite=a.suite,case=a.case,seed=a.seed,methods=len(rows),max_absolute_error=max_error,tolerance=1e-10)
    (dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
