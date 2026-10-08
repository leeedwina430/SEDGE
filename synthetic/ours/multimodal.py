"""Isolated extension: same full suite, new data, old results untouched."""
import argparse,importlib.util,json,sys,statistics
from pathlib import Path
import numpy as np
import torch
from datagen import multimodal as bd
ROOT=Path(__file__).resolve().parents[1]/'runs/multimodal'
from ours import engine
engine.ROOT=ROOT;engine.CASES=bd.CASES;engine.audit=bd.audit;engine.prepare=bd.prepare;engine.mechanism=bd.mechanism;engine.compatible=bd.compatible;engine.binary_probability=bd.binary_probability
original_load=engine.load
def stabilized_load(name,path):
    mod=original_load(name,path)
    if name=='sedge_legacy':
        # Rare binary positives make minibatch OLS rank-deficient. Ridge avoids
        # undefined lstsq gradients; applied only to this extension's adaptation.
        def affine(z,h):
            design=torch.cat([z,torch.ones_like(z[:,:1])],1).double();target=h.double()
            gram=design.T@design;coef=torch.linalg.solve(gram+1e-4*torch.eye(gram.shape[0],device=gram.device,dtype=gram.dtype),design.T@target).float()
            return coef,h-design.float()@coef
        mod.affine=affine
    return mod
engine.load=stabilized_load

def main(index,smoke):
    engine.main(index,smoke);case=bd.CASES[index//3];seed=index%3;dest=ROOT/('smoke' if smoke else 'cells')/case['name']/str(seed)
    samples=torch.load(dest/'samples.pt',map_location='cpu',weights_only=False)['outputs'];rows=json.loads((dest/'results.json').read_text());expected=bd.audit(case)['posterior_right_mode_probability']
    for row in rows:
        x=samples[row['method']].numpy();c=x[:,1];a,b=bd.masks(c);overlap=a&b;valid=bd.mechanism(x,case).all(1)&(np.abs(x)<=1).all(1)&np.isfinite(x).all(1)
        row['right_mode_fraction_among_overlap']=float((c[overlap]>0).mean()) if overlap.any() else None
        row['right_mode_fraction_among_valid']=float((c[valid]>0).mean()) if valid.any() else None
        row['mode_mass_error']=abs(row['right_mode_fraction_among_overlap']-expected) if overlap.any() else None
        row['left_mode_valid_mass']=float((valid&(c<0)).mean());row['right_mode_valid_mass']=float((valid&(c>0)).mean());row['oracle_right_mode_probability']=expected
    (dest/'results.json').write_text(json.dumps(rows,indent=2)+'\n')

def report():
    engine.report()
    # The reused report header describes the previous suite; replace it here.
    p=ROOT/'REPORT.md';text=p.read_text();text=text.replace('# Progressive SEDGE benchmark','# Multimodal discrete SEDGE candidates').replace('Historical binary reference violates the working 1% conditional criterion (5%); all other geometries audited before submission. No claim of guaranteed SEDGE success.','Strict candidates satisfy 1% conditional exclusion; reference candidates are labeled relaxed controls, at most 5%. No outcome-based filtering.');p.write_text(text)
    combined=[]
    for folder in [ROOT.parent/'progressive',ROOT]:
        for path in (folder/'cells').glob('*/*/results.json'):
            for row in json.loads(path.read_text()):combined.append(dict(row,case=path.parent.parent.name,seed=int(path.parent.name),source=str(folder)))
    (ROOT/'all_results.json').write_text(json.dumps(combined,indent=2)+'\n')
    lines=['# Previous and new SEDGE benchmarks','','All completed seeds included; different DGPs are not pooled. Validity and overlap are percentages. Full metrics and all guidance scales are in all_results.json.','','| Case | Seeds | Method | Validity | X2 overlap | X MMD | Mode-mass error |','|---|---:|---|---:|---:|---:|---:|']
    for case in dict.fromkeys(r['case'] for r in combined):
        for method in ['CD','compositional_CD','SEDGE_LWS','SEDGE_LGD_s1','SEDGE_LGD_s2','CG_dense_factorized_s2','CG_dense_joint_s2','additive_residual','compositional_residual','rep4ex_residual']:
            rr=[r for r in combined if r['case']==case and r['method']==method]
            if not rr:continue
            def avg(k,scale=1):return f'{scale*statistics.mean(v[k] for v in rr):.4f}' if all(v.get(k) is not None for v in rr) else 'NA'
            lines.append('| '+' | '.join([case,str(len(rr)),method,avg('validity',100),avg('bridge_overlap',100),avg('x_mmd'),avg('mode_mass_error')])+' |')
    (ROOT/'COMPARISON_ALL.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args();report() if a.report else main(a.index,a.smoke)
