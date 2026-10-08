import os
"""Table-1 matched bridge compatibility and irreducible specification residuals."""
import csv,json,sys,importlib.util,statistics
from pathlib import Path
import numpy as np
import torch
BASE=Path(os.environ.get('SYNTHETIC_RESULTS_ROOT',str(Path(__file__).resolve().parents[1]/'results/reported')))/'progressive'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datagen import progressive as data
spec=importlib.util.spec_from_file_location('binary_data',Path(__file__).resolve().parents[1]/'datagen/multimodal.py')
bd=importlib.util.module_from_spec(spec);spec.loader.exec_module(bd)
OUT=Path(__file__).resolve().parent/'selected'
selection=list(csv.DictReader((OUT/'best_scales_audit.csv').open()))
cases=['multimodal_weighted_reference','linear','cubic','smoothstep','two_dimensional_bridge']
names=['Weighted Discrete','Linear','Cubic','Smoothstep','2D Bridge']
groups=[('Conditional diffusion', [('CD','CD'),('compositional_CD','Compositional CD')]),
('Classifier-guided diffusion',[('CG_sparse_factorized','CG: sparse factorized'),('CG_dense_factorized','CG: dense factorized'),('CG_dense_joint','CG: joint')]),
('Extrapolation-inspired baselines',[('additive_mean',r'Additive mean$^{\dagger}$'),('additive_residual',r'Additive + residual$^{\dagger}$'),('compositional_residual',r'Fixed composition + residual$^{\dagger}$'),('rep4ex_residual',r'Rep4Ex-inspired + residual$^{\dagger}$')]),
('SEDGE (ours)',[('SEDGE_unweighted','SEDGE: no weighting'),('SEDGE_LGD','SEDGE-LGD'),('SEDGE_LWS','SEDGE-LWS')])]
methods=[m for _,g in groups for m,_ in g];rows=[]
for case_name in cases:
    binary=case_name==cases[0]
    case=next(c for c in (bd.CASES if binary else data.CASES) if c['name']==case_name)
    for seed in range(3):
        folder=(BASE.parent/'multimodal' if binary else BASE)/'cells'/case_name/str(seed)
        saved=torch.load(folder/'data.pt',map_location='cpu',weights_only=False)
        z=saved['target'].numpy();outputs=torch.load(folder/'samples.pt',map_location='cpu',weights_only=False)['outputs']
        for method in methods:
            chosen=next(r['selected_method'] for r in selection if r['case']==case_name and r['method']==method)
            x=outputs[chosen].numpy();c=x[:,1:-1];support=np.isfinite(c).all(1)&((c>=(-1 if binary else 0))&(c<=1)).all(1)
            one,two=(bd.compatible if binary else data.compatible)(c,z,case)
            if binary:
                error=np.column_stack([~one,~two]).astype(float)
            else:
                a,b,d,e=data.components(c,case)
                lo=np.column_stack([b,e]);hi=lo+np.column_stack([a,d])
                error=np.maximum(np.maximum(lo-z,z-hi),0)
            # Residual means restricted to legal bridge coordinates. Report the
            # excluded fraction explicitly; compatibility counts all samples.
            legal=error[support];pred=(bd.mechanism if binary else data.mechanism)(x,case)
            allroot=np.isfinite(x).all(1)&((x>=(-1 if binary else 0))&(x<=1)).all(1)
            row=dict(case=case_name,seed=seed,method=method,selected_method=chosen,n=len(x),bridge_support=float(support.mean()),compatible_z1=float((support&one).mean()),compatible_z2=float((support&two).mean()),compatible_both=float((support&one&two).mean()),bridge_mae_z1=float(legal[:,0].mean()) if len(legal) else None,bridge_mae_z2=float(legal[:,1].mean()) if len(legal) else None,bridge_max_residual=float(legal.max(1).mean()) if len(legal) else None,actual_validity=float((allroot&(abs(pred-z)<=(0 if binary else .005)).all(1)).mean()))
            rows.append(row)
with (OUT/'bridge_audit_per_seed.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for case in cases:
    for method in methods:
        rr=[r for r in rows if r['case']==case and r['method']==method];v=dict(case=case,method=method,selected_method=rr[0]['selected_method'])
        for key in ['bridge_support','compatible_z1','compatible_z2','compatible_both','bridge_mae_z1','bridge_mae_z2','bridge_max_residual','actual_validity']:
            vals=[r[key] for r in rr];v[key+'_mean']=statistics.mean(vals) if all(a is not None for a in vals) else None;v[key+'_sd']=statistics.stdev(vals) if all(a is not None for a in vals) else None
        summary.append(v)
with (OUT/'bridge_audit_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
def num(v,dec,bold=False):
    if v is None:return '--'
    a,b=f'{v:.{dec}f}'.split('.');fmt=lambda s:r'\textbf{'+s+'}' if bold else s
    return r'\makebox[1.8em][r]{'+fmt(a)+r'}\makebox[.3em]{'+fmt('.')+r'}\makebox['+str(dec*.58)+r'em][l]{'+fmt(b)+'}'
tables=[]
for kind,keys,labels in [('compatibility',['compatible_z1','compatible_z2','compatible_both'],[r'$Z_1\uparrow$',r'$Z_2\uparrow$',r'Both$\uparrow$']),('residual',['bridge_mae_z1','bridge_mae_z2','bridge_max_residual'],[r'$e_1\downarrow$',r'$e_2\downarrow$',r'$\max(e_1,e_2)\downarrow$'])]:
    for std in [False,True]:
        percent=kind=='compatibility';dec=2 if percent else 4
        caption=('Bridge compatibility rates (percent).' if percent else 'Mean irreducible specification residuals at generated bridge coordinates.')+' Same settings, methods and previously selected scales as Table 1; no scale reselection. '+('Mean and sample standard deviation across three seeds.' if std else 'Means across three seeds.')
        lines=[r'\begin{table*}[p]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\caption{'+caption+'}',r'\label{tab:bridge-'+kind+('-std' if std else '')+'}',r'\resizebox{\linewidth}{!}{\begin{tabular}{l'+('ccc@{\hspace{10pt}}'*4)+'ccc}',r'\toprule','Method & '+' & '.join(r'\multicolumn{3}{c}{'+n+'}' for n in names)+r' \\', ' & '+' & '.join(labels*5)+r' \\']
        for title,group in groups:
            lines += [r'\midrule',r'\multicolumn{16}{l}{\textit{'+title+r'}} \\']
            for method,label in group:
                if method.startswith('SEDGE'):lines.append(r'\rowcolor{sedgepurple}')
                cells=[]
                for case in cases:
                    r=next(v for v in summary if v['case']==case and v['method']==method)
                    for key in keys:
                        vals=[v[key+'_mean'] for v in summary if v['case']==case and v[key+'_mean'] is not None];best=(max if percent else min)(vals);v=r[key+'_mean'];scale=100 if percent else 1
                        cell=num(None if v is None else v*scale,dec,v is not None and abs(v-best)<1e-10)
                        if std:cell=r'\shortstack{'+cell+r'\\$\pm$ '+num(None if r[key+'_sd'] is None else r[key+'_sd']*scale,dec)+'}'
                        cells.append(cell)
                lines.append(label+' & '+' & '.join(cells)+r' \\')
        lines += [r'\bottomrule\end{tabular}}',r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize',r'Compatibility asks whether some legal private variable can satisfy the requested specification at the generated bridge, not whether the generated private variable does so. Both requires the same bridge to permit both specifications. Out-of-support or nonfinite bridges count as incompatible.',r'For continuous mechanisms $Z_i=a_i(C)X_i+b_i(C)$, $e_i=\max\{b_i(C)-z_i,z_i-b_i(C)-a_i(C),0\}$. Binary $e_i$ is the incompatibility indicator. Residual means condition on bridge root support; support rates and actual full-sample validity are in the companion CSV. Residuals use raw specification units; compare within each DGP. Bold indicates the best displayed mean, not significance. $^{\dagger}$Paper-inspired adaptations.',r'\end{minipage}',r'\end{table*}']
        tex='\n'.join(lines)+'\n';name='five_dgp_bridge_'+kind+('_std' if std else '')+'.tex';(OUT/name).write_text(tex);tables.append(tex)
preview='\n'.join([r'\documentclass[11pt]{article}',r'\usepackage[a4paper,landscape,margin=12mm]{geometry}',r'\usepackage{booktabs,graphicx}',r'\usepackage[table]{xcolor}',r'\definecolor{sedgepurple}{HTML}{F0EDF7}',r'\begin{document}',('\n'+r'\clearpage'+'\n').join(tables),r'\end{document}'])
(OUT/'preview_bridge_audit.tex').write_text(preview)
print('Computed',len(rows),'seed-method-setting rows; wrote four matching tables.')
