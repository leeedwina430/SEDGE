"""Post-hoc validity-selected scales, retaining all metrics from one scale."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent/'selected'
raw=json.loads((ROOT/'all_methods_per_seed.json').read_text())
cases=['multimodal_weighted_reference','linear','cubic','smoothstep','two_dimensional_bridge']
names=['Weighted Discrete','Linear','Cubic','Smoothstep','2D Bridge']
groups=[(r'Conditional diffusion (generation direction $Z\to X$)',[
    ('CD','CD'),('compositional_CD','Compositional CD')]),
    ('Extrapolative generation baselines',[
    ('additive_mean',r'Additive mean$^{\dagger}$'),('additive_residual',r'Additive + residual$^{\dagger}$'),
    ('compositional_residual',r'Fixed composition + residual$^{\dagger}$'),('rep4ex_residual',r'Rep4Ex-inspired + residual$^{\dagger}$')]),
    (r'Classifier-guided diffusion (likelihood direction $X_t\to Z$)',[
    ('CG_sparse_factorized','CG: sparse factorized'),('CG_dense_factorized','CG: dense factorized'),('CG_dense_joint','CG: dense joint')]),
    ('SEDGE (ours)',[('SEDGE_unweighted','SEDGE: no weighting'),('SEDGE_LGD','SEDGE-LGD'),('SEDGE_LWS','SEDGE-LWS')])]
methods=[m for _,group in groups for m,_ in group]
guided=['CG_sparse_factorized','CG_dense_factorized','CG_dense_joint','SEDGE_LGD']
selected={};values={}
for case in cases:
    for method in methods:
        candidates=[f'{method}_s{s:g}' for s in [.5,1,2]] if method in guided else [method]
        rows_by={m:[r for r in raw if r['case']==case and r['method']==m] for m in candidates}
        assert all(len(rows)==3 and sorted(r['seed'] for r in rows)==[0,1,2] for rows in rows_by.values())
        # Stable tie-break: smaller scale. One scale for all metrics and seeds.
        chosen=max(candidates,key=lambda m:statistics.mean(r['validity'] for r in rows_by[m]))
        selected[case,method]=chosen
        values[case,method]=[(statistics.mean(v),statistics.stdev(v)) for v in
            [[r[k]*(1 if k=='joint_mmd' else 100) for r in rows_by[chosen]] for k in ['validity','joint_mmd','nn_coverage']]]
with (ROOT/'best_scales_audit.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['case','method','selected_method','validity_mean','validity_sd','joint_mmd_mean','joint_mmd_sd','coverage_mean','coverage_sd'])
    for case in cases:
        for method in methods:w.writerow([case,method,selected[case,method],*[v for pair in values[case,method] for v in pair]])

def number(v,j,bold=False):
    a,b=(f'{v:.4f}' if j==1 else f'{v:.2f}').split('.');dot='.'
    if bold:a=r'\textbf{'+a+'}';b=r'\textbf{'+b+'}';dot=r'\textbf{.}'
    iw,fw=('0.575em','2.3em') if j==1 else ('1.725em','1.15em')
    return r'\makebox['+iw+r'][r]{'+a+'}'+r'\makebox[0.32em][c]{'+dot+'}'+r'\makebox['+fw+r'][l]{'+b+'}'

for std in [False,True]:
    name='best_scales_appendix_std' if std else 'best_scales'
    env='sidewaystable' if std else 'table*'
    lines=[r'% Requires booktabs, graphicx, rotating (for appendix), and xcolor[table].',
        r'\definecolor{sedgepurple}{HTML}{F0EDF7}',r'\begin{'+env+'}[p]' if std else r'\begin{table*}[t]',
        r'\centering\small',r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.18}',
        r'\caption{Validity (\%), joint MMD, and coverage (\%) across five DGPs. '+
        (r'Each entry shows the mean above its $\pm$ sample standard deviation across three seeds. ' if std else 'Results are means across three seeds. ')+
        r'For each CG variant and SEDGE-LGD, one scale per DGP is selected by highest mean validity among $s\in\{0.5,1,2\}$; all metrics use that same scale. This is post-hoc test-set selection, not validation-based tuning.}',
        r'\label{tab:sedge-'+name.replace('_','-')+'}',r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{l'+r'ccc@{\hspace{10pt}}'*4+'ccc}',r'\toprule',
        'Method & '+' & '.join(r'\multicolumn{3}{c}{'+n+'}' for n in names)+r' \\',
        ''.join(r'\cmidrule(lr){'+f'{2+3*i}-{4+3*i}'+'}' for i in range(5)),
        ' & '+' & '.join([r'Valid.\,$\uparrow$ & J-MMD\,$\downarrow$ & Cover.\,$\uparrow$']*5)+r' \\']
    for title,group in groups:
        lines.extend([r'\midrule',r'\multicolumn{16}{l}{\textit{'+title+r'}} \\'])
        for method,label in group:
            if method.startswith('SEDGE_'):lines.append(r'\rowcolor{sedgepurple}')
            cells=[]
            for case in cases:
                for j,(mean,sd) in enumerate(values[case,method]):
                    best=(min if j==1 else max)(values[case,m][j][0] for m in methods)
                    v=number(mean,j,abs(mean-best)<1e-10)
                    cells.append(r'\shortstack{\makebox[0.8em]{}'+v+r'\\[-1pt]\makebox[0.8em]{$\pm$}'+number(sd,j)+'}' if std else v)
            lines.append(label+' & '+' & '.join(cells)+(r' \\[3pt]' if std else r' \\'))
    lines.extend([r'\bottomrule',r'\end{tabular}}',r'\par\vspace{4pt}',r'\begin{minipage}{\linewidth}\footnotesize',
        r'Selected scales in column order: sparse-factorized CG $(2,2,2,2,2)$; dense-factorized CG $(2,2,2,2,2)$; dense-joint CG $(0.5,2,2,2,2)$; SEDGE-LGD $(1,1,2,1,2)$.',
        r'Validity requires both specifications and root support; J-MMD measures joint $(X,Z)$ fidelity; coverage uses the reference-based fixed-radius criterion. Bold marks the best displayed mean per DGP/metric, not statistical significance.',
        r'$^{\dagger}$Paper-inspired adaptations, not faithful reproductions; additive mean is a deterministic control. Weighted Discrete has conditional exclusion 2.5\%/5\%; continuous settings pass the working 1\% band-conditional criterion. The expanded LGD sweep was cancelled before execution.'])
    if std:lines.append(r'Standard deviations use $n-1$ ($n=3$), not standard errors; displayed zeros may reflect rounding.')
    lines.extend([r'\end{minipage}',r'\end{'+env+'}'])
    table='\n'.join(lines)+'\n';(ROOT/f'five_dgp_{name}.tex').write_text(table)
    (ROOT/f'preview_{name}.tex').write_text('\n'.join([r'\documentclass[11pt]{article}',
        r'\usepackage[a4paper,'+('' if std else 'landscape,')+r'margin=12mm]{geometry}',
        r'\usepackage{rotating,booktabs,graphicx}',r'\usepackage[table]{xcolor}',r'\pagestyle{empty}',
        r'\begin{document}',table,r'\end{document}'])+'\n')
