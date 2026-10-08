"""Alternative presentation of the same selected results; no metric changes."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent/'selected'
rows = list(csv.DictReader((ROOT/'five_dgp_metrics.csv').open()))
cases = list(dict.fromkeys(r['dgp'] for r in rows))
methods = list(dict.fromkeys(r['method'] for r in rows))
names = dict(multimodal_weighted_reference='Weighted Discrete',linear='Linear',
             cubic='Cubic',smoothstep='Smoothstep',two_dimensional_bridge='2D Bridge')
labels = dict(CD='CD',compositional_CD='Compositional CD',
              additive_mean=r'Additive mean$^{\dagger}$',
              additive_residual=r'Additive + residual$^{\dagger}$',
              compositional_residual=r'Fixed composition + residual$^{\dagger}$',
              rep4ex_residual=r'Rep4Ex-inspired + residual$^{\dagger}$',
              SEDGE_unweighted='SEDGE: no weighting',SEDGE_LWS='SEDGE-LWS')
for method in methods:
    if method.startswith('SEDGE_LGD_s'):
        labels[method]=r'SEDGE-LGD ($s='+method.split('_s')[-1]+'$)'
    if method.startswith('CG_'):
        base,scale=method.rsplit('_s',1)
        labels[method]='CG: '+base[3:].replace('_',' ')+r' ($s='+scale+'$)'
keys=['validity_percent','joint_mmd','nn_coverage_percent']
values={(r['dgp'],r['method']):[float(r[k]) for k in keys] for r in rows}
lines=[r'% Requires booktabs, graphicx, and xcolor with the table option.',
       r'\definecolor{sedgepurple}{HTML}{F0EDF7}',
       r'\begin{table*}[t]',r'\centering',r'\small',
       r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.18}',
       r'\caption{Validity (\%), joint MMD, and reference coverage (\%) across five DGPs (three-seed means). Bold marks the best displayed method for each DGP and metric.}',
       r'\label{tab:sedge-five-subcolumns}',r'\resizebox{\textwidth}{!}{%',
       r'\begin{tabular}{l'+r'ccc@{\hspace{10pt}}'*4+'ccc}',r'\toprule',
       r'Method & '+' & '.join(r'\multicolumn{3}{c}{'+names[c]+'}' for c in cases)+r' \\',
       ''.join(r'\cmidrule(lr){'+f'{2+3*i}-{4+3*i}'+'}' for i in range(5)),
       ' & '+' & '.join([r'Valid.\,$\uparrow$ & J-MMD\,$\downarrow$ & Cover.\,$\uparrow$']*5)+r' \\',
       r'\midrule']
for method in methods:
    if method=='SEDGE_unweighted': lines.append(r'\midrule')
    if method.startswith('SEDGE_'): lines.append(r'\rowcolor{sedgepurple}')
    cells=[]
    for case in cases:
        for j,value in enumerate(values[case,method]):
            best=(min if j==1 else max)(values[case,m][j] for m in methods)
            integer,fraction=(f'{value:.4f}' if j==1 else f'{value:.2f}').split('.')
            dot='.'
            if abs(value-best)<1e-10:
                integer=r'\textbf{'+integer+'}';fraction=r'\textbf{'+fraction+'}';dot=r'\textbf{.}'
            iw,fw=('0.575em','2.3em') if j==1 else ('1.725em','1.15em')
            cells.append(r'\makebox['+iw+r'][r]{'+integer+'}'+r'\makebox[0.32em][c]{'+dot+'}'+r'\makebox['+fw+r'][l]{'+fraction+'}')
    lines.append(labels[method]+' & '+' & '.join(cells)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}%',r'}',r'\par\vspace{4pt}',
          r'\begin{minipage}{\textwidth}\footnotesize',
          r'Validity requires both specifications and root support. J-MMD compares joint $(X,Z)$ distributions. Coverage is the fraction of reference samples covered under the original fixed-radius criterion. Higher validity/coverage and lower J-MMD are preferred; compare methods within each DGP.',
          r'$^{\dagger}$Paper-inspired adaptations, not faithful reproductions. Additive mean is a deterministic control rather than a posterior sampler. Existing LGD scales are all shown; CG scales 1 and 2 are shown. The expanded LGD sweep is not included.',
          r'Weighted Discrete is a relaxed Assumption-3 control (conditional exclusion 2.5\% and 5\%); continuous settings meet the working 1\% criterion. This is a results-informed selection, not a claim of uniform superiority. No significance claim is implied by boldface.',
          r'\end{minipage}',r'\end{table*}']
table='\n'.join(lines)+'\n'
(ROOT/'five_dgp_subcolumns.tex').write_text(table)
(ROOT/'preview_subcolumns.tex').write_text('\n'.join([
    r'\documentclass[11pt]{article}',r'\usepackage[a4paper,landscape,margin=12mm]{geometry}',
    r'\usepackage{booktabs,graphicx}',r'\usepackage[table]{xcolor}',r'\pagestyle{empty}',
    r'\begin{document}',table,r'\end{document}'])+'\n')
