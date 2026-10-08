"""Three-seed sample SDs for the same selected Table 1, rotated for appendix."""
import csv
import json
from pathlib import Path
import runpy
import statistics

HERE = Path(__file__).resolve().parent
meta = runpy.run_path(str(HERE/'build_subcolumns.py'))
ROOT = meta['ROOT']
cases, methods, names, labels = (meta[k] for k in ['cases','methods','names','labels'])
raw = json.loads((ROOT/'all_methods_per_seed.json').read_text())
keys = ['validity','joint_mmd','nn_coverage']
stats = {}
for case in cases:
    for method in methods:
        rows = [r for r in raw if r['case']==case and r['method']==method]
        assert sorted(r['seed'] for r in rows)==[0,1,2]
        stats[case,method] = []
        for key in keys:
            vals = [r[key]*(1 if key=='joint_mmd' else 100) for r in rows]
            stats[case,method].append((statistics.mean(vals),statistics.stdev(vals)))
        for j in range(3):
            assert abs(stats[case,method][j][0]-meta['values'][case,method][j])<1e-10

with (ROOT/'five_dgp_mean_std.csv').open('w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['dgp','method','seeds',*[f'{k}_{t}' for k in keys for t in ['mean','sample_std']]])
    for case in cases:
        for method in methods:
            writer.writerow([case,method,3,*[v for pair in stats[case,method] for v in pair]])

lines=[r'% Preamble: \usepackage{rotating,booktabs,graphicx}; \usepackage[table]{xcolor}',
       r'% Use sidewaystable* instead of sidewaystable in a two-column document.',
       r'\definecolor{sedgepurple}{HTML}{F0EDF7}',
       r'\begin{sidewaystable}[p]',r'\centering',r'\small',
       r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.1}',
       r'\caption{Generation with observed variables and known structure: mean and sample standard deviation across three seeds. Each entry shows the mean above its $\pm$ standard deviation. Validity and coverage are percentages. Bold marks the best mean within each DGP and metric.}',
       r'\label{tab:sedge-five-appendix-std}',r'\resizebox{\linewidth}{!}{%',
       r'\begin{tabular}{l'+r'ccc@{\hspace{10pt}}'*4+'ccc}',r'\toprule',
       'Method & '+' & '.join(r'\multicolumn{3}{c}{'+names[c]+'}' for c in cases)+r' \\',
       ''.join(r'\cmidrule(lr){'+f'{2+3*i}-{4+3*i}'+'}' for i in range(5)),
       ' & '+' & '.join([r'Valid.\,$\uparrow$ & J-MMD\,$\downarrow$ & Cover.\,$\uparrow$']*5)+r' \\',r'\midrule']
def aligned(value,j,bold=False):
    integer,fraction=(f'{value:.4f}' if j==1 else f'{value:.2f}').split('.')
    dot='.'
    if bold:
        integer=r'\textbf{'+integer+'}';fraction=r'\textbf{'+fraction+'}';dot=r'\textbf{.}'
    iw,fw=('0.575em','2.3em') if j==1 else ('1.725em','1.15em')
    return r'\makebox['+iw+r'][r]{'+integer+'}'+r'\makebox[0.32em][c]{'+dot+'}'+r'\makebox['+fw+r'][l]{'+fraction+'}'
for method in methods:
    if method=='SEDGE_unweighted': lines.append(r'\midrule')
    if method.startswith('SEDGE_'): lines.append(r'\rowcolor{sedgepurple}')
    cells=[]
    for case in cases:
        for j,(mean,sd) in enumerate(stats[case,method]):
            best=(min if j==1 else max)(stats[case,m][j][0] for m in methods)
            cells.append(r'\shortstack{\makebox[0.8em]{}'+aligned(mean,j,abs(mean-best)<1e-10)
                         +r'\\[-1pt]\makebox[0.8em]{$\pm$}'+aligned(sd,j)+'}')
    lines.append(labels[method]+' & '+' & '.join(cells)+r' \\[3pt]')
lines += [r'\bottomrule',r'\end{tabular}%',r'}',r'\par\vspace{4pt}',
          r'\begin{minipage}{\linewidth}\footnotesize',
          r'Standard deviations use $n-1$ in the denominator ($n=3$), not standard errors. Displayed zeros may reflect rounding. Validity requires both specifications and root support; J-MMD measures joint $(X,Z)$ fidelity; coverage uses the reference-based fixed-radius criterion.',
          r'$^{\dagger}$Paper-inspired adaptations, not faithful reproductions. Additive mean is a deterministic control, not a posterior sampler. All three original LGD scales and CG scales 1 and 2 are shown; the expanded LGD sweep is excluded.',
          r'Weighted Discrete is a relaxed Assumption-3 control (conditional exclusion 2.5\%/5\%); continuous settings meet the working 1\% band-conditional criterion. This results-informed selection does not establish uniform superiority. Boldface does not imply statistical significance.',
          r'\end{minipage}',r'\end{sidewaystable}']
table='\n'.join(lines)+'\n'
(ROOT/'five_dgp_appendix_std.tex').write_text(table)
(ROOT/'preview_appendix_std.tex').write_text('\n'.join([
    r'\documentclass[11pt]{article}',r'\usepackage[a4paper,margin=12mm]{geometry}',
    r'\usepackage{rotating,booktabs,graphicx}',r'\usepackage[table]{xcolor}',r'\pagestyle{empty}',
    r'\begin{document}',table,r'\end{document}'])+'\n')
