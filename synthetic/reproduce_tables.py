"""Rebuild tables from saved per-seed results; fail on numerical differences."""
import argparse,subprocess,sys,csv,json,os,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def csv_equal(a,b):
    left=list(csv.reader(a.open()));right=list(csv.reader(b.open()))
    if len(left)!=len(right):return False
    for x,y in zip(left,right):
        if len(x)!=len(y):return False
        for v,w in zip(x,y):
            if v==w:continue
            try:
                if not math.isclose(float(v),float(w),rel_tol=1e-12,abs_tol=1e-12):return False
            except ValueError:return False
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bridge',action='store_true',help='Also recompute bridge diagnostics from saved tensors (requires torch/numpy/scipy).')
    p.add_argument("--results-root",type=Path,help="Alternative results root containing progressive/ and multimodal/; comparison to historical tables is skipped.")
    a=p.parse_args()
    if a.results_root:os.environ["SYNTHETIC_RESULTS_ROOT"]=str(a.results_root.resolve())
    tasks=[['build_all.py'],['build.py'],['build.py','--selected'],['build_subcolumns.py'],['build_appendix_std.py'],['build_best_scales.py']]
    if a.bridge:tasks.append(['build_bridge_audit.py'])
    for name,*args in tasks:
        subprocess.run([sys.executable,str(ROOT/'tables'/name),*args],check=True,cwd=ROOT)
    if a.results_root:
        print("Built tables from",a.results_root,"; historical comparison skipped for new runs.")
        return
    matched=[]
    for path in sorted((ROOT/'tables').rglob('*')):
        if path.suffix not in {'.csv','.json','.tex'}:continue
        ref=ROOT/'results/published_tables'/path.relative_to(ROOT/'tables')
        if not ref.exists():continue
        if path.suffix=='.json':same=json.loads(path.read_text())==json.loads(ref.read_text())
        elif path.suffix=='.csv':same=csv_equal(path,ref)
        else:same=path.read_bytes()==ref.read_bytes()
        if not same:raise AssertionError(f'Reproduced table differs: {path.relative_to(ROOT)}')
        matched.append(str(path.relative_to(ROOT)))
    result={'matched_published_files':len(matched),'files':matched,'csv_numeric_tolerance':1e-12,'latex_comparison':'byte-exact','json_comparison':'exact'}
    (ROOT/'provenance/table_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
