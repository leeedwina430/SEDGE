"""Train and evaluate one suite/seed, writing only to a fresh run directory."""
import argparse
from pathlib import Path
from common import runtime

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['progressive','multimodal'],default='progressive')
    p.add_argument('--index',type=int,default=0,help='case_index * 3 + seed (seed 0, 1, 2)')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,help='Suite output directory; defaults to runs/<suite>')
    a=p.parse_args()
    if a.suite=='progressive':
        from ours import engine
        runner=engine
    else:
        from ours import multimodal as runner
        engine=runner.engine
    if not 0<=a.index<len(engine.CASES)*3:p.error('index is outside the suite')
    if a.output:
        engine.ROOT=a.output.resolve()
        runner.ROOT=engine.ROOT
    protected=Path(__file__).resolve().parent/'results'
    if engine.ROOT.resolve()==protected or protected in engine.ROOT.resolve().parents:
        p.error('output must not overwrite bundled results')
    import torch
    if not torch.cuda.is_available():p.error('Historical training/sampling requires CUDA; use reproduce_tables.py for CPU-only reproduction.')
    runner.main(a.index,a.smoke)
if __name__=='__main__':main()
