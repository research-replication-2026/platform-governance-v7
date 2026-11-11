"""Recompute all V8 fits in a new sibling directory; never overwrite supplied data."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,shutil,subprocess
from pathlib import Path
from public_materialize import materialize
BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='V8_fresh_reproduction')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    materialize()
    assert Path(args.output).name==args.output,'Output must be a new directory name, not a path'
    target=BASE.parent/args.output
    if target.exists(): raise FileExistsError('Choose a new output directory; existing results are preserved')
    target.mkdir()
    for name in ['v8_engine.py','run_v8.py','run_v8_replacements.py','V8_STATISTICAL_ANALYSIS.py',
                 'public_verify.py','public_materialize.py','validate_v8.py','V8_EXPERIMENTAL_PROTOCOL.md',
                 'V8_PROTOCOL_AMENDMENT_01.md','V8_EXPERIMENT_MANIFEST.json','V8_AMENDMENT_01_MANIFEST.json',
                 'PUBLIC_DATA_MANIFEST.json','PRIMARY_EFFECTIVE_SEED_AUDIT.csv']:
        shutil.copy2(BASE/name,target/name)
    shutil.copytree(BASE/'vendor',target/'vendor')
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[['run_v8.py','--stage','A','--workers',str(args.workers)],
              ['run_v8.py','--stage','BC','--workers',str(args.workers)],
              ['run_v8_replacements.py','--workers',str(args.workers)],['V8_STATISTICAL_ANALYSIS.py'],
              ['public_verify.py','--fresh']]
    for command in commands:
        subprocess.run([sys.executable,'-B',str(target/command[0])]+command[1:],cwd=BASE.parent,env=environment,check=True)
    # The verifier checks the freshly generated panel without needing gzip copies.
    print('Recomputed campaign and analyses in '+str(target))
    print('Compare V8_RESULTS.csv numerically with the supplied verified CSV; timestamps/figure metadata can differ.')

if __name__=='__main__': main()
