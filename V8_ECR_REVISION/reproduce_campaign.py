"""Run preserved experiments in a new directory; never overwrite supplied outputs."""
import sys
sys.dont_write_bytecode=True
import argparse,os,shutil,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['primary','budget'],required=True);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();target=Path(args.output).resolve()
    if target.exists():raise FileExistsError('Choose a new output directory. Existing data are never overwritten.')
    target.mkdir(parents=True);run=target/'run';run.mkdir()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if args.stage=='primary':
        for name in ['v8_engine.py','run_v8.py','run_v8_replacements.py']:
            shutil.copy2(B/'FROZEN_SIMULATOR'/name,run/name)
        shutil.copytree(B/'FROZEN_SIMULATOR'/'vendor',run/'vendor')
        shutil.copy2(B/'FROZEN_SIMULATOR'/'frozen_primary_analysis.py',run/'V8_STATISTICAL_ANALYSIS.py')
        for name in ['V8_EXPERIMENTAL_PROTOCOL.md','V8_PROTOCOL_AMENDMENT_01.md','V8_EXPERIMENT_MANIFEST.json','V8_AMENDMENT_01_MANIFEST.json']:
            shutil.copy2(B/'ORIGINAL_V8_PROTOCOL'/name,run/name)
        archive=target/'experiments_v7'/'outputs'/'tables';archive.mkdir(parents=True)
        shutil.copy2(B/'V7_REFERENCE_DATA'/'v7_main_replications.csv',archive/'v7_main_replications.csv')
        commands=[['run_v8.py','--stage','A','--workers',str(args.workers)],['run_v8.py','--stage','BC','--workers',str(args.workers)],['run_v8_replacements.py','--workers',str(args.workers)],['V8_STATISTICAL_ANALYSIS.py']]
    else:
        shutil.copytree(B/'FROZEN_SIMULATOR',run/'FROZEN_SIMULATOR')
        for name in ['run_budget_sensitivity.py','BUDGET_MANIFEST.json','V8_PROTOCOL.md','V8_RESULTS.csv']:
            shutil.copy2(B/name,run/name)
        commands=[['run_budget_sensitivity.py','--prepare'],['run_budget_sensitivity.py','--pilot'],['run_budget_sensitivity.py','--workers',str(args.workers)]]
    for command in commands:subprocess.run([sys.executable,'-B',str(run/command[0])]+command[1:],cwd=target,env=env,check=True)
    print('Fresh results: '+str(run))
if __name__=='__main__':main()
