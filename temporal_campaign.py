"""Run validation tuning then final testing, one dataset at a time; resumable."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from temporal_data import DATASETS
from experiments import MAIN_MODELS

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--models',nargs='+',choices=MAIN_MODELS+('all',),default=['all'])
    p.add_argument('--mechanisms',nargs='+',choices=['RT','UMCAR'],default=['RT','UMCAR'])
    p.add_argument('--rates',nargs='+',type=float,default=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99])
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/temporal/prepared_v2')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.out/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for ds in a.datasets:
        base=a.out/ds
        common=['--datasets',ds,'--data-dir',str(a.data_dir.resolve()),'--models',*a.models,
                '--mechanisms',*a.mechanisms,'--rates',*map(str,a.rates),'--device',a.device,
                '--batch-size',str(a.batch_size)]
        for phase in ['tune','test']:
            argv=[sys.executable,str(ROOT/'temporal_run.py'),*common,'--phase',phase,'--out',str(base/phase)]
            argv+=['--seeds','2026'] if phase=='tune' else ['--selection',str(base/'tune')]
            (a.out/'status.json').write_text(json.dumps({'status':'running','dataset':ds,'phase':phase})+'\n')
            subprocess.run(argv,cwd=ROOT,check=True)
            subprocess.run([sys.executable,str(ROOT/'temporal_summarize.py'),str(base/phase)],cwd=ROOT,check=True)
    (a.out/'status.json').write_text(json.dumps({'status':'complete'})+'\n')

if __name__=='__main__':main()
