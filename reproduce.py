"""Install the download bundle, check it, or launch the GPU deployment experiments."""
from __future__ import annotations
import argparse
import os
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,help='Extracted CBRS-CEDAR-RT-ASSETS folder')
    p.add_argument('--task',choices=['install','check','smoke','full','report'],default='check')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.assets and not a.dry_run:
        from scripts.release_assets import install
        install(a.assets,ROOT,'rt')
    if a.task=='install':
        if not a.assets: p.error('install requires --assets')
        return
    if not a.dry_run and a.task in ('check','smoke','full'):
        from scripts.release_assets import digest
        records=json.loads((ROOT/'artifacts/weights/manifest.json').read_text())
        if len(records)!=12: raise ValueError('Expected 12 reference checkpoints')
        for record in records:
            path=ROOT/record['path']
            if not path.is_file() or digest(path)!=record['sha256']:
                raise ValueError(f'Missing or mismatched checkpoint: {path}')
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONDONTWRITEBYTECODE']='1'
    py=sys.executable
    if a.task=='report':
        commands=[[py,'-B','analyze_results.py']]
    elif a.task=='check':
        commands=[[py,'-B','-m','cbrs_quant','validate-inputs','--data','data/yolo_split',
                   '--profile','configs/spectrogram.json','--output','artifacts/dataset.json']]
    else:
        commands=[[py,'-B','run_experiments.py']]
        if a.task=='smoke':
            commands[0]+=['--models','cbrs_yolo8n','sig_yolo_v8n_f20t20_c100','--batches','1','--smoke']
    for c in commands:
        if a.dry_run:
            print(subprocess.list2cmdline(c))
        else:
            subprocess.run(c,cwd=ROOT,env=env,check=True)


if __name__=='__main__':
    main()
