#!/usr/bin/env python3
"""Offline trigger stealth evaluation; never trains a model or fabricates UAPs."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import torch, yaml
from attacks.triggers import apply_trigger
from data.dataset import load_cifar10
from evaluation.evaluator import compute_stealth_metrics

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/summaries/mainexp'
def load_delta(output):
    candidates=[]
    for pattern in ('*delta*.pt','*trigger*.pt','*uap*.pt'): candidates.extend(output.rglob(pattern) if output.exists() else [])
    for path in sorted(set(candidates)):
        try:
            value=torch.load(path,map_location='cpu',weights_only=True)
            if isinstance(value,dict): value=next((value[k] for k in ('delta','perturbation','trigger') if k in value),None)
            if isinstance(value,torch.Tensor): return value, path
        except (OSError,RuntimeError,ValueError,KeyError): pass
    return None,None
def main():
    p=argparse.ArgumentParser(); p.add_argument('--configs',nargs='+',type=Path,required=True); p.add_argument('--num-samples',type=int,default=1000); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); rows=[]
    for path in args.configs:
        path=path if path.is_absolute() else ROOT/path; c=yaml.safe_load(path.read_text()); a=c['attack']; trigger=a['trigger']; method=c['experiment']['name']; output=ROOT/c['experiment']['output_dir']
        _,test=load_cifar10(root=ROOT/c['dataset']['root'],download=c['dataset'].get('download',False)); images=torch.stack([test[i][0] for i in range(min(args.num_samples,len(test)))])
        if trigger['type']=='patch':
            altered=apply_trigger(images,'patch',poison_indices=torch.arange(len(images)),patch_size=trigger.get('patch_size',3),pixel_value=trigger.get('pixel_value',1.0)); note='offline fixed-patch evaluation'
        else:
            delta,delta_path=load_delta(output)
            if delta is None: rows.append({'method':method,'config_path':str(path.relative_to(ROOT)),'num_samples':len(images),'psnr':'N/A','ssim':'N/A','linf':'N/A','note':f'missing saved UAP delta under {output.relative_to(ROOT)}'}); continue
            altered=apply_trigger(images,'adversarial',perturbation=delta); note=f'loaded {delta_path.relative_to(ROOT)}'
        m=compute_stealth_metrics(images,altered,altered-images); rows.append({'method':method,'config_path':str(path.relative_to(ROOT)),'num_samples':len(images),'psnr':m['psnr'],'ssim':m['ssim'],'linf':m['linf'],'note':note})
    with (OUT/'trigger_stealth_metrics.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=['method','config_path','num_samples','psnr','ssim','linf','note']); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
