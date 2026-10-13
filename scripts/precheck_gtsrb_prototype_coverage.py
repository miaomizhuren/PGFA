"""Non-training GTSRB IID and single-client prototype coverage precheck."""
from pathlib import Path
import csv
import sys
import torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import get_dataset_spec, load_dataset
from data.partition import partition_iid
from models.resnet18 import build_resnet18
from attacks.prototype import collect_class_statistics, aggregate_class_statistics, enforce_strict_prototype_coverage

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/saved_results/gtsrb_precheck_20260912'; N=43; IDS=[0,1,2]
def main():
 OUT.mkdir(parents=True,exist_ok=True); spec=get_dataset_spec('gtsrb'); train,test=load_dataset('gtsrb',ROOT/'datasets',download=False); parts=partition_iid(len(train),100,2026); rows=[]
 for cid in IDS:
  counts=[sum(train.targets[i]==k for i in parts[cid]) for k in range(N)]; total=sum(counts); covered=sum(x>0 for x in counts); dom=max(range(N),key=lambda k:counts[k]); rows.append({'client_id':cid,**{f'class_{k}':counts[k] for k in range(N)},'total_samples':total,'covered_classes':covered,'missing_classes':';'.join(str(k) for k,x in enumerate(counts) if not x),'dominant_class':dom,'dominant_ratio':counts[dom]/total})
 chosen=sorted(rows,key=lambda r:(-r['covered_classes'],r['dominant_ratio'],r['client_id']))[0]; chosen['recommended_as_prototype_source']=True
 for r in rows:r['recommended_as_prototype_source']=r['client_id']==chosen['client_id']
 with (OUT/'GTSRB_malicious_clients_label_distribution.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=build_resnet18(N).to(device); loader=DataLoader(Subset(train,parts[chosen['client_id']]),batch_size=128,shuffle=False); stats=collect_class_statistics(model,loader,device,'layer4',-1); bank,details=aggregate_class_statistics([stats],0,1,N,'skip'); feature_dim=int(bank['prototypes'].shape[1]) if bank else None; valid=len(details['classes_covered']); missing=details['missing_classes']; input_shape=tuple(next(iter(loader))[0].shape[1:]); target_count=sum(x==0 for x in test.targets); non_target=len(test)-target_count
 enforce_strict_prototype_coverage(bank,num_classes=N,dataset='gtsrb',prototype_source_client=chosen['client_id'],center_batches=-1)
 zero_loader=DataLoader(Subset(train,parts[0]),batch_size=128,shuffle=False); zero_stats=collect_class_statistics(model,zero_loader,device,'layer4',-1); zero_bank,_=aggregate_class_statistics([zero_stats],0,1,N,'skip'); negative_pass=False
 try: enforce_strict_prototype_coverage(zero_bank,num_classes=N,dataset='gtsrb',prototype_source_client=0,center_batches=-1)
 except RuntimeError: negative_pass=True
 if not negative_pass: raise RuntimeError('strict coverage negative test failed: client0 did not raise')
 proto={'prototype_source_client':chosen['client_id'],'num_classes':N,'valid_classes':valid,'missing_classes':';'.join(map(str,missing)),'feature_layer':'layer4','feature_dim':feature_dim,'center_batches_setting':-1,'use_full_local_loader':True,'samples_per_class':str([stats['class_counts'].get(k,0) for k in range(N)]),'status':'pass' if valid==N and negative_pass else 'coverage_warning'}
 with (OUT/'GTSRB_prototype_bank_precheck.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(proto));w.writeheader();w.writerow(proto)
 asr={'dataset':'GTSRB','num_classes':N,'target_label':0,'test_size':len(test),'num_target_label_samples':target_count,'num_non_target_samples':non_target,'asr_eval_excludes_target':True,'delta_shape':str(input_shape),'input_shape':str(input_shape),'status':'pass'}
 with (OUT/'GTSRB_asr_eval_precheck.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(asr));w.writeheader();w.writerow(asr)
 recommendation='yes' if valid==N else 'no'; report=f'''# GTSRB precheck\n\n- Load: pass; train={len(train)}, test={len(test)}, num_classes=43.\n- ResNet18 output classes: {model.fc.out_features}; normalization={spec['mean']}/{spec['std']}; input shape={input_shape}.\n- target_label=0; test non-target samples={non_target}.\n- IID partition: pass. Coverage: client0={rows[0]['covered_classes']}/43, client1={rows[1]['covered_classes']}/43, client2={rows[2]['covered_classes']}/43.\n- Recommended prototype source: client {chosen['client_id']}; 43/43={valid==N}.\n- Full local loader: supported using center_batches=-1. Prototype bank layer4 feature_dim={feature_dim}; valid_classes={valid}/43; samples_per_class={proto['samples_per_class']}.\n- Formal CE-UAP/PBU recommended: {recommendation}.\n- Draft configurations: configs/final_gtsrb_cross_dataset_60_150_r350/gtsrb_ce_uap_s60_e150_r350.yaml and gtsrb_pbu_s60_e150_r350.yaml.\n''';(OUT/'GTSRB_PRECHECK_REPORT.md').write_text(report);(OUT/'README_where_to_open.txt').write_text('Open GTSRB_PRECHECK_REPORT.md first.\n');print(report)
if __name__=='__main__':main()
