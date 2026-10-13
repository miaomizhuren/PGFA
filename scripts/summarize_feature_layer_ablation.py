#!/usr/bin/env python3
"""Extract the real layer3/layer4 full-run results for thesis ablation tables."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/'experiments/summaries/mainexp/mainexp_summary.csv'; OUT=ROOT/'experiments/summaries/feature_ablation'
SPECS=(('Proto-UAP Layer3','layer3',256,'mainexp_03_proto_uap_layer3_baseline'),('Proto-UAP Layer4 Proposed','layer4',512,'mainexp_04_proto_uap_layer4_proposed'))
ACC=(100,120,130,150,180,200)
ASR=(100,110,120,130,140,150,160,170,180,190,200)
FIELDS=['feature_layer','feature_dim','method_source',*[f'CleanACC@{r}' for r in ACC],*[f'ASR@{r}' for r in ASR],'AUC@100-200','Half-life','PSNR','SSIM','L_inf']
def fmt(value, digits=2):
    try: return f'{float(value):.{digits}f}'
    except (ValueError,TypeError): return 'N/A'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source={r['method']:r for r in csv.DictReader(SOURCE.open(encoding='utf-8'))}; rows=[]
    for method,layer,dimension,origin in SPECS:
        r=source.get(method,{})
        missing=[key for key in [*[f'CleanACC@{x}' for x in ACC],*[f'ASR@{x}' for x in ASR],'post_attack_auc_100_200','half_life_round','PSNR','SSIM','L_inf'] if key not in r or r[key] in ('',None)]
        if missing: raise ValueError(f'{method} missing source fields: {missing}')
        row={'feature_layer':layer,'feature_dim':dimension,'method_source':origin}
        row.update({f'CleanACC@{round_index}':fmt(r.get(f'CleanACC@{round_index}')) for round_index in ACC})
        row.update({f'ASR@{round_index}':fmt(r.get(f'ASR@{round_index}')) for round_index in ASR})
        row.update({'AUC@100-200':fmt(r.get('post_attack_auc_100_200')),'Half-life':r.get('half_life_round','N/A'),'PSNR':fmt(r.get('PSNR')),'SSIM':fmt(r.get('SSIM'),4),'L_inf':fmt(r.get('L_inf'),4)})
        rows.append(row)
    with (OUT/'feature_layer_ablation_summary.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    lines=['# 特征层选择消融实验真实结果','', '## 表 A：完整 ASR 曲线表','', '| feature_layer | feature_dim | '+' | '.join(f'ASR@{r}' for r in ASR)+' |','|---|---:|'+'---:|'*len(ASR)]
    lines += [f"| {r['feature_layer']} | {r['feature_dim']} | "+' | '.join(r[f'ASR@{x}'] for x in ASR)+' |' for r in rows]
    lines += ['', '## 表 B：Clean ACC 表','', '| feature_layer | '+' | '.join(f'CleanACC@{r}' for r in ACC)+' |','|---|'+'---:|'*len(ACC)]
    lines += [f"| {r['feature_layer']} | "+' | '.join(r[f'CleanACC@{x}'] for x in ACC)+' |' for r in rows]
    lines += ['', '## 表 C：综合指标表','', '| feature_layer | AUC@100-200 | Half-life | PSNR | SSIM | L_inf |','|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['feature_layer']} | {r['AUC@100-200']} | {r['Half-life']} | {r['PSNR']} | {r['SSIM']} | {r['L_inf']} |" for r in rows]
    lines += ['', '## 可直接替换论文 6.1 的文本', '', '### 6.1 特征层选择分析', '', '为分析特征层选择对原型引导 UAP 后门持久性的影响，本文比较了 layer3 和 layer4 两个特征层。二者除 feature_layer 外，其余设置保持一致，均采用 ε=0.10、steps=120、poison_ratio=0.70、λ_delta=9.0 和 λ_train=5.0。在当前 ResNet18 实现中，layer3 表示第三阶段卷积输出经全局平均池化后的 256 维特征，layer4 表示最后卷积阶段输出经全局平均池化后的 512 维分类头前特征。由于 avgpool 与 penultimate 在代码实现中均对应 layer4 后的池化展平特征，因此本文不再额外设置 avgpool 或 penultimate 变体。', '', '实验结果表明，Proto-UAP(layer3) 虽然在攻击结束轮次达到 99.94% ASR，但攻击停止后表现出明显不稳定性，其 ASR@110 迅速下降至 29.61%，随后在 Round 120 短暂回升至 62.82%，但中后期继续下降，Round 150 和 Round 200 分别仅为 26.72% 和 8.99%。这说明 layer3 中层特征约束难以稳定支撑后门在后续良性训练中的保持。', '', '相比之下，Proto-UAP(layer4) 在攻击停止后的早期和中期均保持明显更高 ASR，其 ASR@110、ASR@120、ASR@130、ASR@140 和 ASR@150 分别为 99.98%、98.81%、95.03%、85.39% 和 82.58%。在整体持久性指标上，layer4 的 AUC@100–200 为 70.53，half-life round 为 163，明显高于 layer3 的 26.64 和 101。同时，二者 Clean ACC@150 分别为 84.44% 和 84.54%，L∞ 均约为 0.0262，说明性能差异并非来自主任务准确率下降或扰动幅度变化，而主要来自特征层语义表达能力的差异。', '', '该结果说明，并非任意中间层原型约束都能带来持久性增强。layer3 更偏向纹理和局部结构，其特征空间与最终类别决策之间仍存在差距；layer4 更接近分类头，包含更强的类别语义，因此基于 layer4 的 prototype alignment 更容易将触发样本稳定嵌入目标类别语义区域，从而延缓后门在后续良性训练中的遗忘。']
    (OUT/'feature_layer_ablation_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
