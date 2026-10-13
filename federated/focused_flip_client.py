"""Isolated malicious client for AAAI-23 Focused-Flip (F3BA)."""
from __future__ import annotations

import copy
from typing import ClassVar

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from attacks.focused_flip import FocusedFlip, apply_pattern
from federated.client import Client


class FocusedFlipClient(Client):
    """Official 80/20 handcraft split, focused flips, then 3-epoch mixed loss."""
    client_patterns: ClassVar[dict[int, torch.Tensor]] = {}
    frozen_eval_pattern: ClassVar[torch.Tensor | None] = None
    frozen_round: ClassVar[int | None] = None

    @classmethod
    def reset_shared_state(cls) -> None:
        cls.client_patterns, cls.frozen_eval_pattern, cls.frozen_round = {}, None, None

    @classmethod
    def evaluation_pattern(cls) -> torch.Tensor | None:
        if cls.frozen_eval_pattern is not None: return cls.frozen_eval_pattern
        if not cls.client_patterns: return None
        return torch.stack([p.cpu() for _,p in sorted(cls.client_patterns.items())]).mean(0)

    @classmethod
    def freeze_evaluation_pattern(cls, round_index: int) -> None:
        pattern = cls.evaluation_pattern()
        if pattern is None:
            raise RuntimeError("cannot freeze F3BA trigger before any malicious client produced one")
        cls.frozen_eval_pattern = pattern.detach().cpu().clone()
        cls.frozen_round = round_index

    @classmethod
    def restore_frozen_pattern(cls, pattern: torch.Tensor, round_index: int) -> None:
        if not isinstance(pattern, torch.Tensor) or tuple(pattern.shape) != (3, 32, 32):
            raise ValueError("F3BA frozen trigger must be a [3, 32, 32] tensor")
        cls.frozen_eval_pattern = pattern.detach().cpu().clone()
        cls.frozen_round = round_index

    def __init__(self, *args, attack_config, **kwargs) -> None:
        super().__init__(*args, **kwargs); self.attack_config=attack_config
        self.is_malicious = True
        n=len(self.indices); split=int(.8*n)
        self.train_loader=DataLoader(Subset(self.local_dataset, range(split)),batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
        self.handcraft_loader=DataLoader(Subset(self.local_dataset, range(split,n)),batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
        self.focused=FocusedFlip(conv_rate=attack_config.f3ba_conv_rate,fc_rate=attack_config.f3ba_fc_rate,flip_factor=attack_config.f3ba_flip_factor,trigger_step=attack_config.f3ba_trigger_step,trigger_epochs=attack_config.f3ba_trigger_epochs)

    def local_train(self, global_model: nn.Module, round_index: int, neurotoxin_masks=None, effective_local_epochs: int | None=None) -> dict:
        active=self.attack_config.is_attack_round(round_index)
        if not active:
            if round_index == self.attack_config.attack_end_round + 1 and FocusedFlipClient.frozen_eval_pattern is None:
                FocusedFlipClient.freeze_evaluation_pattern(self.attack_config.attack_end_round)
            result = super().local_train(global_model,round_index,neurotoxin_masks,effective_local_epochs)
            result.update({"is_malicious": True, "attack_active": False, "num_poisoned_samples": 0})
            return result
        local=copy.deepcopy(global_model).to(self.device); local.train(); masks=self.focused.initialize_or_select(global_model)
        diag=dict(self.focused.last_diagnostics); diag["trigger_optimization_steps"]=0; diag["trigger_optimization_time_sec"]=0.0
        diag["bn_mode"] = self.attack_config.f3ba_bn_mode
        if masks is not None:
            # Match the official handcraft sequence: enter eval mode first, then
            # trigger_loss switches to train mode and injection keeps that mode.
            local.eval()
            diag.update(self.focused.optimize_trigger(
                local, masks, self.handcraft_loader, self.device,
                official_bn_mode=(self.attack_config.f3ba_bn_mode == "official_compatible"),
            ))
            diag.update(self.focused.inject(local,masks,self.handcraft_loader,self.device)); local.train()
        elif self.focused.pattern is None:
            from attacks.focused_flip import make_pattern
            self.focused.pattern,self.focused.mask=make_pattern(self.device,next(local.parameters()).dtype)
        criterion=nn.CrossEntropyLoss(); opt=torch.optim.SGD(local.parameters(),lr=self.learning_rate,momentum=self.momentum,weight_decay=self.weight_decay)
        total=0.;count=0; poisoned_samples=0
        for _ in range(effective_local_epochs or self.local_epochs):
            for images,labels in self.train_loader:
                images,labels=images.to(self.device),labels.to(self.device); n=max(1,int(images.size(0)*self.attack_config.poison_ratio)); poisoned=apply_pattern(images[:n],self.focused.pattern,self.focused.mask); opt.zero_grad(); normal=criterion(local(images),labels); backdoor=criterion(local(poisoned),torch.full((n,),self.attack_config.target_label,device=self.device,dtype=labels.dtype)); loss=self.attack_config.f3ba_normal_scale*normal+self.attack_config.f3ba_backdoor_scale*backdoor; loss.backward(); opt.step(); total+=float(loss.item())*labels.size(0); count+=labels.size(0); poisoned_samples+=n
        FocusedFlipClient.client_patterns[self.client_id]=self.focused.pattern.detach().cpu().clone()
        state={n:t.detach().cpu().clone() for n,t in local.state_dict().items()}
        diag["handcraft_train_samples"] = len(self.train_loader.dataset)
        diag["handcraft_holdout_samples"] = len(self.handcraft_loader.dataset)
        return {"client_id":self.client_id,"state_dict":state,"num_samples":len(self.local_dataset),"train_loss":total/count,"is_malicious":True,"attack_active":True,"num_poisoned_samples":poisoned_samples,"f3ba":diag}
