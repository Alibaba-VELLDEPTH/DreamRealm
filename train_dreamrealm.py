"""Single-GPU physical-machine trainer for DreamRealm whole-family objectives."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import torch
import transformers
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from joint_dpo.trainability import configure_non_lora_trainability
from dreamrealm.encoding import FamilyEncoder, score_cell
from dreamrealm.engine import backward_family, CPUAdamW
from dreamrealm.prepare import read_jsonl, validate_family
from dreamrealm.checkpoint_io import save_parts, load_parts


def probe(model):
    candidates = [(n,p) for n,p in model.named_parameters() if p.requires_grad and ('merger' in n or 'layers.0.' in n or 'lm_head' in n)]
    return {n:p.detach().flatten()[:2048].float().cpu().clone() for n,p in candidates}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--media-root',default='/jindofs_temp');p.add_argument('--max-length',type=int,default=6144)
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--tau',type=float,default=1.0);p.add_argument('--row-weight',type=float,default=.5);p.add_argument('--col-weight',type=float,default=.5)
    p.add_argument('--lr',type=float,default=5e-6);p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--lr-scheduler',choices=['constant','cosine'],default='constant')
    p.add_argument('--warmup-ratio',type=float,default=0.)
    p.add_argument('--max-grad-norm',type=float,default=1.)
    p.add_argument('--checkpoint-reentrant',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--families-per-step',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=8);p.add_argument('--gpu-memory-fraction',type=float,default=.72)
    p.add_argument('--ordered',action='store_true');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--save-model',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--save-optimizer',action='store_true');p.add_argument('--resume',help='Checkpoint directory created with --save-optimizer')
    a=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1')) != 1:raise ValueError('This physical-machine entry supports one GPU; do not launch via torchrun')
    if a.families_per_step<1 or a.epochs<1 or a.max_steps<0:raise ValueError('Invalid training limits')
    if not 0<=a.warmup_ratio<=1 or a.max_grad_norm<=0:raise ValueError('Invalid scheduler/gradient clipping')
    if a.tau<=0 or min(a.row_weight,a.col_weight)<0 or not all(math.isfinite(v) for v in (a.tau,a.row_weight,a.col_weight,a.lr)):raise ValueError('Invalid loss/lr')
    torch.set_num_threads(a.cpu_threads);random.seed(a.seed);torch.manual_seed(a.seed)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    data=[validate_family(f) for f in read_jsonl(a.data)]
    if not data or len({f['family_id'] for f in data})!=len(data):raise ValueError('Empty data or duplicate families')
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,max_pixels=a.max_pixels)
    encoder=FamilyEncoder(processor,a.max_length,a.media_root)
    if a.preflight_only:
        audit=[]
        for f in data:
            enc=encoder.encode(f)
            audit.append(dict(family_id=f['family_id'],shape=enc['shape'],target_token_counts=enc['target_token_counts'],
                              max_sequence_length=max(c['inputs']['input_ids'].shape[1] for c in enc['cells'])))
        (output/'preflight.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('Real training requires CUDA')
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    device=torch.device('cuda:0')
    manifest=dict(arguments=vars(a),data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,transformers_version=transformers.__version__,
                  families=len(data),edited_families=sum(bool(f['edited_pairs']) for f in data),
                  objective='full-answer SFT + full-family row/column CE; reference-free; two-pass exact first-order gradients')
    (output/'run_config.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Loading actual Qwen3.5-9B policy',flush=True)
    source=str(Path(a.resume)/'model') if a.resume else a.model
    model=Qwen3_5ForConditionalGeneration.from_pretrained(source,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    if hasattr(model.config,'text_config'):model.config.text_config.use_cache=False
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
        for attr in ('attention_dropout','hidden_dropout'):
            if hasattr(module,attr) and isinstance(getattr(module,attr),(float,int)):setattr(module,attr,0.)
    configure_non_lora_trainability(model,'full',True,False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':a.checkpoint_reentrant})
    if hasattr(model,'enable_input_require_grads'):model.enable_input_require_grads()
    model.to(device);model.train()
    trainable=sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(json.dumps({'trainable_parameters':trainable,'all_parameters':sum(x.numel() for x in model.parameters()),'mode':'full_language_and_projector; frozen_vision_backbone'}),flush=True)
    optimizer=CPUAdamW(model,a.lr,a.weight_decay)
    total_steps=a.max_steps or math.ceil(len(data)/a.families_per_step)*a.epochs
    warmup_steps=math.ceil(total_steps*a.warmup_ratio)
    scheduler=transformers.get_scheduler(a.lr_scheduler,optimizer.optimizer,
                                         num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    step=0;start_epoch=0;start_offset=0
    if a.resume:
        state=load_parts(Path(a.resume)/'optimizer_parts') if (Path(a.resume)/'optimizer_parts').exists() else torch.load(Path(a.resume)/'optimizer.pt',map_location='cpu',weights_only=True)
        if state['data_sha256']!=manifest['data_sha256']:raise ValueError('Resume data changed')
        expected={k:vars(a)[k] for k in ('tau','row_weight','col_weight','max_length','max_pixels','families_per_step','ordered','seed','lr','weight_decay')}
        if state['training_contract']!=expected:raise ValueError('Resume training contract changed')
        optimizer.load_state_dict(state['optimizer']);step=state['step'];start_epoch=state['epoch'];start_offset=state['offset']
        if state.get('schedule_contract')!=[a.lr_scheduler,a.warmup_ratio,total_steps,a.max_grad_norm,a.checkpoint_reentrant]:raise ValueError('Resume scheduler contract changed')
        scheduler.load_state_dict(state['scheduler'])
        del state
    initial=probe(model);start=time.monotonic();history=[];stop=False
    log=(output/'steps.jsonl').open('w',buffering=1)
    next_epoch=start_epoch;next_offset=start_offset
    for epoch in range(start_epoch,a.epochs):
        order=list(range(len(data)))
        if not a.ordered:random.Random(a.seed+epoch).shuffle(order)
        offset0=start_offset if epoch==start_epoch else 0
        for offset in range(offset0,len(order),a.families_per_step):
            if a.max_steps and step>=a.max_steps:stop=True;break
            batch=[data[i] for i in order[offset:offset+a.families_per_step]]
            optimizer.zero_grad();metrics=[];before=probe(model)
            for f in batch:
                enc=encoder.encode(f)
                print(json.dumps({'phase':'family_begin','step':step+1,'family_id':f['family_id'],'shape':enc['shape']}),flush=True)
                m=backward_family(enc,lambda cell:score_cell(model,cell,device),scale=1/len(batch),tau=a.tau,row_weight=a.row_weight,col_weight=a.col_weight)
                metrics.append(m);del enc
            grad_norm=torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],a.max_grad_norm,error_if_nonfinite=True)
            if float(grad_norm)<=0:raise RuntimeError('No nonzero gradient')
            print(json.dumps({'phase':'optimizer_begin','step':step+1,'grad_norm':float(grad_norm)}),flush=True)
            applied_lr=optimizer.optimizer.param_groups[0]['lr']
            optimizer.step();scheduler.step();step+=1
            after=probe(model)
            delta=max(float((after[n]-before[n]).abs().max()) for n in before)
            changed=sum(int(torch.count_nonzero(after[n]!=before[n])) for n in before)
            if changed==0 and applied_lr>0:raise RuntimeError('No observed model parameter update')
            event=dict(step=step,epoch=epoch,families=metrics,loss=sum(m['loss'] for m in metrics)/len(metrics),
                       grad_norm=float(grad_norm),learning_rate=applied_lr,probe_changed_values=changed,probe_max_abs_delta=delta,
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start)
            history.append(event);log.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
            next_epoch=epoch;next_offset=min(offset+len(batch),len(order))
        if stop:break
        next_epoch=epoch+1;next_offset=0
    log.close()
    if not history:raise RuntimeError('No training steps executed')
    final=probe(model)
    torch.save(dict(initial=initial,final=final),output/'parameter_probes.pt')
    if a.save_model:
        print('Saving model checkpoint',flush=True)
        model.save_pretrained(output/'model',safe_serialization=True,max_shard_size='1GB')
        processor.save_pretrained(output/'model')
    if a.save_optimizer:
        save_parts(dict(optimizer=optimizer.state_dict(),parameter_names=[n for n,_ in optimizer.named],data_sha256=manifest['data_sha256'],step=step,epoch=next_epoch,offset=next_offset,
                        scheduler=scheduler.state_dict(),schedule_contract=[a.lr_scheduler,a.warmup_ratio,total_steps,a.max_grad_norm,a.checkpoint_reentrant],
                        training_contract={k:vars(a)[k] for k in ('tau','row_weight','col_weight','max_length','max_pixels','families_per_step','ordered','seed','lr','weight_decay')}),output/'optimizer_parts')
    summary=dict(status='PASS',completed_steps=len(history),global_step=step,trainable_parameters=trainable,
                 peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,
                 branches=sorted({m['branch'] for e in history for m in e['families']}),
                 shapes=[m['shape'] for e in history for m in e['families']],saved_model=a.save_model,saved_optimizer=a.save_optimizer)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
