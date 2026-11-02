"""One approved 5000->20000 continuation; fixed configuration, no fallback/retry."""
import argparse
import copy
import json
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from src.step05A_act import ROOT,CFG,IMAGE,STATE,setup,dataset,raw_batch,observations,load_bundle,digest,queue_reference,bundle_hashes,write_json,resources,sha
BASE=ROOT/'artifacts/step05A_train_compare/train20000_r02_20261101_01'
C=json.loads((BASE/'spec.json').read_text())
SOURCE=ROOT/C['training_source'];MODELS=ROOT/C['model_root']

def consume(kind,label):
    p=BASE/'budget.json';b=json.loads(p.read_text())
    key={'train':'train_forwards','live':'live_forwards','diagnostic':'diagnostic_forwards'}[kind]
    maximum=C['max_'+key]
    assert b[key]<maximum and sum(b[k] for k in ['train_forwards','live_forwards','diagnostic_forwards'])<C['max_total_forwards']
    b[key]+=1;b['last_forward']=dict(kind=kind,label=label)
    write_json(p,b)

def count_predictions(policy,label):
    original=policy.predict_action_chunk
    def call(batch):
        assert batch.get('action') is None
        consume('diagnostic',label)
        return original(batch)
    policy.predict_action_chunk=call

def rng(sampler):
    return dict(torch_rng=torch.get_rng_state().clone(),cuda_rng=[t.clone() for t in torch.cuda.get_rng_state_all()],
        numpy_rng=copy.deepcopy(np.random.get_state()),python_rng=random.getstate(),sampler_rng=sampler.get_state().clone())

def restore(state,sampler):
    for key in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng']:assert key in state
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng']);sampler.set_state(state['sampler_rng'])

def same_rng(a,b):
    return (torch.equal(a['torch_rng'],b['torch_rng']) and torch.equal(a['sampler_rng'],b['sampler_rng'])
        and len(a['cuda_rng'])==len(b['cuda_rng']) and all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))
        and pickle.dumps(a['numpy_rng'])==pickle.dumps(b['numpy_rng']) and a['python_rng']==b['python_rng'])

def finite_state(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_state(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_state(v) for v in value)
    if isinstance(value,float):return bool(np.isfinite(value))
    return True

def reference_check(policy,pre,post,directory,label):
    actual=queue_reference(policy,pre,post,load_file(directory/'reference_observations.safetensors'))
    expected=load_file(directory/'reference_actions.safetensors')['actions_rad']
    passed=torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
    result=dict(pass_=passed,max_error_rad=float((actual-expected).abs().max()),strict=True,local_files_only=True,observations=5,forwards=5)
    write_json(BASE/f'{label}_restoration.json',result);assert passed


def diagnostic(policy,pre,post,ds,sampler,step):
    before=rng(sampler);mode=policy.training;policy.eval();numbers=np.zeros(6);counts=np.zeros(6);entries=[]
    try:
        with torch.inference_mode():
            for i in C['diagnostic_indices']:
                batch=raw_batch(ds,[i]);policy.reset()
                predicted=policy.predict_action_chunk(pre(observations(batch)))
                action=torch.stack([post(predicted[:,t]).squeeze(0).cpu() for t in range(50)])
                target=batch['action'].squeeze(0);mask=(~batch['action_is_pad'].squeeze(0)).unsqueeze(-1).expand_as(target)
                assert torch.isfinite(action).all()
                error=(action-target).abs().double();num=(error*mask).sum(0).numpy();cnt=mask.sum(0).numpy()
                numbers+=num;counts+=cnt;entries.append(dict(index=i,mae_rad=float(error[mask].mean()),joint_mae_rad=(num/cnt).tolist()))
    finally:
        policy.reset();restore(before,sampler);policy.train(mode)
    assert same_rng(before,rng(sampler)) and policy.training==mode
    result=dict(step=step,mode='eval batch1, action-free inference; fixed training inputs',indices=C['diagnostic_indices'],
        inference_action_mae_rad=float(numbers.sum()/counts.sum()),joint_mae_rad=(numbers/counts).tolist(),entries=entries,
        rng_preserved=True,training_mode_restored=mode)
    write_json(BASE/f'diagnostic_{step}.json',result);return result


def save_checkpoint(policy,pre,post,optimizer,sampler,step,origin,diagnostics):
    target=MODELS/f'step{step}';target.mkdir(exist_ok=False)
    state=rng(sampler);mode=policy.training
    fixture=load_file(SOURCE/'reference_observations.safetensors')
    reference=queue_reference(policy,pre,post,fixture)
    policy.save_pretrained(target);pre.save_pretrained(target);post.save_pretrained(target)
    save_file(fixture,target/'reference_observations.safetensors');save_file({'actions_rad':reference.contiguous()},target/'reference_actions.safetensors')
    for name in ['statistics.json','data_audit.json']:shutil.copyfile(SOURCE/name,target/name)
    write_json(target/'training_spec.json',dict(base_config=CFG,approved_step=C,continuation_step=step))
    restore(state,sampler);policy.train(mode)
    assert same_rng(state,rng(sampler)) and policy.training==mode
    assert finite_state(policy.state_dict()) and finite_state(optimizer.state_dict())
    torch.save(dict(optimizer=optimizer.state_dict(),step=step,**state,initial_model_digest=origin),target/'resume.pt')
    result=dict(cumulative_updates=step,actual_updates_in_stage=step-5000,batch_size=4,
        after_model_digest=digest(policy),original_initial_model_digest=origin,diagnostic=diagnostics,
        rng_preserved_during_save=True,training_mode_restored=mode,resources=resources())
    write_json(target/'training_result.json',result);write_json(target/'bundle.sha256.json',bundle_hashes(target))
    print(json.dumps(dict(checkpoint=step,digest=result['after_model_digest'],mae=diagnostics['inference_action_mae_rad'])),flush=True)


def train():
    assert not (BASE/'training_started.json').exists(),'no retraining'
    write_json(BASE/'training_started.json',dict(at=time.time(),max_additional_updates=15000))
    start=time.monotonic();setup(42);assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    policy,pre,post=load_bundle(SOURCE);count_predictions(policy,'training-stage diagnosis/reference')
    assert digest(policy)==json.loads((SOURCE/'training_result.json').read_text())['after_model_digest']
    reference_check(policy,pre,post,SOURCE,'source5000')
    ds=dataset();assert len(ds)==9531 and ds.num_episodes==5
    state=torch.load(SOURCE/'resume.pt',map_location='cpu',weights_only=False)
    assert state['step']==5000 and all(k in state for k in ['optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','initial_model_digest'])
    assert finite_state(state['optimizer'])
    config=policy.config;assert config.chunk_size==50 and config.n_action_steps==10
    optcfg=config.get_optimizer_preset();optimizer=optcfg.build(policy.get_optim_params());optimizer.load_state_dict(state['optimizer'])
    assert optimizer.state_dict()['param_groups']==state['optimizer']['param_groups']
    def equal_state(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.detach().cpu())
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
        return a==b
    assert equal_state(optimizer.state_dict(),state['optimizer']),'optimizer tensor/state mismatch'
    assert type(optimizer).__name__=='AdamW' and optcfg.grad_clip_norm==10
    assert all(g['lr']==1e-5 and g['weight_decay']==1e-4 and g['betas']==(.9,.999) and g['eps']==1e-8 for g in optimizer.param_groups)
    assert config.kl_weight==10 and config.dropout==.1 and config.pretrained_backbone_weights is None
    assert all(float(v['step'])==5000 for v in optimizer.state.values())
    sampler=torch.Generator().manual_seed(42);restore(state,sampler)
    assert same_rng(state,rng(sampler)), 'full RNG restoration mismatch'
    write_json(BASE/'resume_check.json',dict(pass_=True,step=5000,optimizer='AdamW',state_entries=len(optimizer.state),
        all_optimizer_steps=5000,optimizer_groups_exact=True,optimizer_all_states_exact=True,rng_exact=True,rng_keys_restored=list(rng(sampler)),dataset_frames=len(ds),episodes=ds.num_episodes))
    policy.train();initial_digest=digest(policy);d0=diagnostic(policy,pre,post,ds,sampler,5000)
    baseline=json.loads((ROOT/C['baseline']/'diagnostic_5000.json').read_text())
    assert d0['inference_action_mae_rad']==baseline['inference_action_mae_rad'] and d0['joint_mae_rad']==baseline['joint_mae_rad'],'source diagnostic changed'
    restore(state,sampler);policy.train()
    expected_sampler=torch.Generator();expected_sampler.set_state(state['sampler_rng'])
    expected_first=torch.randint(len(ds),(4,),generator=expected_sampler).tolist()
    last=5000;first_ids=None;all_diag=[d0]
    with (BASE/'updates.jsonl').open('x') as log:
        for step in range(5001,20001):
            assert time.monotonic()-start<3480,'training finalization reserve'
            update_start=time.monotonic()
            ids=torch.randint(len(ds),(4,),generator=sampler).tolist()
            if step==5001:
                first_ids=ids;assert ids==expected_first
                assert same_rng({**state,'sampler_rng':sampler.get_state()},rng(sampler)), 'global RNG drift before first sample'
            batch=pre(raw_batch(ds,ids));assert policy.training
            optimizer.zero_grad(set_to_none=True);consume('train',str(step));loss,parts=policy(batch)
            assert torch.isfinite(loss);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),optcfg.grad_clip_norm,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();last=step
            assert finite_state(policy.state_dict()) and finite_state(optimizer.state_dict())
            b=json.loads((BASE/'budget.json').read_text());b['optimizer_updates']+=1;b['last_step']=step;write_json(BASE/'budget.json',b)
            row=dict(step=step,additional_update=step-5000,batch_size=4,sample_indices=ids,loss=float(loss.detach()),**parts,grad_norm=float(grad),finite_weights_optimizer=True,update_wall_seconds=time.monotonic()-update_start,wall_elapsed_seconds=time.monotonic()-start)
            log.write(json.dumps(row)+'\n');log.flush()
            if step%100==0:print(json.dumps(row),flush=True)
            if step in [10000,20000]:
                diag=diagnostic(policy,pre,post,ds,sampler,step);all_diag.append(diag)
                save_checkpoint(policy,pre,post,optimizer,sampler,step,state['initial_model_digest'],diag)
        assert last==20000 and digest(policy)!=initial_digest
    write_json(BASE/'training_result.json',dict(pass_=True,first_step=5000,last_step=last,optimizer_updates=15000,
        source_digest=initial_digest,final_digest=digest(policy),first_sample_indices=first_ids,expected_first_indices=expected_first,
        diagnostics=all_diag,resources=resources(),wall_seconds=time.monotonic()-start,additional_series=1))
    print(json.dumps(dict(training_complete=True,step=last,wall_seconds=time.monotonic()-start)),flush=True)


def reload_final():
    assert json.loads((BASE/'training_result.json').read_text())['last_step']==20000
    assert not (BASE/'reload_pass.json').exists()
    setup(42);results=[]
    for step in [10000,20000]:
        directory=MODELS/f'step{step}';policy,pre,post=load_bundle(directory);count_predictions(policy,f'independent reload{step}')
        expected=json.loads((directory/'training_result.json').read_text())['after_model_digest']
        assert digest(policy)==expected and finite_state(policy.state_dict())
        for name in ["config.json","policy_postprocessor.json","policy_postprocessor_step_0_unnormalizer_processor.safetensors","policy_preprocessor.json","policy_preprocessor_step_3_normalizer_processor.safetensors","statistics.json","data_audit.json"]:
            assert sha(directory/name)==sha(SOURCE/name),f'fixed configuration/statistics changed: {name}'
        reference_check(policy,pre,post,directory,f'checkpoint{step}')
        results.append(dict(step=step,digest=expected,finite=True,all_hashes_verified=True,configuration_statistics_unchanged=True))
        del policy,pre,post
        import gc
        gc.collect();torch.cuda.empty_cache()
    write_json(BASE/'reload_pass.json',dict(pass_=True,checkpoints=results,resources=resources()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','reload']);args=parser.parse_args()
    train() if args.mode=='train' else reload_final()
