"""Paired static-GNN forecasting campaigns; validation-only tuning and resume.

Use --phase tune to select configurations without computing test predictions.
Use --phase test --selection DIR for untouched final-seed evaluations.
--phase reference evaluates fixed historical hyperparameters without tuning.
Any --max-windows / --epochs override is recorded as a pilot, not a full run.
"""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import copy,hashlib,json,platform,random,time,traceback
import importlib.metadata
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from temporal_data import DATASETS,StaticWindows,sha256
from temporal_models import StaticRegressor,SpatialBatch
from experiments import MAIN_MODELS
from models import PCFI, FeaturePropagation
from fisf import FISF
from temporal_imputation import DistanceLookup, FastPCFI, FastFISF

ROOT=Path(__file__).resolve().parent
CONTROLLED=('engrad','pems_bay','graphmso')
RATES=[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99]


def write_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def array_hash(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape,x.dtype)).encode()+x.tobytes()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)




def reference_config(model):
    hidden=128 if model in ('gnnzero','gnnmi','gnnmedian','gnnmim') else 64
    if model in ('gcnmf','gcnmf_pe'): hidden=16
    return dict(hidden=hidden,dropout=.5,lr=.01,
                k=5,q=8,likelihood='legacy' if model in ('gcnmf','gcnmf_pe') else 'not_applicable')


def candidates(model):
    ref=reference_config(model)
    if model=='gcnmf_pe':
        # Full observed Gaussian density implements the manuscript exactly.
        grid=[dict(ref,k=k,q=q,likelihood='gaussian') for k in (2,3,5) for q in (4,8,16)]
        grid += [dict(ref,lr=lr) for lr in (.001,.003,.01)]
    elif model=='gcnmf':
        # Offer the same density correction to the no-PE comparator: improvements
        # must not be attributable solely to fixing SPAR's shared likelihood bug.
        grid=[dict(ref,likelihood=l,lr=lr,dropout=p) for l in ('legacy','gaussian')
              for lr in (.001,.003,.01) for p in (0.,.5)]
    else:
        grid=[dict(ref,lr=lr,dropout=p) for lr in (.001,.003,.01) for p in (0.,.1,.3,.5)]
    return grid


def impute_windows(x,method,edges,weights,device="cpu"):
    # Retained imputers interpret sparse rows as receivers; PyG uses sources.
    incoming=np.asarray(edges)[::-1].copy()
    out=[]; edge=torch.as_tensor(incoming,device=device);weight=torch.as_tensor(weights,device=device)
    lookup=DistanceLookup(incoming,x.shape[1]) if method in ('pcfi','fisf') else None
    imputer=FastPCFI(lookup,40,.9,1.) if method=='pcfi' else FastFISF(lookup,40,.5,.1,.5) if method=='fisf' else None
    for row in x:
        v=torch.as_tensor(row.copy(),device=device);observed=~v.isnan()
        if method=='fp':
            filled=FeaturePropagation(40).propagate(v,edge,observed,None,weight)
        elif method=='pcfi':
            # Feature-level masks require the column-specific uniform branch.
            filled=imputer.propagate(v,edge,observed,'uniform',weight)
        elif method=='fisf':
            v=torch.nan_to_num(v)
            if v.std(dim=0).max()<1e-8:filled=v
            else:
                np.random.seed(0)
                filled=imputer.propagate(v,edge,observed,'uniform',weight)
        else:raise ValueError(method)
        if not torch.isfinite(filled).all():raise ValueError(f'{method} produced non-finite input')
        out.append(filled.cpu().numpy())
    return np.stack(out)


def selected_indices(windows,max_windows):
    if max_windows is None:return windows.splits
    # Even chronological coverage, chosen without values/labels/scores.
    return [v[np.unique(np.linspace(0,len(v)-1,min(max_windows,len(v)),dtype=int))] for v in windows.splits]


def make_inputs(windows,indices,model,data,cache,phase,device="cpu"):
    result=[]
    count=2 if phase=='tune' else 3
    train_x,_,_=windows.get(indices[0])
    flat=train_x.reshape(-1,train_x.shape[-1])
    stats={}
    if model=='gnnmi':
        mean=np.nan_to_num(np.nanmean(flat,axis=0),nan=0.);stats['mean']=array_hash(mean)
    if model=='gnnmedian':
        median=np.nan_to_num(np.nanmedian(flat,axis=0),nan=0.);stats['median']=array_hash(median)
    del flat,train_x
    for split,idx in enumerate(indices[:count]):
        x,y,valid=windows.get(idx)
        if model=='gnnmi':x=np.where(np.isnan(x),mean,x)
        elif model=='gnnmedian':x=np.where(np.isnan(x),median,x)
        elif model=='gnnzero':x=np.nan_to_num(x,nan=0.)
        elif model=='gnnmim':x=np.concatenate([np.nan_to_num(x,nan=0.),np.isnan(x).astype('float32')],axis=-1)
        elif model in ('fp','pcfi','fisf'):
            imputer=model
            key=hashlib.sha256((array_hash(x)+array_hash(data['edge_index'])+
                array_hash(data['edge_weight'])+imputer+str(device)+sha256(ROOT/'models.py')+
                sha256(ROOT/'fisf.py')+sha256(ROOT/'filling_strategies.py')+
                sha256(Path(__file__))+sha256(ROOT/'temporal_imputation.py')).encode()).hexdigest()
            file=cache/f'{key}.npy'
            cache_manifest=file.with_suffix('.json')
            if file.exists():
                if not cache_manifest.exists() or json.loads(cache_manifest.read_text())['sha256']!=sha256(file):
                    raise ValueError('Imputation cache changed or is incomplete')
                x=np.load(file,mmap_mode='r')
            else:
                sensor_width=windows.window*windows.values.shape[-1]
                recovered=impute_windows(x[:,:,:sensor_width],imputer,data['edge_index'],data['edge_weight'],device)
                x=np.concatenate([recovered,x[:,:,sensor_width:]],axis=-1)
                np.save(file,x)
                write_json(cache_manifest,{'sha256':sha256(file)})
        result.append((x,y,valid))
    return result,stats


def gmm_init(x,pe,q,max_rows=4096):
    flat=x.reshape(-1,x.shape[-1]);idx=np.unique(np.linspace(0,len(flat)-1,min(max_rows,len(flat)),dtype=int))
    if q:return np.concatenate([flat[idx],pe[idx%x.shape[1],:q]],1)
    return flat[idx]


def metrics(model,split,center,scale,batch_size):
    x,y,valid=split; model.eval()
    channels=len(center); horizon=y.shape[-1]//channels
    total_abs=np.zeros((horizon,channels));total_sq=total_abs.copy();count=total_abs.copy()
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            pred=model(torch.as_tensor(x[start:start+batch_size],device=model.graph.device)).cpu().numpy()
            target=y[start:start+batch_size].reshape(*pred.shape[:2],horizon,channels)
            mask=valid[start:start+batch_size].reshape(target.shape)
            pred=pred.reshape(target.shape)*scale+center
            if not np.isfinite(pred).all():raise ValueError('Non-finite prediction')
            error=np.where(mask,pred-target,0.)
            total_abs+=np.abs(error).sum((0,1));total_sq+=(error**2).sum((0,1));count+=mask.sum((0,1))
    if count.sum()==0:raise ValueError('No observed targets in evaluated split')
    def safe_averages(num,den):
        return [float(a/b) if b>0 else None for a,b in zip(num,den)]
    return dict(mae=float(total_abs.sum()/count.sum()),rmse=float(np.sqrt(total_sq.sum()/count.sum())),
        mae_per_channel=safe_averages(total_abs.sum(0),count.sum(0)),
        mae_per_horizon=safe_averages(total_abs.sum(1),count.sum(1)),n_targets=int(count.sum()))


def fit(model_name,config,seed,arrays,data,windows,args,checkpoint,embedding=None):
    seed_all(seed)
    graph=SpatialBatch(data['edge_index'],data['edge_weight'],data['values'].shape[1],args.device)
    q=config['q'] if model_name=='gcnmf_pe' else 0
    init=gmm_init(arrays[0][0],data['pe'],q)
    model=StaticRegressor(model_name,arrays[0][0].shape[-1],arrays[0][1].shape[-1],graph,data['pe'],init,config,embedding)
    model=model.to(args.device)
    # Disables the debug anomaly hook enabled globally by the original models.
    torch.autograd.set_detect_anomaly(False)
    optim=torch.optim.Adam(model.parameters(),lr=config['lr'])
    maximum=args.epochs or (500 if model_name in ('gnnzero','gnnmi','gnnmedian','gnnmim','gcnmf_pe') else 1000)
    patience=args.patience or (50 if maximum==500 else 40)
    center=torch.as_tensor(np.tile(windows.center,windows.horizon),device=args.device)
    scale=torch.as_tensor(np.tile(windows.scale,windows.horizon),device=args.device)
    x,y,valid=arrays[0]
    best=float('inf');best_state=None;stale=0;best_epoch=None;history=[]
    rng=np.random.default_rng(seed)
    start_epoch=0
    last=checkpoint.with_suffix('.last.pt')
    if last.exists():
        saved=torch.load(last,map_location=args.device,weights_only=False)
        model.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer'])
        best=saved['best'];best_state=saved['best_state'];best_epoch=saved['best_epoch']
        stale=saved['stale'];history=saved['history'];start_epoch=saved['epoch']+1
        rng.bit_generator.state=saved['rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    if stale>=patience:start_epoch=maximum
    for epoch in range(start_epoch,maximum):
        model.train();order=rng.permutation(len(x));train_loss=0.;steps=0
        for start in range(0,len(x),args.batch_size):
            ids=order[start:start+args.batch_size]
            xx=torch.as_tensor(x[ids],device=args.device)
            yy=torch.as_tensor(y[ids],device=args.device); vv=torch.as_tensor(valid[ids],device=args.device)
            if not vv.any():continue
            optim.zero_grad(set_to_none=True)
            prediction=model(xx)*scale+center
            loss=(prediction[vv]-yy[vv]).abs().mean()
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step()
            train_loss+=float(loss.detach());steps+=1
        val=metrics(model,arrays[1],windows.center,windows.scale,args.batch_size)
        history.append({'epoch':epoch+1,'training_loss':train_loss/max(steps,1),'val_mae':val['mae']})
        if val['mae']<best:
            best=val['mae'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_epoch=epoch+1;stale=0
        else:stale+=1
        if (epoch+1)%10==0:
            temporary=last.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),optimizer=optim.state_dict(),best=best,best_state=best_state,
                best_epoch=best_epoch,stale=stale,history=history,epoch=epoch,rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else []),temporary)
            temporary.replace(last)
            write_json(checkpoint.with_suffix('.history.json'),history)
            print(f'epoch={epoch+1} train_loss={train_loss/max(steps,1):.6f} validation_mae={val["mae"]:.6f}',flush=True)
        if stale>=patience:break
    if best_state is None:raise ValueError('No valid checkpoint')
    torch.save({'state_dict':best_state,'config':config,'seed':seed,'best_epoch':best_epoch},checkpoint)
    write_json(checkpoint.with_suffix('.history.json'),history)
    model.load_state_dict(best_state)
    record=dict(validation_mae=best,best_epoch=best_epoch,epochs_run=len(history))
    if args.phase!='tune':record.update(metrics(model,arrays[2],windows.center,windows.scale,args.batch_size))
    if last.exists():last.unlink()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/temporal/prepared_v2'))
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--models',nargs='+',choices=MAIN_MODELS+('all',),default=['all'])
    p.add_argument('--rates',nargs='+',type=float,default=RATES)
    p.add_argument('--mechanisms',nargs='+',choices=['RT','UMCAR'],default=['RT','UMCAR'])
    p.add_argument('--seeds',nargs='+',type=int,default=[1,43,15,118,222])
    p.add_argument('--phase',choices=['reference','tune','test'],default='reference')
    p.add_argument('--selection',type=Path)
    p.add_argument('--preset',type=Path,help='Frozen completed selection supplied in configs/presets')
    p.add_argument('--extended-pemix',action='store_true',help='Use the 48-candidate PEMix search; other models keep 12 candidates')
    p.add_argument('--mask-cache',type=Path,default=Path('cache/rt_masks'))
    p.add_argument('--epochs',type=int);p.add_argument('--patience',type=int)
    p.add_argument('--max-windows',type=int,help='Pilot only; deterministic coverage per split')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--stride',type=int,help='Default horizon; explicit 1 recovers overlapping forecasting windows')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.models=list(dict.fromkeys(m for a in args.models for m in (MAIN_MODELS if a=='all' else [a])))
    args.requested_models=list(args.models)
    for field in ('datasets','rates','mechanisms','seeds'):setattr(args,field,list(dict.fromkeys(getattr(args,field))))
    if any(not np.isfinite(r) or not 0<=r<=1 for r in args.rates):p.error('Invalid missing rate')
    for field in ('epochs','patience','max_windows','batch_size','stride','threads'):
        value=getattr(args,field)
        if value is not None and value<1:p.error(f'{field} must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA requested but unavailable')
    if any(seed<0 for seed in args.seeds):p.error('Seeds must be nonnegative')
    if not args.models:p.error('No regression-compatible model selected')
    if args.phase=='tune' and args.seeds!=[2026]:p.error('Tuning uses --seeds 2026; reserve the five final seeds')
    if args.selection and args.preset:p.error('Use only one of --selection and --preset')
    if args.preset and args.phase!='test':p.error('--preset requires --phase test')
    if args.phase=='test' and args.selection is None and args.preset is None:p.error('Final tuned test requires --selection or --preset')
    if args.phase!='tune' and 2026 in args.seeds:p.error('2026 is reserved for tuning')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    run_lock=(args.out/'run.lock').open('a')
    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('Another process is already writing this output directory')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='out'}
    config['pilot']=args.max_windows is not None or args.epochs is not None
    config['search_space_sha256']=sha256(ROOT/'configs/extended_search.json')
    config['preset_sha256']=sha256(args.preset) if args.preset else None
    config['source_hashes']={v.name:sha256(v) for v in sorted(v for v in ROOT.iterdir() if v.suffix in ('.py','.cpp'))}
    config['input_hashes']={v:sha256(args.data_dir/f'{v}.npz') for v in args.datasets}
    config['versions']={v:importlib.metadata.version(v) for v in ('torch','torch-geometric','numpy','scipy','pandas','scikit-learn','gensim')}
    config['python']=platform.python_version()
    config['device_name']=torch.cuda.get_device_name(0) if args.device=='cuda' else platform.processor()
    config['cuda_runtime']=torch.version.cuda
    cp=args.out/'config.json'
    if cp.exists():
        if json.loads(cp.read_text())!=config:p.error('Resume configuration/source/data/environment mismatch; choose a new --out')
    else:
        if any(v.name!='run.lock' for v in args.out.iterdir()):p.error('New output directory must be empty')
        write_json(cp,config)
    for part in ('logs','checkpoints','masks','cache'):(args.out/part).mkdir(exist_ok=True)
    records_path=args.out/'per_seed.jsonl'
    records=[json.loads(v) for v in records_path.read_text().splitlines()] if records_path.exists() else []
    keys=('dataset','mechanism','rate','model','seed','candidate')
    done={tuple(v[k] for k in keys) for v in records}
    if len(done)!=len(records):raise ValueError('Duplicate result keys')
    selection=None
    if args.selection:
        complete=args.selection/'COMPLETE.json'
        if not complete.is_file() or json.loads(complete.read_text()).get('failed',1):
            raise ValueError('Selection requires a completed tuning run without failed fits')
        selection=json.loads((args.selection/'selection.json').read_text())
        sc=json.loads((args.selection/'config.json').read_text())
        if sc['source_hashes']!=config['source_hashes'] or sc['input_hashes']!=config['input_hashes']:
            raise ValueError('Selection source/input hashes mismatch')
        if sc['pilot'] and not config['pilot']:raise ValueError('Pilot hyperparameters cannot silently become full results')
        for field in ('versions','python','batch_size','stride','epochs','patience','max_windows','extended_pemix','search_space_sha256'):
            if sc[field]!=config[field]:raise ValueError(f'Selection mismatch: {field}')
    if args.preset:
        preset=json.loads(args.preset.read_text())
        for name in args.datasets:
            if preset['input_hashes'].get(name)!=config['input_hashes'][name]:
                raise ValueError('Preset input fingerprint mismatch')
        selection=preset['selection']
    torch.set_num_threads(args.threads);torch.autograd.set_detect_anomaly(False)
    expected=0
    with threadpool_limits(limits=args.threads):
        for name in args.datasets:
            with np.load(args.data_dir/f'{name}.npz') as f:data={k:f[k] for k in f.files}
            settings=[(m,r) for m in args.mechanisms for r in args.rates] if name in CONTROLLED else [('natural',0.)]
            for mechanism,rate in settings:
                for seed in args.seeds:
                    windows=StaticWindows.build(data,name,seed,rate,mechanism,args.stride,args.mask_cache)
                    indices=selected_indices(windows,args.max_windows)
                    if any(len(v)==0 for v in indices):raise ValueError('Empty temporal split')
                    tag=f'{name}_{mechanism}_{rate}_{seed}'
                    maskpath=args.out/'masks'/f'{tag}.npz'
                    metadata=dict(input_mask_hash=array_hash(windows.input_observed),
                        original_mask_hash=array_hash(windows.observed),
                        split_hash=array_hash(np.concatenate(indices)),
                        input_missing_fraction=float((~windows.input_observed).mean()),
                        scaler_center=windows.center.tolist(),scaler_scale=windows.scale.tolist())
                    mask_manifest=maskpath.with_suffix('.json')
                    if maskpath.exists():
                        if not mask_manifest.exists() or json.loads(mask_manifest.read_text())['sha256']!=sha256(maskpath):
                            raise ValueError('Saved masks/splits were modified or lack their manifest')
                    else:
                        np.savez_compressed(maskpath,input_observed=windows.input_observed,
                        original_observed=windows.observed,train_origins=indices[0],val_origins=indices[1],test_origins=indices[2],
                            center=windows.center,scale=windows.scale)
                        write_json(mask_manifest,dict(sha256=sha256(maskpath),**metadata))
                    for model in args.models:
                        configs=candidates(model) if args.phase=='tune' else [reference_config(model)]
                        if args.phase=='tune' and args.extended_pemix and model=='gcnmf_pe':
                            from extended_search import pemix_candidates
                            configs=pemix_candidates()
                        if selection is not None:
                            configs=[selection[f'{name}/{mechanism}/{rate}/{model}']['config']]
                        expected+=len(configs)
                        pending=[(i,c) for i,c in enumerate(configs) if (name,mechanism,rate,model,seed,i) not in done]
                        if not pending:continue
                        logfile=args.out/'logs'/f'{tag}_{model}.log'
                        with logfile.open('a') as log,redirect_stdout(log),redirect_stderr(log):
                            prep_error=None
                            try:
                                seed_all(seed)
                                prep_started=time.perf_counter()
                                arrays,stats=make_inputs(windows,indices,model,data,args.out/'cache',args.phase,args.device)
                                embedding=None
                                preprocessing_seconds=time.perf_counter()-prep_started
                            except Exception as e:prep_error=e;traceback.print_exc();preprocessing_seconds=time.perf_counter()-prep_started
                            for candidate,cfg in pending:
                                row=dict(dataset=name,mechanism=mechanism,rate=rate,model=model,seed=seed,candidate=candidate,
                                         config=cfg,pilot=config['pilot'],phase=args.phase,**metadata)
                                start=time.perf_counter()
                                try:
                                    if prep_error is not None:raise prep_error
                                    row['imputation_statistics_hashes']=stats
                                    result=fit(model,cfg,seed,arrays,data,windows,args,
                                        args.out/'checkpoints'/f'{tag}_{model}_{candidate}.pt',embedding)
                                    row.update(result,status='ok',error=None)
                                except Exception as exc:
                                    traceback.print_exc();row.update(status='failed',error=f'{type(exc).__name__}: {exc}',mae=None,rmse=None)
                                row['seconds']=time.perf_counter()-start
                                row['preprocessing_seconds']=preprocessing_seconds
                                with records_path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                                records.append(row);done.add(tuple(row[k] for k in keys))
                        print(f'{tag} {model}: '+', '.join(r['status'] for r in records[-len(pending):]),flush=True)
    if len(done)!=expected:raise ValueError(f'Unexpected fit count: {len(done)} != {expected}')
    failed=sum(r['status']!='ok' for r in records)
    if args.phase=='tune':
        chosen={}
        for row in records:
            if row['status']!='ok':continue
            key=f'{row["dataset"]}/{row["mechanism"]}/{row["rate"]}/{row["model"]}'
            if key not in chosen or row['validation_mae']<chosen[key]['validation_mae']:
                chosen[key]={k:row[k] for k in ('config','validation_mae','candidate')}
        write_json(args.out/'selection.json',chosen)
    write_json(args.out/'COMPLETE.json',dict(attempted=len(done),failed=failed,pilot=config['pilot'],phase=args.phase))
    raise SystemExit(bool(failed))

if __name__=='__main__':main()
