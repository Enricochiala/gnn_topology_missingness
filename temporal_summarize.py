"""Summarize only paired static forecasting runs, keeping pilots explicit."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(root):
    root=Path(root);cfg=json.loads((root/'config.json').read_text())
    file=root/'per_seed.jsonl'
    rows=[json.loads(v) for v in file.read_text().splitlines()] if file.exists() else []
    if not rows:raise ValueError('No attempted fits')
    df=pd.DataFrame(rows);df.to_csv(root/'per_seed.csv',index=False)
    if cfg['phase']=='tune':return
    from curve_summary import export_auc
    export_auc(rows,cfg,root,'rate','mae')
    expected=set(cfg['seeds']);summary=[]
    for key,part in df.groupby(['dataset','mechanism','rate','model'],dropna=False):
        good=part[part.status=='ok']; seeds=set(good.seed)
        complete=seeds==expected and len(good)==len(expected)
        record=dict(zip(['dataset','mechanism','rate','model'],key))
        record.update(expected_seeds=len(expected),successful_seeds=len(seeds),complete=complete,pilot=cfg['pilot'])
        for metric in ('mae','rmse'):
            record[f'{metric}_mean']=float(good[metric].mean()) if complete else None
            record[f'{metric}_sd']=float(good[metric].std(ddof=1)) if complete and len(good)>1 else None
        summary.append(record)
    summary=pd.DataFrame(summary);summary.to_csv(root/'summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset,part in summary.groupby('dataset'):
        models=list(dict.fromkeys(cfg['models']));fig,axes=plt.subplots(2,5,figsize=(19,7),squeeze=False)
        for ax,model in zip(axes.flat,models):
            for mechanism,curve in part[part.model==model].groupby('mechanism'):
                curve=curve.sort_values('rate');x=curve.rate.to_numpy();y=curve.mae_mean.to_numpy(dtype=float)
                sd=curve.mae_sd.to_numpy(dtype=float)
                ax.plot(x,y,marker='o',label=mechanism)
                if np.isfinite(sd).any():ax.fill_between(x,y-sd,y+sd,alpha=.15)
            ax.set_title(model);ax.set_xlabel('Additional missing rate');ax.set_ylabel('MAE');ax.legend(fontsize=7)
        fig.suptitle(dataset+(' — PILOT, not scientific comparison' if cfg['pilot'] else ' — mean ± sample SD'))
        fig.tight_layout();fig.savefig(root/f'{dataset}.png',dpi=150);fig.savefig(root/f'{dataset}.pdf');plt.close(fig)
    state=json.loads((root/'COMPLETE.json').read_text()) if (root/'COMPLETE.json').exists() else None
    text=['# Static GNN forecasting results','',
          '**PILOT: these scores are installation checks, not a model comparison.**' if cfg['pilot'] else
          '**Full campaign**; confirm complete five-seed groups before interpreting scores.','',
          f'Campaign status: {state if state is not None else "INCOMPLETE / in progress"}.','',
          'Failed fits are retained, not converted to zeros.','',
          'Incomplete groups have no aggregate score. Error bands are sample SD, not confidence intervals.','',
          summary.to_markdown(index=False, floatfmt='.5f'),'']
    (root/'REPORT.md').write_text('\n'.join(text))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);args=p.parse_args();summarize(args.out)
