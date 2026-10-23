#!/usr/bin/env python3
"""Render display derivatives from saved arrays; never alter original images/data."""
import argparse,copy,csv,hashlib,importlib.util,json,sys
from pathlib import Path

def render(base,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import pandas as pd
 sys.path.insert(0,str(base/'scripts'))
 from surface_papl_figures import Figures
 r=base/'results/archie_8193723_complete';g=json.loads((r/'review_bundle.json').read_text());byid={f['figure_id']:f for f in g['figures']};out.mkdir(parents=True,exist_ok=True);derivatives={};F=Figures(r)
 def finish(fid,fig):
  rec=copy.deepcopy(byid[fid]);axes=[];outside=0
  for ax in fig.axes:
   label=ax.get_ylabel();absolute=('pAPL' in label or 'Projected APL' in label) and not any(w in label.lower() for w in ['delta','change','slope'])
   if absolute:
    ax.set_ylim(40,90);ax.set_yticks([40,50,60,70,80,90])
    for line in ax.lines:
     y=line.get_ydata();outside+=sum(float(v)<40 or float(v)>90 for v in y)
    for c in ax.collections:
     if hasattr(c,'get_offsets'):outside+=sum(float(v[1])<40 or float(v[1])>90 for v in c.get_offsets())
   if ('pAPL' in ax.get_xlabel()) and not any(w in ax.get_xlabel().lower() for w in ['delta','change','slope']):ax.set_xlim(40,90);ax.set_xticks([40,50,60,70,80,90])
   axes.append(dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel()))
  if fid in ('binned_frame_reconciliation','temporal_selected_trajectories'):
   rec['title']+=' [historical timing unresolved]'
   rec['interpretation']='Historical saved numerical comparison; exact local timing/alignment remains unresolved. No current aligned-time inference. '+rec['interpretation']
  fig.suptitle(rec['title']+' — standardised display derivative'+(' | OUTSIDE DISPLAY RANGE' if outside else ''),fontsize=11);fig.tight_layout(rect=(0,0,1,.93))
  rec['original_display']={k:rec[k] for k in ['png_path','svg_path','png_sha256','svg_sha256','axes','generator']}
  for ext in ['png','svg']:
   path=out/(fid+'_axis40_90.'+ext);fig.savefig(path,dpi=160);rec[ext+'_path']='surface_papl_analysis/figures/'+ext+'/'+rec['family']+'/'+path.name;rec[ext+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  plt.close(fig);rec['axes']={'variables':byid[fid]['axes'],'saved_axis_limits':axes,'y_standard':'Absolute projected APL 40–90 Å²; other metrics retain original scales'};rec['generator']='staging/papl_pc3_integration_update_20261009/standardise_apl_axes.py';rec['limitations']+=' Display derivative only: original arrays/figures preserved. Historical timing assumptions remain unverified where noted.';rec['display_source_hashes']={s:hashlib.sha256((r/s).read_bytes()).hexdigest() for s in rec['source_tables'] if (r/s).is_file()};rec['out_of_range_points']=outside;derivatives[fid]=rec
 p=r/'surface_papl_analysis/pristine_extraction_8202573';data={s:pd.read_csv(p/f'pristine_frames_{s}.csv').sort_values('time_ns') for s in ['B','M']}
 for fid,sides in [('pristine_papl_b',['B']),('pristine_papl_m',['M']),('pristine_papl_combined',['B','M'])]:
  fig,ax=plt.subplots(figsize=(10,5.5))
  for s in sides:ax.plot(data[s].time_ns,data[s].projected_APL_A2,color='tab:blue' if s=='B' else 'tab:orange',label=s,linewidth=1.8)
  ax.set(xlabel='Simulation time (ns)',ylabel='Projected APL (Å²/lipid)');ax.grid(alpha=.25);ax.legend();finish(fid,fig)
 d=pd.read_csv(r/'surface_papl_analysis/tables/matched_late_dataset.csv')
 for fid,x,y,data in [('endpoint_late','pAPL_endpoint_A2','pAPL_late_A2',d),('binned_frame_reconciliation','pAPL_late_binned_A2','pAPL_late_A2',d[d.binned_pAPL_available])]:
  fig,axs=plt.subplots(1,2,figsize=(11,5))
  for ax,(s,h) in zip(axs,data.groupby('side')):
   F.scatter(ax,h,x,y);ax.set_title(s+'; n='+str(len(h)))
   if fid=='binned_frame_reconciliation':ax.plot([40,90],[40,90],'k--',lw=.6)
  if fid=='binned_frame_reconciliation':byid[fid]['limitations']+=' Historical late binned scalar assumes 100 ns offset; exact matching remains unresolved; no new alignment claim.'
  finish(fid,fig)
 # Exact exported historical arrays. Standardise only pAPL columns, retain geometry scale.
 d=pd.read_csv(r/'surface_papl_analysis/tables/highlighted_temporal_trajectories.csv');fig,axs=plt.subplots(2,2,figsize=(12,8))
 for (s,h),axes in zip(d.groupby('side'),axs):
  for pid,q in h.groupby('pair_id',sort=False):
   q=q.sort_values('local_start_ns');axes[0].plot(q.local_start_ns+2.5,q.pAPL_A2,label=pid.split('|')[0]);axes[1].plot(q.local_start_ns+2.5,q.geometry_nm2,label=pid.split('|')[0])
  for ax in axes:ax.set_xlabel('Historical local bin centre (ns); offset unverified');ax.grid(alpha=.2);ax.legend(fontsize=7);ax.set_title(s)
  axes[0].set_ylabel('pAPL bin mean (Å²)');axes[1].set_ylabel('Curved minus projected area (nm²)')
 finish('temporal_selected_trajectories',fig)
 (out/'axis_derivatives.json').write_text(json.dumps(derivatives,indent=2)+'\n');return derivatives
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base.resolve();o=a.output.resolve()
 if not o.is_relative_to(b/'staging/papl_pc3_integration_update_20261009'):p.error('Output must be inside existing stage')
 render(b,o);print('STAGED_ONLY: six display derivatives; no canonical writes')
