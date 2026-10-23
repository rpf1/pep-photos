"""Paired/system-level analyses; no leaflet pseudo-replicates and no composite score."""
from itertools import combinations
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

CORE=['height_rms_nm','midplane_excess_area_fraction','H_rms_nm_inv','thickness_median_nm']
ROLE={'POPC60CHOL40_W_WF':'mammalian','POPE75POPG25_W_WF':'bacterial','POPC80POPS20_W_WF':'additional'}
PAIR=['molecule','nmol','physical_run']


def bh(values):
    p=np.asarray(values,float);q=np.full(len(p),np.nan);keep=np.flatnonzero(np.isfinite(p))
    if len(keep):
        order=keep[np.argsort(p[keep])];adj=p[order]*len(order)/np.arange(1,len(order)+1)
        q[order]=np.minimum(1,np.minimum.accumulate(adj[::-1])[::-1])
    return q


def paired_table(d,metrics):
    x=d[d.role.isin(['bacterial','mammalian'])]
    if x.duplicated(PAIR+['role']).any(): raise ValueError('Duplicate physical pair member')
    b=x[x.role=='bacterial'];m=x[x.role=='mammalian']
    p=b[PAIR+['manual_label']+metrics].merge(m[PAIR+['manual_label']+metrics],on=PAIR,suffixes=('_bacterial','_mammalian'),validate='one_to_one')
    p['label_pair']=p.manual_label_bacterial+' | '+p.manual_label_mammalian
    p['ddAPL']=p.APL_change_BOX_bacterial-p.APL_change_BOX_mammalian
    for k in metrics:
        p[k+'__bacterial_minus_mammalian']=p[k+'_bacterial']-p[k+'_mammalian']
    p['dAPL_bacterial']=p.APL_change_BOX_bacterial;p['dAPL_mammalian']=p.APL_change_BOX_mammalian
    p['abs_ddAPL']=abs(p.ddAPL);p['mean_dAPL']=(p.dAPL_bacterial+p.dAPL_mammalian)/2
    p['max_dAPL_between_membranes']=p[['dAPL_bacterial','dAPL_mammalian']].max(axis=1)
    p['min_dAPL_between_membranes']=p[['dAPL_bacterial','dAPL_mammalian']].min(axis=1)
    return p


def correlation_rows(d,columns,dataset):
    rows=[]
    for a,b in combinations(columns,2):
        x=d[[a,b]].replace([np.inf,-np.inf],np.nan).dropna();n=len(x)
        rho=p=np.nan
        if n>=3 and x[a].nunique()>1 and x[b].nunique()>1:
            rho,p=stats.spearmanr(x[a],x[b])
        rows.append({'dataset':dataset,'metric_x':a,'metric_y':b,'n':n,'spearman_rho':rho,'p_raw':p,'abs_rho':abs(rho)})
    return rows


def analyse(d,prefix):
    metrics=[x for x in ['APL_change_BOX','APL_percent_change_BOX','thickness_change']+[prefix+phase+'__'+k for phase in ['baseline','endpoint','delta'] for k in CORE] if x in d]
    for col in metrics:d[col]=pd.to_numeric(d[col],errors='coerce')
    # Separate quantitative strata; a repeated sequence within a membrane is not independent.
    if d.duplicated(['molecule','role']).any():
        raise ValueError('Repeated sequence within membrane stratum: a clustered/repeated-measures model is required')
    tables={};omnibus=[];pairwise=[];desc=[];corr=[];discord=[]
    for role,s in d.groupby('role'):
        for metric in metrics:
            groups=[];labels=[]
            for label,x in s.groupby('manual_label'):
                vals=x[metric].replace([np.inf,-np.inf],np.nan).dropna()
                desc.append({'membrane':role,'metric':metric,'class':label,'n':len(vals),'median':vals.median(),'q25':vals.quantile(.25),'q75':vals.quantile(.75)})
                if len(vals)>=(3 if metric.startswith(prefix) else 2): groups.append(vals.to_numpy());labels.append(label)
            H=p=eps=np.nan;n=sum(map(len,groups));k=len(groups)
            if k>=2 and np.unique(np.concatenate(groups)).size>1:
                H,p=stats.kruskal(*groups);eps=max(0,(H-k+1)/(n-k)) if n>k else np.nan
            omnibus.append({'membrane':role,'metric':metric,'classes_in_test':';'.join(labels),'n_total':n,'H':H,'p_raw':p,'epsilon_squared':eps})
            for (la,x),(lb,y) in combinations(zip(labels,groups),2):
                u,pval=stats.mannwhitneyu(x,y,alternative='two-sided',method='auto')
                pairwise.append({'membrane':role,'metric':metric,'class_a':la,'class_b':lb,'n_a':len(x),'n_b':len(y),'U':u,'rank_biserial_a_minus_b':2*u/(len(x)*len(y))-1,'p_raw':pval})
        corr+=correlation_rows(s,metrics,role)
        for k in CORE:
            col=prefix+'endpoint__'+k
            if col not in s:continue
            vals=s[['physical_key','molecule','manual_label','APL_change_BOX',col]].dropna().copy()
            if len(vals):
                ar=vals.APL_change_BOX.rank(pct=True);mr=vals[col].rank(pct=True)
                vals['example_reason']=np.where((ar>=.75)&(mr<=.25),'high_APL_low_morphology',np.where((ar<=.25)&(mr>=.75),'low_APL_high_morphology',''))
                vals=vals[vals.example_reason!=''];vals['metric']=col;vals['membrane']=role
                discord.extend(vals.to_dict('records'))
    pairs=paired_table(d,metrics)
    concord=[];paired_tests=[];label_tests=[];pair_corr=[]
    if len(pairs):
        tab=pd.crosstab(pairs.manual_label_mammalian,pairs.manual_label_bacterial)
        chi=p=np.nan;dof=0;v=np.nan
        if min(tab.shape)>1:
            chi,p,dof,_=stats.chi2_contingency(tab,correction=False);v=np.sqrt(chi/(tab.values.sum()*(min(tab.shape)-1)))
        concord=[{'n_pairs':len(pairs),'exact_label_agreement_fraction':float((pairs.manual_label_mammalian==pairs.manual_label_bacterial).mean()),'chi2':chi,'chi2_dof':dof,'chi2_p':p,'cramers_v':v,'note':'chi-square association is descriptive for sparse tables; exact agreement retained'}]
        tables['label_transition_counts']=tab.rename_axis('mammalian_label').reset_index()
        for k in metrics:
            x=pairs[[k+'_bacterial',k+'_mammalian']].dropna();delta=x.iloc[:,0]-x.iloc[:,1]
            w=p=np.nan
            if len(delta)>=3:
                if (delta==0).all(): w,p=0.,1.
                else: w,p=stats.wilcoxon(delta,alternative='two-sided')
            paired_tests.append({'metric':k,'n_pairs':len(delta),'median_bacterial_minus_mammalian':delta.median(),'W':w,'p_raw':p})
        groups=[x.ddAPL.dropna().to_numpy() for _,x in pairs.groupby('label_pair') if x.ddAPL.notna().sum()>=5]
        if len(groups)>=2 and np.unique(np.concatenate(groups)).size>1:
            h,p=stats.kruskal(*groups);label_tests=[{'metric':'ddAPL','H':h,'p_raw':p,'groups':len(groups),'n':sum(map(len,groups))}]
        targets=['dAPL_bacterial','dAPL_mammalian','ddAPL','abs_ddAPL','mean_dAPL','max_dAPL_between_membranes','min_dAPL_between_membranes']
        pair_corr+=correlation_rows(pairs,targets,'primary_tripeptoid_pairs')
        for k in metrics:
            if k.startswith(prefix):
                for side in ['bacterial','mammalian']:
                    pair_corr+=correlation_rows(pairs,['ddAPL',k+'_'+side],side+'_morphology_vs_ddAPL')
                pair_corr+=correlation_rows(pairs,['ddAPL',k+'__bacterial_minus_mammalian'],'morphology_difference_vs_ddAPL')
    tables.update(class_descriptives=pd.DataFrame(desc),class_omnibus=pd.DataFrame(omnibus),class_pairwise=pd.DataFrame(pairwise),
                  system_correlations=pd.DataFrame(corr),pairs=pairs,concordance=pd.DataFrame(concord),
                  paired_membrane_tests=pd.DataFrame(paired_tests),label_pair_ddapl_tests=pd.DataFrame(label_tests),
                  pair_correlations=pd.DataFrame(pair_corr),discordant_examples=pd.DataFrame(discord))
    for name in ['class_omnibus','class_pairwise','system_correlations','paired_membrane_tests','label_pair_ddapl_tests','pair_correlations']:
        if not tables[name].empty:tables[name]['p_BH_FDR']=bh(tables[name].p_raw)
    return tables


def run(a,new_dir,write_csv,write_json,sha):
    d=pd.read_csv(a.master)
    if 'physical_key' not in d and 'physical_system_key' in d:d['physical_key']=d.physical_system_key
    if d.physical_key.duplicated().any():raise ValueError('Duplicate physical master identities')
    d['role']=d['bilayer_type' if 'bilayer_type' in d else 'bilayer'].map(ROLE)
    truth=lambda s:s.astype(str).str.lower().isin(['true','1'])
    if 'primary_system_eligible' not in d:raise ValueError('Scientific master lacks eligibility provenance')
    d=d[truth(d.primary_system_eligible)].copy()
    prefix='morph__' if a.pilot else 'production_morph__'
    if not a.pilot:
        if prefix+'qc_status' not in d:raise ValueError('No production morphology available; use --pilot only for historical regression')
        eligible=d[prefix+'qc_status'].eq('PASS')
    else: eligible=pd.Series(True,index=d.index)
    tri=d[(d.molecule_type=='peptoid')&(d.length==3)&d.role.isin(['bacterial','mammalian'])].copy()
    # Primary paired population is selected BEFORE morphology QC, then retain both members or neither.
    keys=paired_table(tri,['APL_change_BOX'])[PAIR]
    tri=tri.merge(keys,on=PAIR,validate='many_to_one')
    if not a.pilot:
        good=tri.groupby(PAIR)[prefix+'qc_status'].agg(lambda x:len(x)==2 and x.eq('PASS').all())
        idx=set(good[good].index);tri=tri.loc[np.array([tuple(x) in idx for x in tri[PAIR].itertuples(index=False,name=None)],dtype=bool)]
    out=new_dir(a.out)
    alltables=analyse(tri,prefix)
    for name,t in alltables.items():write_csv(out/(name+'.csv'),t)
    secondary=d[eligible].copy()
    secondary_status='COMPLETE'
    try:
        for name,t in analyse(secondary,prefix).items():write_csv(out/('secondary_'+name+'.csv'),t)
    except ValueError as exc:
        secondary_status='BLOCKED: '+str(exc)
    write_csv(out/'qc_excluded.csv',d[~eligible])
    write_json(out/'statistics.json',{'source_sha256':sha(a.master),'primary_systems':len(tri),'primary_pairs':len(alltables['pairs']),
        'mode':'HISTORICAL_PILOT_REGRESSION' if a.pilot else 'PRODUCTION_QC_PASS_PAIRED',
        'secondary_status':secondary_status,'multiple_testing':'Benjamini-Hochberg within each output inferential family; primary and secondary separate',
        'units':'canonical thickness_change remains source units/sign (typically Angstrom, start minus finish); local-normal thickness is nm, endpoint minus baseline',
        'replication':'paired sequences for membrane contrasts; per-membrane system tests; leaflets never independent',
        'interpretation':'no descriptor modelling, no composite target; sparse classes reported descriptively'})
