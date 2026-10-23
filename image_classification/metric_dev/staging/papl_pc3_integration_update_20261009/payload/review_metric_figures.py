#!/usr/bin/env python3
"""Read-only scientific figure guide; only personal review state is mutable."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from figure_layout import atomic_write, read_manifest, validate_layout, sha

STATUSES=('not_reviewed','understood','revisit','important')

def local_source(results, value):
    path=(results/value).resolve()
    package=results.parent.parent.resolve()
    if not path.is_relative_to(package) or not path.is_file():
        raise ValueError('Missing or outside-package source: '+str(path))
    return path

def load_review(results):
    results=Path(results).resolve()
    if (results/'review_bundle.json').exists():
        return load_continuation_review(results)
    validation=validate_layout(results)
    try: guide=json.loads((results/'figure_guide.json').read_text())
    except (OSError,ValueError) as exc: raise ValueError('Cannot load figure_guide.json: '+str(exc)) from exc
    if guide.get('schema_version')!=1: raise ValueError('Unsupported guide schema')
    manifest=read_manifest(results); byid={r['figure_id']:r for r in manifest}
    figures=guide.get('figures',[]); ids=[f['figure_id'] for f in figures]
    if len(ids)!=len(set(ids)) or set(ids)!=set(byid): raise ValueError('Guide/manifest figure IDs do not reconcile')
    for section in ('metrics','analysis_techniques','figure_families','provenance_sources','project_context','terminology','result_set'):
        if not guide.get(section): raise ValueError('Missing guide section: '+section)
    for f in figures:
        fid=f['figure_id'];r=byid[fid]
        for key in ('title','scientific_purpose','original_question','original_hypothesis','hypothesis_status','hypothesis_provenance','expected_pattern','axes','colour_grouping_semantics','figure_specific_context','what_to_look_for','observed_result','numeric_evidence','interpretation','limitations','common_misreadings','source_tables','provenance','result_provenance'):
            if not f.get(key): raise ValueError(fid+': missing explanation '+key)
        for field,section in [('metric_ids','metrics'),('analysis_technique_ids','analysis_techniques')]:
            if not f.get(field) or any(k not in guide[section] for k in f[field]): raise ValueError(fid+': unresolved '+field)
        if f['family'] not in guide['figure_families'] or f['family']!=r['family']: raise ValueError(fid+': family mismatch')
        if any(k not in byid for k in f.get('related_figures',[])): raise ValueError(fid+': broken related figure')
        for ext in ('png','svg'):
            if f[ext+'_path']!=r[ext+'_filename']: raise ValueError(fid+': '+ext+' guide path mismatch')
        status=f['hypothesis_status']
        if status not in ('documented_expectation','exploratory_no_directional_hypothesis','not_explicitly_recorded'): raise ValueError(fid+': invalid hypothesis status')
        if status=='documented_expectation' and not any(p['provenance_type']=='contemporaneous_design' for p in f['hypothesis_provenance']): raise ValueError(fid+': historical hypothesis lacks design provenance')
        if any(p['provenance_type']!='completed_result' for p in f['result_provenance']): raise ValueError(fid+': result provenance must be completed_result')
        for source in f['source_tables']: local_source(results,source)
        for item in f['numeric_evidence']:
            local_source(results,item['source_table'])
            if not item.get('row_filter') or not item.get('columns') or not item.get('records'): raise ValueError(fid+': incomplete numeric evidence')
    allowed={'contemporaneous_design','method_definition','completed_result','historical_decision','standard_method_explanation','explicit_inference'}
    def walk(value):
        if isinstance(value,dict):
            if 'source_document' in value:
                local_source(results,value['source_document'])
                if value.get('provenance_type') not in allowed or not value.get('section_or_heading'): raise ValueError('Invalid provenance: '+str(value))
            for child in value.values(): walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(guide)
    for tid,t in guide['analysis_techniques'].items():
        for key in ('name','purpose','original_question','original_hypothesis','hypothesis_status','hypothesis_provenance','why_this_method_was_chosen','how_to_read_the_plot','what_would_support_the_hypothesis','what_would_argue_against_it','important_limitations'):
            if not t.get(key):raise ValueError('Technique '+tid+': missing '+key)
    for mid,m in guide['metrics'].items():
        for key in ('name','plain_language_definition','technical_definition_or_formula','units','reference_time_definition','sign_direction','large_values','small_zero_values','limitations','provenance'):
            if not m.get(key):raise ValueError('Metric '+mid+': missing '+key)
    figures.sort(key=lambda f:(f['family'],f['figure_id']))
    return guide,figures,validation


def load_continuation_review(results):
    """Merge the continuation registry into the existing reviewer; preserve legacy loading."""
    from PIL import Image
    guide=json.loads((results/'review_bundle.json').read_text())
    if guide.get('schema_version')!=2:raise ValueError('Unsupported continuation registry')
    figures=guide['figures'];ids=[f['figure_id'] for f in figures]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate figure IDs')
    required=('title','family','scientific_purpose','original_question','original_hypothesis','hypothesis_status','hypothesis_provenance','expected_pattern','axes','source_tables','observed_result','interpretation','limitations','status','n','generator','method','provenance','result_provenance','numeric_evidence','figure_specific_context','what_to_look_for','colour_grouping_semantics','common_misreadings')
    for f in figures:
        for k in required:
            if k not in f or f[k] is None or f[k]=='' or f[k]==[]:raise ValueError(f['figure_id']+': missing '+k)
        for ext in ('png','svg'):
            path=local_source(results,f[ext+'_path'])
            if sha(path)!=f[ext+'_sha256']:raise ValueError('Changed figure '+str(path))
            if ext=='png':
                with Image.open(path) as im:im.verify()
        for source in f['source_tables']:local_source(results,source)
        if f['family'] not in guide['figure_families']:raise ValueError('Unregistered family '+f['family'])
        if any(fid not in ids for fid in f.get('related_figures',[])):raise ValueError('Broken related figure')
        for item in f['provenance']+f['result_provenance']+f['hypothesis_provenance']:
            local_source(results,item['source_document'])
        for item in f['numeric_evidence']:local_source(results,item['source_table'])
        # Keep original metric/technique explanations for historical figure IDs.
        if 'original_guide_id' not in f:
            f['metric_ids']=['continuation'];f['analysis_technique_ids']=['continuation']
    guide['metrics']['continuation']={'name':'pAPL / curved-surface continuation','plain_language_definition':'Projected response, geometry correction and physical deviation are distinct constructs.','technical_definition_or_formula':'See VARIABLES_AND_METHODS.md and the exact linked table columns.','units':'Declared in each figure','reference_time_definition':'Primary: Relax-local 80–100 ns; surface changes: Pull initial; projected reference and start changes distinguished.','sign_direction':'Signed deviations retained; q95 RMS is nonnegative.','large_values':'A large measured response under the declared definition.','small_zero_values':'Little response under that definition; not necessarily no contact.','limitations':'No biological endpoint or independent trajectory ensemble.', 'provenance':[]}
    guide['analysis_techniques']['continuation']={'name':'Trajectory-level continuation evidence','why_this_method_was_chosen':'Matched windows, strict run identity, frozen transformation reproduction and grouped nested validation.','null_or_comparison_logic':'Test against counterexamples and within-phenotype trends.','what_to_inspect':'Magnitude, overlap, subset size, sources and caveats.','patterns_to_look_for':'Physical-rank disagreement and stable versus conditional relationships.','what_cannot_be_concluded':'Causality, burial/contact or biological activity.'}
    figures.sort(key=lambda f:(f['family'],f['figure_id']))
    return guide,figures,dict(logical_figures=len(figures),png=len(figures),svg=len(figures),families=len(guide['figure_families']),paths_hashes_axes='passed',registry='continuation schema 2')

def result_identity(results):
    # Stable across moves and guide copy edits; independent of absolute paths.
    p=Path(results)/'analysis_manifest.json'
    identity=sha(p if p.exists() else Path(results)/'data/frozen_parameters.json')
    state=Path(results)/'figure_review_state.json'
    if state.exists():
        old=json.loads(state.read_text()).get('result_identity')
        preserved=Path(results).parents[2]/'figure_gui_analysis/metric_dev/results'/Path(results).name/'analysis_manifest.json'
        if preserved.exists() and old==sha(preserved):return old
    return identity

class ReviewState:
    def __init__(self,path,identity,ids):
        self.path=Path(path);self.identity=identity;self.ids=set(ids)
        self.expected=sha(self.path) if self.path.exists() else None
        self.data=json.loads(self.path.read_text()) if self.path.exists() else dict(schema_version=1,result_identity=identity,last_figure_id=None,per_figure={})
        if self.data.get('schema_version')!=1 or self.data.get('result_identity')!=identity:raise ValueError('Review state belongs to a different result/schema; existing state preserved')
        for fid,v in self.data.get('per_figure',{}).items():
            if v.get('status') not in STATUSES or not isinstance(v.get('note',''),str):raise ValueError('Invalid stored review status/note for '+fid)
        # Old IDs remain preserved if a later guide changes coverage.
    def update(self,fid,status=None,note=None,viewed=False):
        if fid not in self.ids:raise ValueError('Unknown figure '+fid)
        if status is not None and status not in STATUSES:raise ValueError('Invalid review status '+str(status))
        if note is not None and not isinstance(note,str):raise ValueError('Note must be text')
        current=sha(self.path) if self.path.exists() else None
        if current!=self.expected:raise RuntimeError('Review state changed externally. Close and reopen; no overwrite performed.')
        data=copy.deepcopy(self.data);now=datetime.now(timezone.utc).isoformat();v=data['per_figure'].setdefault(fid,dict(status='not_reviewed',note='',first_viewed=now,last_viewed=now,last_modified=now))
        if status is not None:v['status']=status
        if note is not None:v['note']=note
        if viewed:v['last_viewed']=now
        v['last_modified']=now;data['last_figure_id']=fid
        raw=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
        atomic_write(self.path,raw);self.expected=hashlib.sha256(raw).hexdigest();self.data=data


def self_test(results):
    guide,figures,validation=load_review(results)
    from visual_comparison_profiles import ProfileStore
    profiles=ProfileStore(Path(results)/'visual_comparison_profiles.json')
    validation['profiles_schema']='passed (read-only)'
    validation['saved_profiles']=len(profiles.data['profiles'])
    ids=[f['figure_id'] for f in figures]
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/'figure_review_state.json';state=ReviewState(p,result_identity(results),ids)
        state.update(ids[0],status='revisit',note='temporary self-test note',viewed=True)
        restored=ReviewState(p,result_identity(results),ids)
        assert restored.data['last_figure_id']==ids[0]
        assert restored.data['per_figure'][ids[0]]['note']=='temporary self-test note'
        try:restored.update(ids[0],status='invalid')
        except ValueError:pass
        else:raise AssertionError('Invalid status accepted')
        assert len(list(Path(temp).iterdir()))==1
    return dict(**validation,guide_figures=len(figures),metrics=len(guide['metrics']),techniques=len(guide['analysis_techniques']),state_roundtrip='passed',deterministic_order=ids)


def readable(value,level=0):
    """Readable nested guide fields, including exact evidence rows, without raw JSON."""
    if value is None:return 'Not available / not estimable'
    if isinstance(value,float):return format(value,'.7g')
    if isinstance(value,dict):
        return '\n\n'.join(k.replace('_',' ').capitalize()+':\n'+readable(v,level+1) for k,v in value.items())
    if isinstance(value,list):
        if value and all(isinstance(v,dict) for v in value):
            return '\n\n'.join(' | '.join(k+': '+readable(v) for k,v in row.items()) for row in value)
        return '\n'.join('• '+readable(v,level+1) for v in value)
    return str(value)

class FigureReviewGUI:
    def __init__(self,root,results,guide,figures,state):
        import tkinter as tk
        from tkinter import ttk,messagebox
        from PIL import Image,ImageTk
        self.tk=tk;self.ttk=ttk;self.messagebox=messagebox;self.Image=Image;self.ImageTk=ImageTk
        self.root=root;self.results=Path(results);self.guide=guide;self.figures=figures;self.byid={f['figure_id']:f for f in figures};self.state=state
        self.current=None;self.original=None;self.photo=None;self.scale=1.;self.fit_mode=True;self.render_job=None;self.loading=False;self.note_error=False
        root.title('PFPML — Guided scientific figure review');root.geometry('1700x1000');root.minsize(1050,650)
        style=ttk.Style(root);style.configure('TButton',padding=5);style.configure('Treeview',rowheight=27,font=('DejaVu Sans',10))
        top=ttk.Frame(root,padding=6);top.pack(fill='x')
        from visual_comparison_profiles import open_profiles
        ttk.Button(top,text='Profiles / Comparisons',command=lambda:open_profiles(self)).pack(side='left')
        ttk.Button(top,text='Previous  ←',command=lambda:self.navigate(-1)).pack(side='left')
        ttk.Button(top,text='Next  →',command=lambda:self.navigate(1)).pack(side='left')
        self.position=tk.StringVar();ttk.Label(top,textvariable=self.position,font=('DejaVu Sans',11,'bold')).pack(side='left',padx=12)
        ttk.Label(top,text='Search:').pack(side='left');self.search=tk.StringVar();search=ttk.Entry(top,textvariable=self.search,width=24);search.pack(side='left',padx=5)
        self.family=tk.StringVar(value='All families');families=['All families']+sorted(guide['figure_families'])
        chooser=ttk.Combobox(top,textvariable=self.family,values=families,state='readonly',width=27);chooser.pack(side='left',padx=5)
        self.search.trace_add('write',lambda *args:self.refresh_tree());chooser.bind('<<ComboboxSelected>>',lambda e:self.choose_family())
        panes=ttk.Panedwindow(root,orient='horizontal');panes.pack(fill='both',expand=True,padx=6)
        left=ttk.Frame(panes,width=240);centre=ttk.Frame(panes,width=850);right=ttk.Frame(panes,width=570)
        panes.add(left,weight=1);panes.add(centre,weight=5);panes.add(right,weight=3)
        self.tree=ttk.Treeview(left,show='tree',selectmode='browse');ys=ttk.Scrollbar(left,orient='vertical',command=self.tree.yview);self.tree.configure(yscrollcommand=ys.set);ys.pack(side='right',fill='y');self.tree.pack(fill='both',expand=True);self.tree.bind('<<TreeviewSelect>>',self.tree_select)
        toolbar=ttk.Frame(centre);toolbar.pack(fill='x')
        for label,fn in [('Fit [F]',self.fit),('Actual [0]',lambda:self.zoom(absolute=1)),('Zoom +',lambda:self.zoom(1.25)),('Zoom −',lambda:self.zoom(.8)),('Open SVG',self.open_svg)]:ttk.Button(toolbar,text=label,command=fn).pack(side='left')
        self.zoom_label=tk.StringVar();ttk.Label(toolbar,textvariable=self.zoom_label).pack(side='left',padx=8)
        imageframe=ttk.Frame(centre);imageframe.pack(fill='both',expand=True);imageframe.rowconfigure(0,weight=1);imageframe.columnconfigure(0,weight=1)
        self.canvas=tk.Canvas(imageframe,background='#e8e9eb',highlightthickness=0)
        xs=ttk.Scrollbar(imageframe,orient='horizontal',command=lambda *a:self.scroll('x',*a));ys=ttk.Scrollbar(imageframe,orient='vertical',command=lambda *a:self.scroll('y',*a))
        self.canvas.configure(xscrollcommand=xs.set,yscrollcommand=ys.set);self.canvas.grid(row=0,column=0,sticky='nsew');xs.grid(row=1,column=0,sticky='ew');ys.grid(row=0,column=1,sticky='ns')
        self.canvas.bind('<Configure>',lambda e:self.fit() if self.fit_mode else self.schedule_render())
        self.canvas.bind('<ButtonPress-1>',lambda e:(self.canvas.focus_set(),self.canvas.scan_mark(e.x,e.y)))
        self.canvas.bind('<B1-Motion>',self.pan)
        self.canvas.bind('<MouseWheel>',lambda e:self.wheel(e,-1 if e.delta>0 else 1))
        self.canvas.bind('<Button-4>',lambda e:self.wheel(e,-1));self.canvas.bind('<Button-5>',lambda e:self.wheel(e,1))
        self.title=tk.StringVar();ttk.Label(right,textvariable=self.title,wraplength=540,font=('DejaVu Sans',12,'bold'),padding=8).pack(fill='x')
        right.bind('<Configure>',lambda e:self.title_label_wrap(right))
        tabs=ttk.Notebook(right);tabs.pack(fill='both',expand=True);self.panels={}
        for name in ['Overview','Metrics','Design','Read plot','Result','Limits','Sources']:
            frame=ttk.Frame(tabs);tabs.add(frame,text=name);text=tk.Text(frame,wrap='word',font=('DejaVu Sans',11),padx=12,pady=10,background='#ffffff',relief='flat')
            scroll=ttk.Scrollbar(frame,command=text.yview);text.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');text.pack(fill='both',expand=True);text.configure(state='disabled')
            text.tag_configure('heading',font=('DejaVu Sans',12,'bold'),foreground='#245475',spacing1=12,spacing3=5)
            if name=='Design':text.tag_configure('heading',foreground='#805719')
            if name=='Result':text.tag_configure('heading',foreground='#17694e')
            if name=='Limits':text.tag_configure('heading',foreground='#943f36')
            self.panels[name]=text
        relatedframe=ttk.Frame(right,padding=4);relatedframe.pack(fill='x');ttk.Label(relatedframe,text='Related:').pack(side='left');self.related=tk.StringVar();self.related_combo=ttk.Combobox(relatedframe,textvariable=self.related,state='readonly');self.related_combo.pack(side='left',fill='x',expand=True);self.related_combo.bind('<<ComboboxSelected>>',lambda e:self.show(self.related.get()))
        sourceframe=ttk.Frame(right,padding=4);sourceframe.pack(fill='x');self.source=tk.StringVar();self.source_combo=ttk.Combobox(sourceframe,textvariable=self.source,state='readonly');self.source_combo.pack(side='left',fill='x',expand=True);ttk.Button(sourceframe,text='Preview source',command=self.preview_source).pack(side='right')
        bottom=ttk.Frame(root,padding=7);bottom.pack(fill='x');self.status=tk.StringVar(value='not_reviewed')
        ttk.Label(bottom,text='Review status:').grid(row=0,column=0,sticky='w')
        for i,status in enumerate(STATUSES,1):ttk.Radiobutton(bottom,text=status.replace('_',' ').title(),value=status,variable=self.status,command=self.save_status).grid(row=0,column=i,padx=8)
        ttk.Label(bottom,text='Notes (saved immediately):').grid(row=1,column=0,sticky='nw',pady=6)
        self.notes=tk.Text(bottom,height=3,wrap='word',font=('DejaVu Sans',11));self.notes.grid(row=1,column=1,columnspan=4,sticky='ew',pady=6);bottom.columnconfigure(4,weight=1);self.notes.bind('<<Modified>>',self.note_changed)
        self.feedback=tk.StringVar(value='Drag to pan. Ctrl+wheel zooms. Arrow keys navigate outside text fields.');ttk.Label(bottom,textvariable=self.feedback).grid(row=2,column=0,columnspan=5,sticky='w')
        root.bind('<KeyPress>',self.key);root.protocol('WM_DELETE_WINDOW',self.close)
        self.refresh_tree();last=state.data.get('last_figure_id');self.show(last if last in self.byid else figures[0]['figure_id'])
    def title_label_wrap(self,right):
        for child in right.winfo_children():
            if isinstance(child,self.ttk.Label):child.configure(wraplength=max(200,right.winfo_width()-24))
    def matches(self,f):
        query=self.search.get().lower().strip();family=self.family.get()
        hay=' '.join([f['figure_id'],f['title'],f['family']]+f['metric_ids']+f['analysis_technique_ids']).lower()
        return (family=='All families' or f['family']==family) and all(word in hay for word in query.split())
    def refresh_tree(self):
        if not hasattr(self,'tree'):return
        # Do not reset tree expansion/scroll whenever a filter changes.
        old_expanded=getattr(self,'_tree_open',{}).copy()
        old_expanded.update({iid:bool(self.tree.item(iid,'open')) for iid in self.tree.get_children()})
        self._tree_open=old_expanded
        old_scroll=self.tree.yview()
        self.tree.delete(*self.tree.get_children());self.filtered=[f['figure_id'] for f in self.figures if self.matches(f)]
        for fid in self.filtered:
            f=self.byid[fid];parent='family:'+f['family']
            if not self.tree.exists(parent):
                self.tree.insert('', 'end',iid=parent,text=self.guide['figure_families'][f['family']]['name'],open=old_expanded.get(parent,True))
            status=self.state.data['per_figure'].get(fid,{}).get('status','not_reviewed')
            self.tree.insert(parent,'end',iid=fid,text=fid+'  ['+status.replace('_',' ')+']')
        if self.current in self.filtered:self.tree.selection_set(self.current)
        if old_scroll:self.tree.yview_moveto(old_scroll[0])
        self.update_position()
    def choose_family(self):
        self.refresh_tree()
        if self.filtered:self.show(self.filtered[0])
    def update_position(self):
        if self.current:self.position.set(f'{[f["figure_id"] for f in self.figures].index(self.current)+1} / {len(self.figures)}   ({len(getattr(self,"filtered",[]))} matching)')
    def tree_select(self,event=None):
        selection=self.tree.selection()
        if not selection:return
        fid=selection[0]
        # A family row is a folder to expand/collapse, not a figure to open.
        if fid in self.byid and fid!=self.current:self.show(fid)
    def fill(self,name,sections):
        text=self.panels[name];text.configure(state='normal');text.delete('1.0','end')
        for title,content in sections:
            text.insert('end',title+'\n','heading');text.insert('end',readable(content)+'\n\n')
        text.configure(state='disabled');text.yview_moveto(0)
    def show(self,fid):
        if fid not in self.byid:return
        if self.current and not self.save_note():return
        try:
            f=self.byid[fid]
            with self.Image.open(self.results/f['png_path']) as image:original=image.convert('RGB')
            self.state.update(fid,viewed=True)
        except Exception as exc:self.messagebox.showerror('Cannot open figure',str(exc));return
        self.current=fid;self.original=original;self.title.set(f['title']);self.loading=True
        saved=self.state.data['per_figure'][fid];self.status.set(saved['status']);self.notes.delete('1.0','end');self.notes.insert('1.0',saved['note']);self.notes.edit_modified(False);self.loading=False
        techniques=[self.guide['analysis_techniques'][k] for k in f['analysis_technique_ids']]
        self.fill('Overview',[('Purpose',f['scientific_purpose']),('Figure context',f['figure_specific_context']),('Evidence status / cohort',{'status':f.get('status','historical'),'n':f.get('n','see source'),'reconstruction':f.get('reconstruction_status','original/new')} ),('Method',f.get('method','See technique definitions')),('Technique',[t['name'] for t in techniques]),('Axes and encoding',f['axes']),('Project context',self.guide['project_context']),('Terminology',self.guide['terminology'])])
        self.fill('Metrics',[(self.guide['metrics'][k]['name'],self.guide['metrics'][k]) for k in f['metric_ids']])
        self.fill('Design',[('ORIGINAL QUESTION — design record',f['original_question']),('ORIGINAL HYPOTHESIS / EXPLORATORY STATUS',f['original_hypothesis']),('Hypothesis status',f['hypothesis_status']),('Expected patterns — comparison criteria',f['expected_pattern']),('Why this method was chosen',[t['why_this_method_was_chosen'] for t in techniques]),('Design provenance',f['hypothesis_provenance'])])
        self.fill('Read plot',[('What to inspect manually',f['what_to_look_for']),('Visual encoding',f['colour_grouping_semantics']),('Comparison logic',[t['null_or_comparison_logic'] for t in techniques]),('Metric-specific inspection',[t.get('what_to_inspect',t['patterns_to_look_for']) for t in techniques])])
        self.fill('Result',[('OBSERVED RESULT — saved exports and continuation',f['observed_result']),('Interpretation',f['interpretation']),('Numerical evidence — exact filters and values',f['numeric_evidence']),('Result provenance',f['result_provenance'])])
        self.fill('Limits',[('Important limitations',f['limitations']),('Common misreadings',f['common_misreadings']),('What this does not establish',[t['what_cannot_be_concluded'] for t in techniques])])
        self.fill('Sources',[('Figure identity',fid),('PNG',str((self.results/f['png_path']).resolve())),('SVG',str((self.results/f['svg_path']).resolve())),('Tables',f['source_tables']),('Exact provenance',f['provenance']),('Historical source availability',f.get('historical_sources',[])),('Execution and chronology',self.guide['result_set']),('Related figures',f['related_figures'])])
        self.related_combo.configure(values=f['related_figures']);self.related.set('')
        sources=list(dict.fromkeys(f['source_tables']+[p['source_document'] for p in f['provenance']]))
        self.source_combo.configure(values=sources);self.source.set(sources[0] if sources else '')
        self.fit()
        # Update just the selected item; rebuilding the whole tree here
        # expands closed folders and displaces the user's next click.
        if self.tree.exists(fid):
            status=self.state.data['per_figure'].get(fid,{}).get('status','not_reviewed')
            self.tree.item(fid,text=fid+'  ['+status.replace('_',' ')+']')
            if self.tree.selection()!=(fid,):self.tree.selection_set(fid)
            self.tree.see(fid)
        self.update_position();self.feedback.set('Saved view: '+fid)
    def save_note(self):
        if not self.current or self.loading:return True
        note=self.notes.get('1.0','end-1c');old=self.state.data['per_figure'].get(self.current,{})
        if note==old.get('note',''):return True
        try:self.state.update(self.current,note=note);self.note_error=False;self.feedback.set('Notes saved immediately.');return True
        except Exception as exc:
            self.feedback.set('NOT SAVED — '+str(exc))
            if not self.note_error:self.messagebox.showerror('Notes not saved — navigation blocked',str(exc))
            self.note_error=True;return False
    def note_changed(self,event=None):
        if self.notes.edit_modified():
            self.notes.edit_modified(False)
            if not self.loading:self.save_note()
    def save_status(self):
        if not self.current:return
        try:
            self.state.update(self.current,status=self.status.get())
            if self.tree.exists(self.current):
                status=self.state.data['per_figure'][self.current]['status']
                self.tree.item(self.current,text=self.current+'  ['+status.replace('_',' ')+']')
            self.update_position();self.feedback.set('Review status saved.')
        except Exception as exc:
            self.status.set(self.state.data['per_figure'][self.current]['status']);self.messagebox.showerror('Status not saved',str(exc))
    def navigate(self,step):
        if not self.filtered:return
        i=self.filtered.index(self.current) if self.current in self.filtered else (-1 if step>0 else len(self.filtered))
        self.show(self.filtered[max(0,min(len(self.filtered)-1,i+step))])
    def key(self,e):
        if e.widget.winfo_class() in ('Text','Entry','TEntry','TCombobox','Spinbox','TSpinbox'):return
        action={'Left':lambda:self.navigate(-1),'Right':lambda:self.navigate(1),'n':lambda:self.navigate(1),'p':lambda:self.navigate(-1),'f':self.fit,'0':lambda:self.zoom(absolute=1),'plus':lambda:self.zoom(1.25),'equal':lambda:self.zoom(1.25),'minus':lambda:self.zoom(.8)}.get(e.keysym)
        if action:action();return 'break'
    def fit(self):
        if self.original is None:return
        self.fit_mode=True;w,h=self.original.size;self.scale=max(.001,min(max(1,self.canvas.winfo_width()-8)/w,max(1,self.canvas.winfo_height()-8)/h,1))
        self.canvas.configure(scrollregion=(0,0,w*self.scale,h*self.scale));self.canvas.xview_moveto(0);self.canvas.yview_moveto(0);self.schedule_render()
    def zoom(self,factor=None,absolute=None):
        if self.original is None:return
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height();cx=self.canvas.canvasx(w/2)/self.scale;cy=self.canvas.canvasy(h/2)/self.scale
        self.scale=min(8,max(.01,absolute if absolute is not None else self.scale*factor));self.fit_mode=False;iw,ih=self.original.size
        self.canvas.configure(scrollregion=(0,0,iw*self.scale,ih*self.scale));self.canvas.xview_moveto(max(0,(cx*self.scale-w/2)/(iw*self.scale)));self.canvas.yview_moveto(max(0,(cy*self.scale-h/2)/(ih*self.scale)));self.schedule_render()
    def schedule_render(self):
        if self.render_job is not None:self.root.after_cancel(self.render_job)
        self.render_job=self.root.after_idle(self.render)
    def render(self):
        self.render_job=None
        if self.original is None:return
        iw,ih=self.original.size;cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height();s=self.scale
        left=max(0,int(self.canvas.canvasx(0)/s)-2);top=max(0,int(self.canvas.canvasy(0)/s)-2);right=min(iw,int(self.canvas.canvasx(cw)/s)+3);bottom=min(ih,int(self.canvas.canvasy(ch)/s)+3)
        self.canvas.delete('image')
        if right>left and bottom>top:
            tile=self.original.crop((left,top,right,bottom));tile=tile.resize((max(1,round((right-left)*s)),max(1,round((bottom-top)*s))),self.Image.Resampling.LANCZOS)
            self.photo=self.ImageTk.PhotoImage(tile);self.canvas.create_image(left*s,top*s,image=self.photo,anchor='nw',tags='image')
        self.zoom_label.set(f'{s:.0%}')
    def scroll(self,axis,*args):
        (self.canvas.xview if axis=='x' else self.canvas.yview)(*args);self.schedule_render()
    def pan(self,e):self.canvas.scan_dragto(e.x,e.y,gain=1);self.schedule_render()
    def wheel(self,e,direction):
        if e.state&4:self.zoom(1.2 if direction<0 else 1/1.2)
        else:
            (self.canvas.xview_scroll if e.state&1 else self.canvas.yview_scroll)(direction*3,'units');self.schedule_render()
        return 'break'
    def open_svg(self):
        if not self.current:return
        path=(self.results/self.byid[self.current]['svg_path']).resolve()
        self.root.clipboard_clear();self.root.clipboard_append(str(path));self.feedback.set('SVG path copied: '+str(path))
        try:
            if sys.platform=='win32':os.startfile(str(path));return
            cmd=['open' if sys.platform=='darwin' else 'xdg-open',str(path)]
            process=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            def check():
                status=process.poll()
                if status is None:self.root.after(250,check)
                elif status:self.messagebox.showinfo('SVG application unavailable','Path copied to clipboard:\n'+str(path))
            self.root.after(250,check)
        except OSError:self.messagebox.showinfo('SVG application unavailable','Path copied to clipboard:\n'+str(path))
    def preview_source(self):
        import csv,itertools
        if not self.source.get():return
        try:
            path=local_source(self.results,self.source.get())
            if path.suffix=='.csv':
                with path.open(newline='') as stream:
                    reader=csv.reader(stream);header=next(reader);rows=list(itertools.islice(reader,100))
                content='Read-only preview: first 100 rows, first 16 columns. Full path: '+str(path)+'\n\n'
                content+='\t'.join(header[:16])+'\n'+'\n'.join('\t'.join(row[:16]) for row in rows)
            else:
                with path.open(errors='replace') as stream:content=stream.read(150000)
                content='Read-only preview (up to 150,000 characters): '+str(path)+'\n\n'+content
            window=self.tk.Toplevel(self.root);window.title(path.name+' — read only');window.geometry('1100x650');text=self.tk.Text(window,wrap='none',font=('DejaVu Sans Mono',10));y=self.ttk.Scrollbar(window,orient='vertical',command=text.yview);x=self.ttk.Scrollbar(window,orient='horizontal',command=text.xview);text.configure(yscrollcommand=y.set,xscrollcommand=x.set);y.pack(side='right',fill='y');x.pack(side='bottom',fill='x');text.pack(fill='both',expand=True);text.insert('1.0',content);text.configure(state='disabled')
        except Exception as exc:self.messagebox.showerror('Source unavailable',str(exc))
    def close(self):
        if self.save_note():
            window=getattr(self,'_profiles_window',None)
            if window and window.safe(window.save) is False:return
            self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True);parser.add_argument('--check',action='store_true',help='Read-only layout/guide validation and temporary-state self-test; no display needed')
    args=parser.parse_args()
    # Relative --results paths must work even when result_identity() uses parents.
    args.results=args.results.expanduser().resolve()
    try:
        if args.check:
            result=self_test(args.results);result.pop('deterministic_order');print(json.dumps(result,indent=2));return
        guide,figures,_=load_review(args.results)
        import tkinter as tk
        try:root=tk.Tk()
        except tk.TclError as exc:raise ValueError('No usable graphical display. Launch in a ThinLinc desktop terminal, or use --check for headless validation. '+str(exc)) from exc
        lockfd=None
        # A directory advisory lock needs no extra state file and survives JSON replacement.
        if os.name=='posix':
            import fcntl
            lockfd=os.open(str(args.results),os.O_RDONLY)
            try:fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:os.close(lockfd);root.destroy();raise ValueError('Another reviewer has this result open; close it first.')
        try:
            state=ReviewState(args.results/'figure_review_state.json',result_identity(args.results),[f['figure_id'] for f in figures])
            FigureReviewGUI(root,args.results,guide,figures,state);root.mainloop()
        finally:
            if lockfd is not None:os.close(lockfd)
    except (ValueError,OSError,KeyError) as exc:parser.exit(2,'Review error: '+str(exc)+'\n')

if __name__=='__main__':main()
