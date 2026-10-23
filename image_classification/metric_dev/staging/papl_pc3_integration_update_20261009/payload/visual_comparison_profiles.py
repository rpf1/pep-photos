"""Reference-only scientific comparison state and bounded Tkinter gallery."""
import copy,csv,hashlib,json,math,os,tempfile,uuid
from datetime import datetime,timezone
from decimal import Decimal,ROUND_FLOOR
from pathlib import Path

def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None
def read_csv(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def contained(root,relative):
 root=Path(root).resolve();p=(root/relative).resolve()
 if Path(relative).is_absolute() or not p.is_relative_to(root):raise ValueError('Outside allowed image root')
 return p
class ProfileStore:
 def __init__(self,path):
  self.path=Path(path);self.expected=sha(path);self.data=json.loads(self.path.read_text()) if self.path.exists() else {'schema_version':1,'profiles':[]}
  if self.data.get('schema_version')!=1:raise ValueError('Unsupported profile schema; no changes made')
  ids=[p['id'] for p in self.data['profiles']]
  if len(ids)!=len(set(ids)):raise ValueError('Duplicate profile IDs')
  for p in self.data['profiles']:
   if not isinstance(p.get('name'),str) or not p['name'].strip() or not isinstance(p.get('notes'),str) or not isinstance(p.get('entries'),list):raise ValueError('Invalid profile; existing file preserved')
   for e in p['entries']:
    if e.get('kind') not in ('figure','photo') or not isinstance(e.get('notes',''),str):raise ValueError('Invalid profile entry')
    if not isinstance(e.get('figure_id' if e['kind']=='figure' else 'path'),str):raise ValueError('Invalid source reference')
 def save(self):
  # Parent directory lock is the same inode flock used by the existing GUI.
  # GUI already owns it; use optimistic hash guard rather than a second flock.
  if sha(self.path)!=self.expected:raise RuntimeError('Profiles changed externally; close and reopen')
  raw=(json.dumps(self.data,indent=2,ensure_ascii=False)+'\n').encode();self.path.parent.mkdir(parents=True,exist_ok=True);tmp=None
  try:
   with tempfile.NamedTemporaryFile(dir=self.path.parent,prefix='.profiles_',delete=False) as f:tmp=Path(f.name);f.write(raw);f.flush();os.fsync(f.fileno())
   if sha(self.path)!=self.expected:raise RuntimeError('Profile save conflict')
   os.replace(tmp,self.path);self.expected=sha(self.path)
   if os.name=='posix':
    fd=os.open(self.path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
  finally:
   if tmp and tmp.exists():tmp.unlink()
 def mutate(self,fn):
  old=copy.deepcopy(self.data)
  try:result=fn();self.save();return result
  except Exception:self.data=old;raise
 def get(self,pid):return next(p for p in self.data['profiles'] if p['id']==pid)
 def create(self,name):
  if not name.strip():raise ValueError('Profile name required')
  p={'id':str(uuid.uuid4()),'name':name.strip(),'purpose':'','notes':'','entries':[],'query':None,'created_utc':stamp(),'modified_utc':stamp()}
  return self.mutate(lambda:(self.data['profiles'].append(p),p['id'])[1])
 def edit(self,pid,**fields):
  if not set(fields)<= {'name','purpose','notes','entries','query'}:raise ValueError('Unknown profile field')
  if 'name' in fields and not fields['name'].strip():raise ValueError('Profile name required')
  return self.mutate(lambda:self.get(pid).update(**copy.deepcopy(fields),modified_utc=stamp()))
 def duplicate(self,pid,name):
  p=copy.deepcopy(self.get(pid));p.update(id=str(uuid.uuid4()),name=name,created_utc=stamp(),modified_utc=stamp())
  return self.mutate(lambda:(self.data['profiles'].append(p),p['id'])[1])
 def delete(self,pid):return self.mutate(lambda:self.data['profiles'].remove(self.get(pid)))
 def add(self,pid,entry):
  entries=copy.deepcopy(self.get(pid)['entries']);entry=copy.deepcopy(entry);entry.setdefault('item_id',str(uuid.uuid4()));entry.setdefault('notes','');entries.append(entry);self.edit(pid,entries=entries)
 def move(self,pid,index,delta):
  e=copy.deepcopy(self.get(pid)['entries']);j=index+delta
  if 0<=j<len(e):e[index],e[j]=e[j],e[index];self.edit(pid,entries=e)

def bin_number(value,width,origin='0'):
 w=Decimal(str(width))
 if w not in (Decimal('.5'),Decimal('1'),Decimal('2')):raise ValueError('Supported widths: 0.5, 1, 2 Å²')
 return int(((Decimal(str(value))-Decimal(str(origin)))/w).to_integral_value(rounding=ROUND_FLOOR))
def scalar_catalog(results):
 results=Path(results);late=results/'surface_papl_analysis/tables/matched_late_pAPL.csv';u=read_csv(results/'data/unified_metrics.csv');selected={r['pair_id'] for r in u if r['selected_pair'].lower()=='true' and r['qc_pass_pair'].lower()=='true'}
 out=[]
 for r in read_csv(late):
  if r['pair_id'] not in selected:continue
  v=dict(r);v['source_sha256']=sha(late);v['time_definition']='Relax-local 80–100 ns saved 1 ns frame mean';v['reference_definition']='saved pAPL_reference_A2 (initial/pristine reference; not stationary null)';out.append(v)
 return out

def evaluate_query(catalog,q):
 required={'schema_version','metric','width_A2','origin_A2','side','run','time_definition','reference_definition','source_sha256','bin_index'}
 if not required<=q.keys() or q['schema_version']!=1:raise ValueError('Incomplete query provenance')
 key={'raw':'pAPL_late_A2','signed':'delta_pAPL_late_ref_A2','absolute':'delta_pAPL_late_ref_A2'}.get(q['metric'])
 if not key:raise ValueError('Unsupported metric')
 matched=[];counts={}
 for r in catalog:
  if r['source_sha256']!=q['source_sha256'] or r['time_definition']!=q['time_definition'] or r['reference_definition']!=q['reference_definition']:raise ValueError('Saved query source or definition changed; review before refreshing')
  if q['side']!='All' and r['side']!=q['side']:continue
  if q['metric']=='raw' and q['side']=='All':raise ValueError('Raw APL requires one membrane; baselines differ')
  if q['run']!='All' and r['physical_run']!=q['run']:continue
  try:v=float(r[key])
  except (ValueError,KeyError):continue
  if not math.isfinite(v):continue
  if q['metric']=='absolute':v=abs(v)
  n=bin_number(v,q['width_A2'],q['origin_A2']);counts[n]=counts.get(n,0)+1
  if n==int(q['bin_index']):matched.append(r)
 return matched,counts

def photo_catalog(results):
 """Identity comes from saved annotation mapping, never parsed from filenames."""
 p=Path(results)/'visual_comparison_photo_sources.csv'
 if not p.exists():return []
 out=[]
 for r in read_csv(p):
  for path in json.loads(r['image_paths']):out.append(dict(kind='photo',path=path,pair_id=r['pair_id'],side=r['side'],run=r['pair_id'].split('|')[-1],annotation=r['label'],source='live annotation paths reconciled by molecule/run/bilayer; original science references retained',metric='endpoint photograph; not APL',time='final-frame/annotation evidence; not late averaged geometry',evidence_sha256=r['evidence_sha256'],science_annotation=r.get('science_label',''),science_image_paths=r.get('science_image_paths',''),path_status=r.get('path_status','unknown')))
 return out

def resolve_entry(gui,entry):
 if entry['kind']=='figure':
  f=gui.byid.get(entry['figure_id'])
  if not f:return None,'MISSING / RETIRED figure: '+entry['figure_id']
  p=contained(gui.results,f['png_path']);axis=str(f['axes']);warning=''
  if ('APL' in axis or 'pAPL' in axis) and ('40–90' not in axis and '40-90' not in axis):warning=' | LEGACY / axis comparability not verified'
  warning+=(' | SOURCE CHANGED SINCE PROFILE SAVE' if entry.get('image_sha256') and entry['image_sha256']!=f['png_sha256'] else '')
  if p.is_file() and sha(p)!=f['png_sha256']:warning+=' | IMAGE HASH DIFFERS FROM REGISTRY'
  return p,f['figure_id']+' | '+f['family']+' | '+axis+warning
 if entry['kind']=='photo':
  p=contained(gui.results.parents[3]/'Photos',entry['path'])
  warning=' | IMAGE REPLACED SINCE SAVE' if entry.get('image_sha256') and p.is_file() and sha(p)!=entry['image_sha256'] else ''
  return p,entry['pair_id']+' | '+entry['side']+' | live label: '+entry.get('annotation','')+' | historical science label: '+entry.get('science_annotation','')+' | different evidence versions; endpoint image ≠ late mean'+warning
 raise ValueError('Unknown profile reference kind')

class ProfilesWindow:
 def __init__(self,gui):
  import tkinter as tk
  from tkinter import ttk,messagebox,simpledialog
  self.g=gui;self.tk=tk;self.ttk=ttk;self.mb=messagebox;self.ask=simpledialog;self.store=ProfileStore(gui.results/'visual_comparison_profiles.json');self.pid=None;self.photos=[]
  self.w=tk.Toplevel(gui.root);self.w.title('Visual comparison profiles — independent group notes');self.w.geometry('1450x900');self.w.protocol('WM_DELETE_WINDOW',self.close)
  bar=ttk.Frame(self.w);bar.pack(fill='x');self.search=tk.StringVar();ttk.Entry(bar,textvariable=self.search,width=25).pack(side='left');self.search.trace_add('write',lambda *a:self.refresh())
  self.choice=ttk.Combobox(bar,state='readonly',width=45);self.choice.pack(side='left');self.choice.bind('<<ComboboxSelected>>',lambda e:self.open())
  for text,fn in [('New',self.new),('Rename',self.rename),('Duplicate',self.duplicate),('Delete',self.delete),('Add current figure',self.add_figure),('Add photograph',self.add_photo),('APL BINS',self.bins),('Suggested comparisons',self.candidates),('Save notes',self.save)]:ttk.Button(bar,text=text,command=lambda fn=fn:self.safe(fn)).pack(side='left')
  self.info=tk.StringVar(value='References only. Original figure notes remain independent.');ttk.Label(self.w,textvariable=self.info).pack(fill='x')
  self.note=tk.Text(self.w,height=4,wrap='word');self.note.pack(fill='x');self.note.bind('<KeyRelease>',lambda e:self.info.set('Unsaved group notes — Save notes or close to save'))
  controls=ttk.Frame(self.w);controls.pack(fill='x');self.filter=tk.StringVar(value='All');ttk.Label(controls,text='Membrane filter:').pack(side='left');c=ttk.Combobox(controls,textvariable=self.filter,values=['All','B','M'],state='readonly',width=8);c.pack(side='left');c.bind('<<ComboboxSelected>>',lambda e:self.gallery())
  self.runfilter=tk.StringVar(value='All');ttk.Label(controls,text='Run:').pack(side='left');c=ttk.Combobox(controls,textvariable=self.runfilter,values=['All','run_0','run_ML_cycle_01','run_ML_cycle_02'],state='readonly',width=22);c.pack(side='left');c.bind('<<ComboboxSelected>>',lambda e:self.gallery())
  self.textfilter=tk.StringVar();ttk.Label(controls,text='Family / metric / annotation:').pack(side='left');ttk.Entry(controls,textvariable=self.textfilter,width=25).pack(side='left');self.textfilter.trace_add('write',lambda *a:self.gallery())
  self.items=tk.Listbox(self.w,height=5,exportselection=False);self.items.pack(fill='x');self.items.bind('<<ListboxSelect>>',lambda e:self.gallery_images())
  tools=ttk.Frame(self.w);tools.pack(fill='x')
  for text,fn in [('Move up',lambda:self.move(-1)),('Move down',lambda:self.move(1)),('Remove reference',self.remove),('Item note',self.item_note),('Open full image',self.full)]:ttk.Button(tools,text=text,command=lambda fn=fn:self.safe(fn)).pack(side='left')
  self.grid=ttk.Frame(self.w);self.grid.pack(fill='both',expand=True);self.refresh()
 def safe(self,fn):
  try:return fn()
  except Exception as e:self.mb.showerror('Comparison not saved / unavailable',str(e),parent=self.w);return False
 def refresh(self):
  self.visible=[p for p in self.store.data['profiles'] if self.search.get().lower() in (p['name']+' '+p['notes']).lower()];self.choice['values']=[p['name']+' ['+p['id'][:8]+']' for p in self.visible]
 def save(self):
  if self.pid:self.store.edit(self.pid,notes=self.note.get('1.0','end-1c'));self.info.set('Group notes saved; original figure review notes untouched')
  return True
 def open(self,pid=None):
  if self.safe(self.save) is False:return
  self.pid=pid or self.visible[self.choice.current()]['id'];p=self.store.get(self.pid);self.choice.set(p['name']+' ['+p['id'][:8]+']');self.note.delete('1.0','end');self.note.insert('1.0',p['notes']);self.gallery()
 def new(self):
  name=self.ask.askstring('New profile','Name:',parent=self.w)
  if name:self.open(self.store.create(name));self.refresh()
 def rename(self):
  if not self.pid:return
  name=self.ask.askstring('Rename','Name:',initialvalue=self.store.get(self.pid)['name'],parent=self.w)
  if name:self.save();self.store.edit(self.pid,name=name);self.refresh();self.open(self.pid)
 def duplicate(self):
  if self.pid:self.save();pid=self.store.duplicate(self.pid,self.store.get(self.pid)['name']+' (copy)');self.refresh();self.open(pid)
 def delete(self):
  if self.pid and self.mb.askyesno('Delete profile','Delete only this profile definition and notes?',parent=self.w):self.store.delete(self.pid);self.pid=None;self.note.delete('1.0','end');self.refresh();self.gallery()
 def add_figure(self):
  if not self.pid:raise ValueError('Create/open a profile first')
  f=self.g.byid[self.g.current];self.store.add(self.pid,dict(kind='figure',figure_id=self.g.current,side=f.get('side',''),run=f.get('physical_run',''),metric=str(f['axes']),family=f['family'],image_sha256=f['png_sha256']));self.gallery()
 def add_photo(self):
  if not self.pid:raise ValueError('Create/open a profile first')
  catalog=photo_catalog(self.g.results);w=self.tk.Toplevel(self.w);w.title('Existing annotation-mapped photographs');q=self.tk.StringVar();self.ttk.Entry(w,textvariable=q,width=70).pack();l=self.tk.Listbox(w,width=120,height=20);l.pack();matches=[]
  def refresh(*a):
   matches[:]=[e for e in catalog if q.get().lower() in (e['pair_id']+' '+e['side']+' '+e['annotation']).lower()];l.delete(0,'end')
   for e in matches:l.insert('end',e['pair_id']+' | '+e['side']+' | '+e['path'])
  def add():
   if l.curselection():
    entry=copy.deepcopy(matches[l.curselection()[0]]);path,caption=resolve_entry(self.g,entry);entry['image_sha256']=sha(path);self.store.add(self.pid,entry);self.gallery()
  q.trace_add('write',refresh);refresh();self.ttk.Button(w,text='Add selected reference',command=lambda:self.safe(add)).pack()
 def gallery(self):
  self.items.delete(0,'end');self.shown=[]
  if self.pid:
   for i,e in enumerate(self.store.get(self.pid)['entries']):
    try:p,caption=resolve_entry(self.g,e)
    except Exception as exc:caption='MISSING / INVALID: '+str(exc)
    if self.filter.get()!='All' and e.get('side')!=self.filter.get():continue
    if self.runfilter.get()!='All' and e.get('run')!=self.runfilter.get():continue
    if self.textfilter.get().lower() not in (caption+' '+e.get('metric','')+' '+e.get('annotation','')).lower():continue
    self.shown.append((i,e));self.items.insert('end',caption)
  query=self.store.get(self.pid).get('query') if self.pid else None
  if query:
   try:
    matched,counts=evaluate_query(scalar_catalog(self.g.results),query);self.info.set('Saved dynamic query: '+str(len(matched))+' scalar systems; fixed gallery membership unchanged. Open APL BINS to inspect/refresh.')
   except Exception as exc:self.info.set('DYNAMIC QUERY BLOCKED: '+str(exc)+'; curated notes retained')
  self.gallery_images()
 def gallery_images(self):
  from PIL import Image,ImageTk
  for c in self.grid.winfo_children():c.destroy()
  self.photos=[];start=self.items.curselection()[0] if self.items.curselection() else 0
  for col,(i,e) in enumerate(self.shown[start:start+2]):
   frame=self.ttk.Frame(self.grid);frame.grid(row=0,column=col,sticky='nsew');self.grid.columnconfigure(col,weight=1)
   try:
    p,caption=resolve_entry(self.g,e)
    if p is None or not p.is_file():raise FileNotFoundError('Missing source; saved notes retained')
    with Image.open(p) as im:im=im.convert('RGB');im.thumbnail((650,560));photo=ImageTk.PhotoImage(im)
    self.photos.append(photo);self.ttk.Label(frame,image=photo).pack()
   except Exception as exc:caption=str(exc);self.ttk.Label(frame,text='MISSING SOURCE',foreground='red').pack()
   self.ttk.Label(frame,text=caption+'\nItem note: '+e.get('notes',''),wraplength=650).pack(fill='x')
 def index(self):
  if not self.items.curselection():raise ValueError('Select an entry')
  return self.shown[self.items.curselection()[0]][0]
 def move(self,d):self.store.move(self.pid,self.index(),d);self.gallery()
 def remove(self):
  i=self.index();e=copy.deepcopy(self.store.get(self.pid)['entries']);e.pop(i);self.store.edit(self.pid,entries=e);self.gallery()
 def item_note(self):
  i=self.index();e=copy.deepcopy(self.store.get(self.pid)['entries']);s=self.ask.askstring('Item interpretation','Note:',initialvalue=e[i]['notes'],parent=self.w)
  if s is not None:e[i]['notes']=s;self.store.edit(self.pid,entries=e);self.gallery()
 def full(self):
  from PIL import Image,ImageTk
  e=self.store.get(self.pid)['entries'][self.index()];p,caption=resolve_entry(self.g,e)
  if p is None:raise FileNotFoundError(caption)
  w=self.tk.Toplevel(self.w);w.title(caption);w.geometry('1000x750');canvas=self.tk.Canvas(w);canvas.pack(fill='both',expand=True);xs=self.ttk.Scrollbar(w,orient='horizontal',command=canvas.xview);xs.pack(fill='x');ys=self.ttk.Scrollbar(w,command=canvas.yview);ys.pack(side='right',fill='y');canvas.configure(xscrollcommand=xs.set,yscrollcommand=ys.set)
  with Image.open(p) as im:original=im.convert('RGB')
  scale=[1.0]
  def render(factor=1):
   scale[0]=min(4,max(.05,scale[0]*factor));image=original.resize((max(1,int(original.width*scale[0])),max(1,int(original.height*scale[0]))));canvas.photo=ImageTk.PhotoImage(image);canvas.delete('all');canvas.create_image(0,0,image=canvas.photo,anchor='nw');canvas.configure(scrollregion=(0,0,image.width,image.height))
  canvas.bind('<Button-1>',lambda e:canvas.scan_mark(e.x,e.y));canvas.bind('<B1-Motion>',lambda e:canvas.scan_dragto(e.x,e.y,gain=1));canvas.bind('<MouseWheel>',lambda e:render(1.2 if e.delta>0 else 1/1.2));self.ttk.Button(w,text='Zoom +',command=lambda:render(1.2)).pack(side='left');self.ttk.Button(w,text='Zoom −',command=lambda:render(1/1.2)).pack(side='left');render()
 def bins(self):
  if not self.pid:raise ValueError('Create/open a profile first')
  catalog=scalar_catalog(self.g.results)
  if not catalog:raise ValueError('No selected late scalar sources')
  w=self.tk.Toplevel(self.w);w.title('Virtual APL bins — saved scalar query, not trajectory rebinning');values={};defaults={'metric':'signed','width_A2':'0.5','origin_A2':'0','side':'B','run':'All','bin_index':'0'}
  saved=self.store.get(self.pid).get('query')
  if saved:defaults.update({k:str(saved[k]) for k in defaults})
  options={'metric':['raw','signed','absolute'],'width_A2':['0.5','1','2'],'side':['B','M','All'],'run':['All','run_0','run_ML_cycle_01','run_ML_cycle_02']}
  for k,v in defaults.items():
   row=self.ttk.Frame(w);row.pack(fill='x');self.ttk.Label(row,text=k,width=20).pack(side='left');values[k]=self.tk.StringVar(value=v)
   (self.ttk.Combobox(row,textvariable=values[k],values=options[k],state='readonly') if k in options else self.ttk.Entry(row,textvariable=values[k])).pack(side='left')
  text=self.tk.Text(w,height=14,width=95);text.pack();pending={}
  def preview():
   q={k:v.get() for k,v in values.items()};q.update(schema_version=1,time_definition=catalog[0]['time_definition'],reference_definition=catalog[0]['reference_definition'],source_sha256=catalog[0]['source_sha256']);matches,counts=evaluate_query(catalog,q);pending.update(query=q,matches=matches);text.delete('1.0','end');text.insert('1.0',q['time_definition']+'\n'+q['reference_definition']+'\nHalf-open bins; no raw B/M pooling\n'+'\n'.join('['+str(Decimal(q['origin_A2'])+Decimal(k)*Decimal(q['width_A2']))+', '+str(Decimal(q['origin_A2'])+Decimal(k+1)*Decimal(q['width_A2']))+') Å²: '+str(n) for k,n in sorted(counts.items()))+'\nSelected bin scalar systems: '+str(len(matches))+'\nAvailable photo references: '+str(sum((e['pair_id'],e['side']) in {(r['pair_id'],r['side']) for r in matches} for e in photo_catalog(self.g.results)))+'\nSource SHA256: '+q['source_sha256']+'\n'+'\n'.join(r['pair_id']+' '+r['side']+' | raw='+r['pAPL_late_A2']+' | Δ='+r['delta_pAPL_late_ref_A2']+' | frame SD='+r['pAPL_late_std_1ns_A2'] for r in matches))
  def savequery():
   preview();self.store.edit(self.pid,query=pending['query']);self.info.set('Dynamic query saved with source hash; curated membership unchanged')
  def addmatched():
   preview();keys={(r['pair_id'],r['side']) for r in pending['matches']}
   for e in photo_catalog(self.g.results):
    if (e['pair_id'],e['side']) in keys:
     entry=copy.deepcopy(e);path,caption=resolve_entry(self.g,entry);entry['image_sha256']=sha(path);self.store.add(self.pid,entry)
   self.gallery()
  self.ttk.Button(w,text='Preview counts',command=lambda:self.safe(preview)).pack();self.ttk.Button(w,text='Save dynamic query',command=lambda:self.safe(savequery)).pack();self.ttk.Button(w,text='Add matching photo references (curated snapshot)',command=lambda:self.safe(addmatched)).pack()
 def candidates(self):
  p=self.g.results/'visual_comparison_candidates.csv'
  if not p.exists():raise ValueError('Suggested candidate table unavailable')
  rows=read_csv(p);w=self.tk.Toplevel(self.w);w.title('Suggested comparisons — post-hoc hypotheses, not validated exemplars');l=self.tk.Listbox(w,width=100,height=18);l.pack()
  for row in rows:l.insert('end',row['profile_name'])
  text=self.tk.Text(w,width=110,height=12,wrap='word');text.pack()
  def preview(event=None):
   if l.curselection():
    row=rows[l.curselection()[0]];text.delete('1.0','end');text.insert('1.0',row['hypothesis']+'\n'+row['group_note']+'\n'+row['uncertainty']+'\nSources: '+row['source_reference'])
  def commit():
   if not l.curselection():return
   row=rows[l.curselection()[0]];pid=self.store.create(row['profile_name']);self.store.edit(pid,purpose=row['hypothesis'],notes=row['group_note']+'\nLimitations: '+row['uncertainty']+'\nSources: '+row['source_reference']);keys={row['pair_id_1'].replace('/','|'),row['pair_id_2'].replace('/','|')}
   entries=[e for e in photo_catalog(self.g.results) if e['pair_id'] in keys and (row['side'] not in ('B','M') or e['side']==row['side'])]
   for e in entries:
    entry=copy.deepcopy(e);path,caption=resolve_entry(self.g,entry);entry['image_sha256']=sha(path);self.store.add(pid,entry)
   self.refresh();self.open(pid);self.info.set('Suggested profile created: '+str(len(entries))+' mapped photo references; physical photo availability checked on viewing; no classification inferred')
  l.bind('<<ListboxSelect>>',preview);self.ttk.Button(w,text='Create selected suggestion (references only)',command=lambda:self.safe(commit)).pack()
 def close(self):
  if self.safe(self.save) is not False:self.w.destroy();self.g._profiles_window=None

def open_profiles(gui):
 if getattr(gui,'_profiles_window',None) and gui._profiles_window.w.winfo_exists():gui._profiles_window.w.lift();return
 try:gui._profiles_window=ProfilesWindow(gui)
 except Exception as exc:gui.messagebox.showerror('Profiles unavailable',str(exc))
