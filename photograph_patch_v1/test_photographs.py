"""Local contract tests; these do not replace real VMD/trajectory validation."""
import sys
import types
import tempfile
import re
from pathlib import Path
import tkinter
import numpy as np
from PIL import Image
from photograph_validation import validate_gro, CATEGORY
try:
    import MDAnalysis
except ImportError:
    sys.modules['MDAnalysis'] = types.ModuleType('MDAnalysis')
import Photograph as photo

root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    def gro(names):
        p = tmp/'test.gro'
        p.write_text('test\n'+str(len(names))+'\n'+''.join(
            f'{i:5d}{n:<5}{"BB":>5}{i:5d}{0:8.3f}{0:8.3f}{0:8.3f}\n'
            for i,n in enumerate(names,1))+' 1 1 1\n')
        return p
    for names in [['Nfnap','NfeC4'],['Na','Nx'],['NfeB4'],['Nk','Nw','Nv']]:
        result=validate_gro(gro(names+['POPE','W','NA']))
        assert set(result['residues']) == set(names)
        assert result['polymer_atoms'] == len(names)
    for names in [['Nkeqm'],['Nzz'],['W']]:
        try: validate_gro(gro(names))
        except ValueError: pass
        else: raise AssertionError(f'Expected rejection: {names}')
    assert CATEGORY['Nfnap']==CATEGORY['Nfnaph']=='pto_aromatic'
    assert CATEGORY['NfeC4']==CATEGORY['Nfe4Cl']=='pto_aromatic'
    assert CATEGORY['NfeB4']==CATEGORY['Nfe4Br']=='pto_aromatic'
    assert CATEGORY['Na']==CATEGORY['Nx']=='pto_hydrophobic'
    for name in ['Photograph.tcl','Photograph_components.tcl']:
        text=(root/name).read_text()
        tcl=tkinter.Tcl()
        state={'matrix': np.eye(3), 'frames':{}}
        def molinfo(*args):
            if args[1]=='get': return 'BASE'
            state['matrix']=np.eye(3)
            return ''
        def rotate(axis,by,angle):
            a=np.deg2rad(float(angle));c,s=np.cos(a),np.sin(a)
            r=np.array([[1,0,0],[0,c,-s],[0,s,c]]) if axis=='x' else np.array([[c,0,s],[0,1,0],[-s,0,c]])
            state['matrix']=r@state['matrix'];return ''
        def render(kind,filename):
            state['frames'][filename]=state['matrix'].copy();return ''
        for cmd,fn in [('molinfo',molinfo),('rotate',rotate),('render',render)]:tcl.createcommand(cmd,fn)
        block=text[text.index('set photo_top_rotation'):text.index('\nphoto_view x 0')]
        tcl.eval(block)
        for call in re.findall(r'^photo_view .*$',text,re.M):tcl.eval(call)
        frames=state['frames'];assert set(frames)=={f for f,l in photo.SNAPSHOT_ORDER}
        assert abs(np.dot(frames['snapshot_side.png'][2],frames['snapshot_side_y.png'][2]))<1e-12
        assert np.allclose(frames['snapshot_top.png'],np.eye(3))
        assert np.allclose(frames['snapshot_water.png'],frames['snapshot_side.png'])
        assert np.allclose(frames['snapshot_water_y.png'],frames['snapshot_side_y.png'])
    paths=[]
    for filename,label in photo.SNAPSHOT_ORDER:
        p=tmp/filename;Image.new('RGB',(30,30),'red').save(p);paths.append((p,label))
    photo.SHOW_SYSTEM_TITLE=False
    photo.make_horizontal_panel(paths,tmp/'composite.png','test')
    with Image.open(tmp/'composite.png') as im: assert im.size==(2970,372),im.size
print('PASS: GRO coverage/rejection, aliases, both Tcl orientation sequences, nine-panel set, 2970x372 layout without title.')
print('NOT TESTED: VMD rendering, actual datasets, visual bead coverage or original camera equivalence.')
