#!/usr/bin/env python3
"""Peptoid helical-wheel schematic; idealised 120-degree repeat, not a predicted conformation.
Install: python -m pip install numpy matplotlib pillow rdkit
Run: python peptoid_wheel.py Nwe-NfeB4-Nke --mode both --structures
No arguments: render DEFAULT_SEQUENCES with SETTINGS below.
Import plot_wheel(sequence, ...) into other analysis scripts.
Source: uploaded Peptoid Builder (clean & full)(1).cdxml.
Source monomers preserve their drawn protonation; overlay protonation is separate.
Nfes/Nfer import identically and require review; use unspecified Nfex meanwhile.
Np is cyclic and cannot use the single-bond side-chain extraction.
Increasing radius separates turns schematically; it is not physical helix widening.
"""
from pathlib import Path
import argparse
import hashlib
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge, Patch
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_SEQUENCES = ["Nwe-NfeB4-Nke"]
SETTINGS = dict(mode="both", structures=False, output_dir="peptoid_wheels",
                dpi=300, protonation="ph7", sections=False, clockwise=False)
# Complete monomers as supplied; edit these to add or correct monomers.
MONOMER_SMILES = {'Nr': 'N=C(N)NCCCNCC(=O)O',
 'Nf': 'O=C(O)CNCc1ccccc1',
 'Nh': 'O=C(O)CNCc1c[nH]cn1',
 'Nk': 'NCCCCNCC(=O)O',
 'Nt': 'CC(O)NCC(=O)O',
 'Ns': 'O=C(O)CNCO',
 'Nq': 'NC(=O)CCNCC(=O)O',
 'Nn': 'NC(=O)CNCC(=O)O',
 'Nfes': 'C[C@@H](NCC(=O)O)c1ccccc1',
 'Nfer': 'C[C@@H](NCC(=O)O)c1ccccc1',
 'Nd': 'O=C(O)CNCC(=O)O',
 'Ne': 'O=C(O)CCNCC(=O)O',
 'Nc': 'O=C(O)CNCS',
 'Ng': 'NCC(=O)O',
 'Np': 'O=C(O)C1CCCN1',
 'Na': 'CNCC(=O)O',
 'Nv': 'CC(C)NCC(=O)O',
 'Ni': 'CCC(C)NCC(=O)O',
 'Nl': 'CC(C)CNCC(=O)O',
 'Nm': 'CSCCNCC(=O)O',
 'Ny': 'O=C(O)CNCc1ccc(O)cc1',
 'Nw': 'O=C(O)CNCc1c[nH]c2ccccc12',
 'Nfe': 'O=C(O)CNCCc1ccccc1',
 'Nfn': 'O=C(O)CNc1ccccc1',
 'Nwe': 'O=C(O)CNCCc1c[nH]c2ccccc12',
 'NmO': 'COCCNCC(=O)O',
 'Nse': 'O=C(O)CNCCO',
 'Nfnap': 'O=C(O)CNCc1ccc2ccccc2c1',
 'NfeB4': 'O=C(O)CNCc1ccc(Br)cc1',
 'NfeC4': 'O=C(O)CNCc1ccc(Cl)cc1',
 'Nab': 'CCCCNCC(=O)O',
 'Nke': '[NH3+]CCNCC(=O)O',
 'Nkeqm': 'C[N+](C)(C)CCNCC(=O)O',
 'Nf[naph]': 'O=C(O)CNCc1ccc2ccccc2c1',
 'Nfex': 'CC(NCC(=O)O)c1ccccc1'}

GROUPS = {
    "hydrophobic": "Na Nab Ni Nl Nm Nv Np".split(),
    "polar": "Nn Nq Ns Nse Nt NmO Nc Nh Ng".split(),
    "negative": "Nd Ne".split(),
    "aromatic": "Nf Nfn Nfe Nfex Nfes Nfer Nw Nwe Ny NfeB4 NfeC4 Nfnap".split(),
    "positive": "Nk Nke Nr Nkeqm".split(),
}
# Palette follows the user's VMD classes (not a universal chemistry standard).
COLORS = dict(hydrophobic="#e5bd24", polar="#35a34a", negative="#d53032",
              aromatic="#8845a5", positive="#275acb")
ALIASES = {"Nfe4Br": "NfeB4", "Nfe4Cl": "NfeC4", "Nfnaph": "Nfnap",
           "Nf[naph]": "Nfnap"}
# Explicit side-chain assumptions for pH 7; this is not a pKa prediction engine.
# * marks the attachment to the backbone nitrogen.
PH7_SIDECHAINS = {"Nk": "*CCCC[NH3+]", "Nke": "*CC[NH3+]",
                  "Nr": "*CCCNC(=[NH2+])N", "Nd": "*CC(=O)[O-]",
                  "Ne": "*CCC(=O)[O-]", "Nkeqm": "*CC[N+](C)(C)C"}
GRADIENT = LinearSegmentedColormap.from_list("termini", ["#2864cf", "#ffffff", "#d93636"], N=1025)


def parse_sequence(sequence):
    tokens = [x.strip() for x in sequence.strip().split("-")]
    if not tokens or any(not x for x in tokens):
        raise ValueError("Provide nonempty monomer codes separated by hyphens.")
    tokens = [ALIASES.get(x, x) for x in tokens]
    unknown = sorted(set(tokens) - MONOMER_SMILES.keys())
    if unknown:
        raise ValueError("Unknown monomers: " + ", ".join(unknown))
    return tokens


def sidechain_mol(name, protonation="ph7"):
    from rdkit import Chem
    if name in {"Nfes", "Nfer"}:
        raise ValueError("Nfes/Nfer stereochemistry needs review; use Nfex or correct the dictionary.")
    if name == "Np":
        raise ValueError("Np is cyclic: single-attachment side-chain overlay is unsupported.")
    if protonation == "ph7" and name in PH7_SIDECHAINS:
        return Chem.MolFromSmiles(PH7_SIDECHAINS[name])
    if name == "Ng":
        return Chem.MolFromSmiles("*[H]", sanitize=True)
    mol = Chem.MolFromSmiles(MONOMER_SMILES[name])
    if mol is None:
        raise ValueError("Invalid monomer SMILES: " + name)
    pattern = Chem.MolFromSmarts("[N;!$(N-C=O)]-[CH2]-[C](=[O])-[O;H1,-1]")
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        raise ValueError("Cannot identify glycine backbone: " + name)
    # Nd has symmetry-equivalent arms; either yields the same side-chain graph.
    match = matches[0]
    nitrogen = match[0]
    outside = [a.GetIdx() for a in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
               if a.GetIdx() not in match]
    if len(outside) != 1:
        raise ValueError("Ambiguous side-chain attachment: " + name)
    rw = Chem.RWMol(mol)
    anchor = rw.GetAtomWithIdx(nitrogen)
    anchor.SetAtomicNum(0)
    anchor.SetFormalCharge(0)
    anchor.SetNumExplicitHs(0)
    anchor.SetNoImplicit(True)
    for idx in sorted(match[1:], reverse=True):
        rw.RemoveAtom(idx)
    result = rw.GetMol()
    Chem.SanitizeMol(result)
    return result


def draw_sidechain(ax, name, xy, angle, protonation):
    """Render RDKit bonds with an invisible attachment anchor joined to the circle."""
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    from io import BytesIO
    mol = sidechain_mol(name, protonation)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    anchor = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    neighbor = mol.GetAtomWithIdx(anchor).GetNeighbors()[0].GetIdx()
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                       for i in range(mol.GetNumAtoms())])
    coords -= coords[anchor]
    v = coords[neighbor]
    theta = angle - math.atan2(v[1], v[0])
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    coords = coords @ rotation.T
    for i, (x, y) in enumerate(coords): conf.SetAtomPosition(i, (float(x), float(y), 0))
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 450)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.atomLabels[anchor] = " "
    opts.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    a = drawer.GetDrawCoords(anchor)
    drawer.FinishDrawing()
    im = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    # Constant image size in plot units; anchor pixels determine exact attachment.
    w, h = 1.95, 1.4625
    target = np.array(xy) + .34 * np.array([math.cos(angle), math.sin(angle)])
    left, top = target[0] - a.x / 600 * w, target[1] + a.y / 450 * h
    ax.imshow(im, extent=(left, left+w, top-h, top), zorder=6)
    return left, left+w, top-h, top


def plot_wheel(sequence, mode="terminus", structures=False, output_dir="peptoid_wheels",
               dpi=300, protonation="ph7", sections=False, clockwise=False):
    """Return saved PNG Path. N-terminus is residue 1; last residue is C-terminal."""
    if mode not in {"terminus", "functional_group"}: raise ValueError("Invalid colour mode")
    if protonation not in {"ph7", "source"}: raise ValueError("Invalid protonation mode")
    if dpi < 50: raise ValueError("dpi must be at least 50")
    names = parse_sequence(sequence)
    n = len(names)
    group = {m:g for g, members in GROUPS.items() for m in members}
    turns = (n+2)//3
    spacing = 2.8 if structures else .95
    radii = np.array([1.35 + (i//3)*spacing for i in range(n)])
    angles = np.array([(i%3)*2*math.pi/3*(-1 if clockwise else 1) for i in range(n)])
    coords = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
    size = max(6.5, 4 + turns*(2 if structures else 1))
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    bounds = [coords[:,0].min()-.7,coords[:,0].max()+.7,
              coords[:,1].min()-.7,coords[:,1].max()+.7]
    for i, (name, xy, radius, angle) in enumerate(zip(names, coords, radii, angles)):
        if sections and mode == "functional_group":
            deg = math.degrees(angle)
            ax.add_patch(Wedge((0,0), radius+.43, deg-60, deg+60,
                              width=min(spacing,.86), color=COLORS[group[name]], alpha=.13, zorder=0))
        if i < n-1:
            delta = coords[i+1]-xy
            unit = delta/np.linalg.norm(delta)
            ax.add_patch(FancyArrowPatch(xy+.37*unit, coords[i+1]-.37*unit,
                         arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#777777", zorder=1))
        fraction = i/(n-1) if n>1 else .5
        color = GRADIENT(fraction) if mode == "terminus" else COLORS[group[name]]
        ax.add_patch(Circle(xy,.34,facecolor=color,edgecolor="#252525",lw=1.4,zorder=8))
        rgb = np.array(matplotlib.colors.to_rgb(color))
        ink = "white" if np.dot(rgb,[.2126,.7152,.0722]) < .45 else "#111111"
        ax.text(*xy, name+"\n"+str(i+1),ha="center",va="center",fontsize=10,
                color=ink,weight="bold",zorder=9)
        end = "N-ter / C-ter" if n==1 else "N-ter" if i==0 else "C-ter" if i==n-1 else ""
        if end:
            ax.annotate(end, xy, xytext=(0,-34), textcoords="offset points",
                        ha="center",va="top",fontsize=10,weight="bold",zorder=9,
                        bbox=dict(facecolor="white",edgecolor="none",pad=1.5))
        if structures:
            b = draw_sidechain(ax,name,xy,angle,protonation)
            bounds = [min(bounds[0],b[0]),max(bounds[1],b[1]),min(bounds[2],b[2]),max(bounds[3],b[3])]
    ax.set_xlim(bounds[0]-.3,bounds[1]+.3)
    ax.set_ylim(bounds[2]-.4,bounds[3]+.3)
    title = "-".join(names) if n<=12 else f"Peptoid · {n} residues"
    ax.set_title(title,fontsize=13,pad=18,wrap=True)
    if mode == "functional_group":
        used = [g for g in GROUPS if any(group[x]==g for x in names)]
        fig.legend(handles=[Patch(facecolor=COLORS[g],label=g.capitalize()) for g in used],
                   loc="lower center",bbox_to_anchor=(.5,.06),ncol=min(3,len(used)),frameon=False)
    else:
        ca = fig.add_axes([.30,.095,.40,.015])
        from matplotlib.colorbar import ColorbarBase
        cb = ColorbarBase(ca,cmap=GRADIENT,orientation="horizontal")
        cb.set_ticks([0,.5,1],labels=["N-ter","Midpoint","C-ter"])
        cb.ax.tick_params(labelsize=9,length=0)
    caption = "Idealised 120° repeat; successive turns offset radially"
    if structures:
        caption += "\n" + ("Side chains: assumed predominant protonation at pH 7"
                              if protonation=="ph7" else "Side chains: protonation as supplied")
    fig.text(.5,.025,caption,ha="center",fontsize=9)
    fig.subplots_adjust(bottom=.20,top=.9,left=.06,right=.94)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    diameter_px = abs(ax.transData.transform((.60,0))[0]-ax.transData.transform((0,0))[0])
    for text in ax.texts:
        if "\n" in text.get_text():
            box = text.get_window_extent(renderer)
            ratio = min(1.0, diameter_px/max(box.width,1), diameter_px/max(box.height,1))
            text.set_fontsize(text.get_fontsize()*ratio)
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    canonical = "-".join(names)
    stem = canonical if len(canonical)<140 else canonical[:90]+"_"+hashlib.sha256(canonical.encode()).hexdigest()[:12]
    path = out / f"{stem}_{mode}_{'structures' if structures else 'labels'}_{protonation}_{'cw' if clockwise else 'ccw'}_{'sections' if sections else 'plain'}.png"
    fig.savefig(path,dpi=dpi,facecolor="white",bbox_inches="tight",pad_inches=.22)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sequences",nargs="*")
    p.add_argument("--mode",choices=["terminus","functional_group","both"],default=SETTINGS["mode"])
    p.add_argument("--structures",action=argparse.BooleanOptionalAction,default=SETTINGS["structures"])
    p.add_argument("--sections",action=argparse.BooleanOptionalAction,default=SETTINGS["sections"])
    p.add_argument("--clockwise",action=argparse.BooleanOptionalAction,default=SETTINGS["clockwise"])
    p.add_argument("--protonation",choices=["ph7","source"],default=SETTINGS["protonation"])
    p.add_argument("--output-dir",default=SETTINGS["output_dir"])
    p.add_argument("--dpi",type=int,default=SETTINGS["dpi"])
    args=p.parse_args()
    modes=["terminus","functional_group"] if args.mode=="both" else [args.mode]
    try:
        for seq in args.sequences or DEFAULT_SEQUENCES:
            for mode in modes:
                print(plot_wheel(seq,mode,args.structures,args.output_dir,args.dpi,
                                 args.protonation,args.sections,args.clockwise))
    except (ValueError,ImportError) as e:
        p.exit(2,str(e)+"\n")

if __name__ == "__main__":
    main()
