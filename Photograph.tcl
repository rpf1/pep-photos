proc geom_center {selection} {
    set gc [veczero]
    foreach coord [$selection get {x y z}] {
        set gc [vecadd $gc $coord]
    }
    return [vecscale [expr 1.0 / [$selection num]] $gc]
}

mol new AAAADEMC_80.psf type psf
mol addfile AAAADEMC_80.pdb type pdb waitfor all

mol delrep 0 top

# ============================================================================
# CENTRAL DEFINITIONS
# ============================================================================

# Bilayer / solvent / ions
set bilayer_resnames {POPC POPS POPE POPG DPPC DTPC CHOL}
set water_resnames  {H2O W SW TW WF}
set ion_resnames    {NA CL NA+ CL- ION}

# Peptide residue groups
set pep_hydrophobic {ALAC ALAN ALA VALC VALN VAL ILEC ILEN ILE LEUC LEUN LEU METC METN MET CYSC CYSN CYS}
set pep_proline     {PROC PRON PRO}
set pep_polar       {SERC SERN SER THRC THRN THR ASNC ASNN ASN GLNC GLNN GLN HSEC HSEN HSE}
set pep_negative    {ASPC ASPN ASP GLUC GLUN GLU}
set pep_aromatic    {PHEC PHEN PHE TYRC TYRN TYR TRPC TRPN TRP}
set pep_positive    {LYSC LYSN LYS ARGC ARGN ARG}

# --------------------------------------------------------------------------
# IMPORTANT:
# Put your peptoid residue names here.
# These are EXAMPLES only. Replace/extend them to match your actual .gro/.xtc.
# --------------------------------------------------------------------------
set pto_hydrophobic {Na Nf Ni Nl Nm Nv Nw}
set pto_polar       {}
set pto_negative    {}
set pto_aromatic    {}
set pto_positive    {}

# Combined lists
set peptide_resnames [concat \
    $pep_hydrophobic \
    $pep_proline \
    $pep_polar \
    $pep_negative \
    $pep_aromatic \
    $pep_positive]

set peptoid_resnames [concat \
    $pto_hydrophobic \
    $pto_polar \
    $pto_negative \
    $pto_aromatic \
    $pto_positive]

set all_polymer_resnames [concat $peptide_resnames $peptoid_resnames]

# Useful atom selections
set Bilayer [atomselect top "resname [join $bilayer_resnames { }]"]
set W       [atomselect top "resname [join $water_resnames { }]"]
set All     [atomselect top "all"]
set Polymer [atomselect top "resname [join $all_polymer_resnames { }]"]

# Only compute center if polymer exists
if {[$Polymer num] > 0} {
    set COG [geom_center $Polymer]
}

pbc wrap -all
axes location off
color Display Background white
display projection Orthographic
display depthcue off

# Make water actually light blue
color change rgb 17 0.72 0.86 1.00

# ============================================================================
# BILAYER VIEW
# ============================================================================

mol selection "resname [join $bilayer_resnames { }]"
mol color Charge
mol representation QuickSurf 1.2 0.3 1.0 2
mol addrep top
mol on top

render snapshot snapshot_top_bilayer.png
rotate x by -90
render snapshot snapshot_side_bilayer.png
rotate x by 90

# ============================================================================
# POLYMER REPS
# Draw full residues, not just backbone beads
# ============================================================================

# Hydrophobic
mol selection "resname [join [concat $pep_hydrophobic $pto_hydrophobic] { }]"
mol color ColorID 3
mol representation VDW 1.4 12.0
mol addrep top
mol on top

# Proline
mol selection "resname [join $pep_proline { }]"
mol color ColorID 2
mol representation VDW 1.4 12.0
mol addrep top
mol on top

# Polar
if {[llength [concat $pep_polar $pto_polar]] > 0} {
    mol selection "resname [join [concat $pep_polar $pto_polar] { }]"
    mol color ColorID 7
    mol representation VDW 1.4 12.0
    mol addrep top
    mol on top
}

# Negative
if {[llength [concat $pep_negative $pto_negative]] > 0} {
    mol selection "resname [join [concat $pep_negative $pto_negative] { }]"
    mol color ColorID 1
    mol representation VDW 1.4 12.0
    mol addrep top
    mol on top
}

# Aromatic
if {[llength [concat $pep_aromatic $pto_aromatic]] > 0} {
    mol selection "resname [join [concat $pep_aromatic $pto_aromatic] { }]"
    mol color ColorID 11
    mol representation VDW 1.4 12.0
    mol addrep top
    mol on top
}

# Positive
if {[llength [concat $pep_positive $pto_positive]] > 0} {
    mol selection "resname [join [concat $pep_positive $pto_positive] { }]"
    mol color ColorID 0
    mol representation VDW 1.4 12.0
    mol addrep top
    mol on top
}

# Backbone overlay
# Keep this only if your CG backbone beads are really called BB/BAS
mol selection "(resname [join $all_polymer_resnames { }]) and (name BB BAS)"
mol color ColorID 16
mol representation VDW 1.25 12.0
mol addrep top
mol on top

render snapshot snapshot_top.png
rotate x by -90
render snapshot snapshot_side.png
rotate x by -90
render snapshot snapshot_bottom.png
rotate x by 90

# ============================================================================
# WATER VIEW
# Remove all current reps first, then add only water
# ============================================================================

set nreps [molinfo top get numreps]
for {set i [expr {$nreps - 1}]} {$i >= 0} {incr i -1} {
    mol delrep $i top
}

mol selection "resname [join $water_resnames { }]"
mol color ColorID 17
mol representation VDW 1.2 12.0
mol addrep top
mol on top

render snapshot snapshot_water.png

exit
