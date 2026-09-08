# =============================================================================
# VMD PFPEPTOIDS INTERACTIVE VISUALISATION STYLE
# =============================================================================
# Purpose:
#   Apply the established PFPeptoids bilayer/polymer colour scheme to the
#   molecule that is currently "top" in VMD.
#
# Behaviour:
#   - Does not load a structure or trajectory.
#   - Does not rotate the view automatically.
#   - Does not render files automatically.
#   - Does not close VMD.
#   - Keeps the bilayer and polymer representations available interactively.
#
# Use from Extensions -> Tk Console:
#   source "/full/path/to/VMD_PFPeptoids_Interactive_Style.tcl"
# =============================================================================

# =============================================================================
# USER OPTIONS
# =============================================================================

# Remove all existing graphical representations before applying this style.
set CLEAR_EXISTING_REPS 1

# Wrap residues into the periodic box. This changes coordinates in VMD memory,
# but does not alter the original PSF/PDB/trajectory files on disk.
set APPLY_PBC_WRAP 1

# Reset the camera after applying the representations.
set RESET_VIEW 1

# Optional representations.
set SHOW_WATER 0
set SHOW_IONS 0
set SHOW_BACKBONE 0

# Optional visual-only bilayer translation. Normally leave disabled.
set SHIFT_BILAYER_VIEW 0
set BILAYER_TARGET_Z_FRACTION 0.125

# =============================================================================
# HELPER PROCEDURES
# =============================================================================

proc geom_center {selection} {
    set n_atoms [$selection num]
    if {$n_atoms == 0} {
        error "geom_center received an empty atom selection"
    }

    set gc [veczero]
    foreach coord [$selection get {x y z}] {
        set gc [vecadd $gc $coord]
    }
    return [vecscale [expr {1.0 / $n_atoms}] $gc]
}

proc add_rep_if_present {molid selection_text color_args representation_args label} {
    set check_selection [atomselect $molid $selection_text]
    set atom_count [$check_selection num]
    $check_selection delete

    if {$atom_count == 0} {
        puts "SKIPPED: $label (no matching atoms)"
        return 0
    }

    mol selection $selection_text
    mol color {*}$color_args
    mol representation {*}$representation_args
    mol material Opaque
    mol addrep $molid

    puts "ADDED:   $label ($atom_count atoms)"
    return 1
}

# =============================================================================
# REQUIRE A LOADED MOLECULE
# =============================================================================

set loaded_molecules [molinfo list]
if {[llength $loaded_molecules] == 0} {
    error "No molecule is loaded. Load the PSF and coordinates/trajectory first."
}

set molid [molinfo top]
puts ""
puts "Applying PFPeptoids interactive style to molecule $molid: [molinfo $molid get name]"

# =============================================================================
# CENTRAL DEFINITIONS
# =============================================================================

# Bilayer / solvent / ions
set bilayer_resnames {POPC POPS POPE POPG DPPC DTPC CHOL}
set water_resnames   {H2O W SW TW WF}
set ion_resnames     {NA CL NA+ CL- ION}

# Peptide residue groups
set pep_hydrophobic {ALAC ALAN ALA VALC VALN VAL ILEC ILEN ILE LEUC LEUN LEU METC METN MET CYSC CYSN CYS}
set pep_proline     {PROC PRON PRO}
set pep_polar       {SERC SERN SER THRC THRN THR ASNC ASNN ASN GLNC GLNN GLN HSEC HSEN HSE}
set pep_negative    {ASPC ASPN ASP GLUC GLUN GLU}
set pep_aromatic    {PHEC PHEN PHE TYRC TYRN TYR TRPC TRPN TRP}
set pep_positive    {LYSC LYSN LYS ARGC ARGN ARG}

# Peptoid residue groups.
# Both current project names and older topology aliases are retained.
set pto_hydrophobic {Na Nab Ni Nl Nm Nv}
set pto_polar       {Nn Nq Ns Nse Nt NmO}
set pto_negative    {Nd Ne}
set pto_aromatic    {Nf Nfn Nfe Nfex Nw Nwe Ny Nfe4Cl Nfe4Br Nfnaph NfeC4 NfeB4 Nfnap}
set pto_positive    {Nk Nke Nr Nkeqm}

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
set all_known_resnames [concat \
    $bilayer_resnames \
    $water_resnames \
    $ion_resnames \
    $all_polymer_resnames]

# =============================================================================
# CHECK RESIDUE-NAME COVERAGE
# =============================================================================

set All [atomselect $molid "all"]
set observed_resnames [lsort -unique [$All get resname]]
set unclassified_resnames {}

foreach residue_name $observed_resnames {
    if {[lsearch -exact $all_known_resnames $residue_name] < 0} {
        lappend unclassified_resnames $residue_name
    }
}

if {[llength $unclassified_resnames] > 0} {
    puts ""
    puts "WARNING: residue names not classified by this style:"
    foreach residue_name $unclassified_resnames {
        puts "  $residue_name"
    }
    puts "These residues will not appear in the polymer representations."
} else {
    puts "Residue-name coverage check: PASS"
}

# =============================================================================
# OPTIONAL VISUAL-ONLY BILAYER Z SHIFT
# =============================================================================

set bilayer_selection_text "resname [join $bilayer_resnames { }]"
set Bilayer [atomselect $molid $bilayer_selection_text]

if {$SHIFT_BILAYER_VIEW == 1} {
    set boxinfo [molinfo $molid get {a b c}]
    set box_z [lindex $boxinfo 2]

    if {[$Bilayer num] > 0 && $box_z > 0} {
        set bilayer_center [geom_center $Bilayer]
        set bilayer_z [lindex $bilayer_center 2]
        set target_z [expr {$box_z * $BILAYER_TARGET_Z_FRACTION}]
        set dz [expr {$target_z - $bilayer_z}]

        puts ""
        puts "VISUAL BILAYER SHIFT:"
        puts "  box_z     = $box_z"
        puts "  bilayer_z = $bilayer_z"
        puts "  target_z  = $target_z"
        puts "  dz        = $dz"

        $All moveby [list 0.0 0.0 $dz]
    } else {
        puts "WARNING: bilayer shift requested, but no bilayer atoms or valid z box length were found."
    }
}

# =============================================================================
# PERIODIC-BOUNDARY WRAPPING
# =============================================================================

if {$APPLY_PBC_WRAP == 1} {
    if {[catch {package require pbctools} pbctools_error]} {
        puts "WARNING: pbctools could not be loaded; periodic wrapping was skipped."
        puts "         $pbctools_error"
    } else {
        if {[catch {pbc wrap -all -compound residue -molid $molid} wrap_error]} {
            puts "WARNING: pbc wrap failed: $wrap_error"
        } else {
            puts "Periodic wrapping: applied to all loaded frames"
        }
    }
}

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

axes location off
color Display Background white
display projection Orthographic
display depthcue off

# Light-blue water colour.
color change rgb 17 0.72 0.86 1.00

# =============================================================================
# REMOVE EXISTING REPRESENTATIONS
# =============================================================================

if {$CLEAR_EXISTING_REPS == 1} {
    set nreps [molinfo $molid get numreps]
    for {set rep_index [expr {$nreps - 1}]} {$rep_index >= 0} {incr rep_index -1} {
        mol delrep $rep_index $molid
    }
    puts "Existing representations: removed"
}

mol on $molid

# =============================================================================
# BILAYER REPRESENTATION
# =============================================================================

add_rep_if_present \
    $molid \
    $bilayer_selection_text \
    {ColorID 8} \
    {QuickSurf 1.2 0.3 1.0 2} \
    "Bilayer"

# =============================================================================
# POLYMER REPRESENTATIONS
# Full residues are shown rather than backbone beads alone.
# =============================================================================

set hydrophobic_selection "resname [join [concat $pep_hydrophobic $pto_hydrophobic] { }]"
set proline_selection     "resname [join $pep_proline { }]"
set polar_selection       "resname [join [concat $pep_polar $pto_polar] { }]"
set negative_selection    "resname [join [concat $pep_negative $pto_negative] { }]"
set aromatic_selection    "resname [join [concat $pep_aromatic $pto_aromatic] { }]"
set positive_selection    "resname [join [concat $pep_positive $pto_positive] { }]"
set backbone_selection    "(resname [join $all_polymer_resnames { }]) and (name BB BAS)"

add_rep_if_present $molid $hydrophobic_selection {ColorID 3}  {VDW 1.4 12.0} "Hydrophobic polymer residues"
add_rep_if_present $molid $proline_selection     {ColorID 2}  {VDW 1.4 12.0} "Proline residues"
add_rep_if_present $molid $polar_selection       {ColorID 7}  {VDW 1.4 12.0} "Polar polymer residues"
add_rep_if_present $molid $negative_selection    {ColorID 1}  {VDW 1.4 12.0} "Negatively charged polymer residues"
add_rep_if_present $molid $aromatic_selection    {ColorID 11} {VDW 1.4 12.0} "Aromatic polymer residues"
add_rep_if_present $molid $positive_selection    {ColorID 0}  {VDW 1.4 12.0} "Positively charged polymer residues"

if {$SHOW_BACKBONE == 1} {
    add_rep_if_present $molid $backbone_selection {ColorID 16} {VDW 1.25 12.0} "Polymer backbone overlay"
}

# =============================================================================
# OPTIONAL WATER AND ION REPRESENTATIONS
# These are additive; they do not remove the bilayer or polymer reps.
# =============================================================================

if {$SHOW_WATER == 1} {
    set water_selection "resname [join $water_resnames { }]"
    add_rep_if_present $molid $water_selection {ColorID 17} {VDW 1.2 12.0} "Water"
}

if {$SHOW_IONS == 1} {
    set ion_selection "resname [join $ion_resnames { }]"
    add_rep_if_present $molid $ion_selection {Name} {VDW 1.2 12.0} "Ions"
}

# =============================================================================
# FINALISE INTERACTIVE VIEW
# =============================================================================

if {$RESET_VIEW == 1} {
    display resetview
}

$Bilayer delete
$All delete

puts ""
puts "PFPeptoids interactive style applied successfully."
puts "VMD remains open for manual positioning and rendering."
puts "Quick image: render snapshot custom_view.png"
puts "Higher quality: File -> Render -> Tachyon"
puts ""

