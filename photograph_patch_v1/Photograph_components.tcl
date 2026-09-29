proc geom_center {selection} {
    set gc [veczero]

    foreach coord [$selection get {x y z}] {
        set gc [vecadd $gc $coord]
    }

    return [vecscale [expr 1.0 / [$selection num]] $gc]
}


# ============================================================================
# LOAD SYSTEM
# ============================================================================

mol new {__GRO_FILE__} type gro
mol addfile {__XTC_FILE__} type xtc waitfor all

# Explicitly use the final loaded trajectory frame.
animate goto end


# ============================================================================
# OPTIONAL VISUAL-ONLY BILAYER Z SHIFT
# ============================================================================

set SHIFT_BILAYER_VIEW __SHIFT_BILAYER_VIEW__
set BILAYER_TARGET_Z_FRACTION __BILAYER_TARGET_Z_FRACTION__


# ============================================================================
# REMOVE DEFAULT VMD REPRESENTATION
# ============================================================================

mol delrep 0 top


# ============================================================================
# CENTRAL DEFINITIONS
# ============================================================================

# Full set of membrane residue names used by PFPeptoids.
set bilayer_resnames {
    POPC
    POPS
    POPE
    POPG
    DPPC
    DTPC
    CHOL
}

set water_resnames {
    H2O
    W
    SW
    TW
    WF
}

set ion_resnames {
    NA
    CL
    NA+
    CL-
    ION
}


# ============================================================================
# COMPONENT-COLOURED MEMBRANE DEFINITIONS
# ============================================================================

set component_1_resnames {
    __COMPONENT_1_RESNAME__
}

set component_2_resnames {
    __COMPONENT_2_RESNAME__
}

set component_1_color_id __COMPONENT_1_COLOR_ID__
set component_2_color_id __COMPONENT_2_COLOR_ID__


# Custom diagnostic colours.
#
# These are deliberately separated from the standard Photograph.tcl
# polymer colours.

color change rgb __COMPONENT_1_COLOR_ID__ __COMPONENT_1_RGB__
color change rgb __COMPONENT_2_COLOR_ID__ __COMPONENT_2_RGB__


# ============================================================================
# PEPTIDE RESIDUE GROUPS
# ============================================================================

# ============================================================================
# PEPTOID RESIDUE GROUPS
# ============================================================================

# ============================================================================
# COMBINED POLYMER DEFINITIONS
# ============================================================================

source [file join [file dirname [info script]] photograph_residues.tcl]

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

set all_polymer_resnames [concat \
    $peptide_resnames \
    $peptoid_resnames]


# ============================================================================
# USEFUL ATOM SELECTIONS
# ============================================================================

# Fail closed on every residue outside the explicit environment/polymer lists.
set photo_known [concat $bilayer_resnames $water_resnames $ion_resnames $all_polymer_resnames]
set photo_unknown {}
foreach rn [lsort -unique [[atomselect top all] get resname]] {
    if {[lsearch -exact $photo_known $rn] < 0} {lappend photo_unknown $rn}
}
if {[llength $photo_unknown] > 0} {
    puts stderr "FATAL: unsupported GRO residues: $photo_unknown; no snapshots rendered"
    exit 2
}
set photo_polymer [atomselect top "resname [join $all_polymer_resnames { }]"]
if {[$photo_polymer num] == 0} {
    puts stderr "FATAL: no recognized polymer atoms; no snapshots rendered"
    exit 2
}
puts "PHOTO_POLYMER_ATOMS=[$photo_polymer num]"
$photo_polymer delete

set Bilayer [
    atomselect top \
    "resname [join $bilayer_resnames { }]"
]

set W [
    atomselect top \
    "resname [join $water_resnames { }]"
]

set All [
    atomselect top "all"
]

set Polymer [
    atomselect top \
    "resname [join $all_polymer_resnames { }]"
]


# ============================================================================
# OPTIONAL VISUAL BILAYER SHIFT
# ============================================================================

if {$SHIFT_BILAYER_VIEW == 1} {

    set boxinfo [
        molinfo top get {a b c}
    ]

    set box_z [
        lindex $boxinfo 2
    ]

    if {
        [$Bilayer num] > 0
        &&
        $box_z > 0
    } {

        set bilayer_center [
            geom_center $Bilayer
        ]

        set bilayer_z [
            lindex $bilayer_center 2
        ]

        set target_z [
            expr {
                $box_z
                *
                $BILAYER_TARGET_Z_FRACTION
            }
        ]

        set dz [
            expr {
                $target_z
                -
                $bilayer_z
            }
        ]

        puts "VISUAL SHIFT DEBUG:"
        puts "  box_z      = $box_z"
        puts "  bilayer_z  = $bilayer_z"
        puts "  target_z   = $target_z"
        puts "  dz         = $dz"

        $All moveby [
            list 0.0 0.0 $dz
        ]

        pbc wrap \
            -all \
            -compound residue
    }
}


# ============================================================================
# POLYMER CENTRE
# ============================================================================

if {[$Polymer num] > 0} {
    set COG [
        geom_center $Polymer
    ]
}


# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

pbc wrap -all

axes location off

color Display Background white

display projection Orthographic

display depthcue off


# Water colour used by standard Photograph.tcl.
color change rgb 17 0.72 0.86 1.00


# ============================================================================
# COMPONENT-COLOURED BILAYER
# ============================================================================
#
# IMPORTANT:
#
# The standard Photograph.tcl uses ONE representation:
#
#     all membrane residues
#         -> QuickSurf
#
# Here we create TWO independent QuickSurf representations:
#
#     membrane component 1
#         -> QuickSurf
#         -> component colour 1
#
#     membrane component 2
#         -> QuickSurf
#         -> component colour 2
#
# The QuickSurf parameters are intentionally IDENTICAL to the standard
# Photograph.tcl:
#
#     QuickSurf 1.2 0.3 1.0 2
#
# Therefore the membrane geometry/look remains directly comparable.
# ============================================================================


# Component 1

mol selection \
    "resname [join $component_1_resnames { }]"

mol color \
    ColorID $component_1_color_id

mol representation \
    QuickSurf 1.2 0.3 1.0 2

mol addrep top

mol on top


# Component 2

mol selection \
    "resname [join $component_2_resnames { }]"

mol color \
    ColorID $component_2_color_id

mol representation \
    QuickSurf 1.2 0.3 1.0 2

mol addrep top

mol on top


# ============================================================================
# BILAYER-ONLY VIEWS
# ============================================================================

# Save the original top orientation; each view starts from this matrix.
set photo_top_rotation [lindex [molinfo top get rotate_matrix] 0]
proc photo_view {axis angle filename} {
    global photo_top_rotation
    molinfo top set rotate_matrix [list $photo_top_rotation]
    if {$angle != 0} {rotate $axis by $angle}
    # Align the Y-view membrane horizontally; retain orthogonality.
    if {$axis eq "y"} {rotate z by -90}
    render snapshot $filename
}

photo_view x -180 snapshot_bottom_bilayer.png
photo_view x 0 snapshot_top_bilayer.png
photo_view x -90 snapshot_side_bilayer.png
photo_view y -90 snapshot_side_y_bilayer.png


# ============================================================================
# POLYMER REPRESENTATIONS
# ============================================================================
#
# Keep these equivalent to Photograph.tcl.
#
# The component-coloured bilayer representations remain active underneath.
# ============================================================================


# Hydrophobic

mol selection \
    "resname [join [concat $pep_hydrophobic $pto_hydrophobic] { }]"

mol color ColorID 3

mol representation VDW 1.4 12.0

mol addrep top

mol on top


# Proline

mol selection \
    "resname [join $pep_proline { }]"

mol color ColorID 2

mol representation VDW 1.4 12.0

mol addrep top

mol on top


# Polar

if {
    [llength [concat $pep_polar $pto_polar]] > 0
} {

    mol selection \
        "resname [join [concat $pep_polar $pto_polar] { }]"

    mol color ColorID 7

    mol representation VDW 1.4 12.0

    mol addrep top

    mol on top
}


# Negative

if {
    [llength [concat $pep_negative $pto_negative]] > 0
} {

    mol selection \
        "resname [join [concat $pep_negative $pto_negative] { }]"

    mol color ColorID 1

    mol representation VDW 1.4 12.0

    mol addrep top

    mol on top
}


# Aromatic

if {
    [llength [concat $pep_aromatic $pto_aromatic]] > 0
} {

    mol selection \
        "resname [join [concat $pep_aromatic $pto_aromatic] { }]"

    mol color ColorID 11

    mol representation VDW 1.4 12.0

    mol addrep top

    mol on top
}


# Positive

if {
    [llength [concat $pep_positive $pto_positive]] > 0
} {

    mol selection \
        "resname [join [concat $pep_positive $pto_positive] { }]"

    mol color ColorID 0

    mol representation VDW 1.4 12.0

    mol addrep top

    mol on top
}


# Backbone overlay

mol selection \
    "(resname [join $all_polymer_resnames { }]) and (name BB BAS)"

mol color ColorID 16

mol representation VDW 1.25 12.0

mol addrep top

mol on top


# ============================================================================
# POLYMER + COMPONENT-COLOURED BILAYER VIEWS
# ============================================================================

photo_view x 0 snapshot_top.png
photo_view x -90 snapshot_side.png
photo_view y -90 snapshot_side_y.png
photo_view x -180 snapshot_bottom.png


# ============================================================================
# WATER VIEW
# ============================================================================
#
# Delete every current representation so the water image remains equivalent
# to Photograph.tcl and is not contaminated by the membrane/polymer reps.
# ============================================================================

set nreps [
    molinfo top get numreps
]

for {
    set i [expr {$nreps - 1}]
} {
    $i >= 0
} {
    incr i -1
} {

    mol delrep $i top
}


mol selection \
    "resname [join $water_resnames { }]"

mol color ColorID 17

mol representation VDW 1.2 12.0

mol addrep top

mol on top


photo_view x -90 snapshot_water.png
photo_view y -90 snapshot_water_y.png


# ============================================================================
# FINISH
# ============================================================================

exit
