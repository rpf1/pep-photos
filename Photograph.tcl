proc geom_center {selection} {
        # set the geometrical center to 0
        set gc [veczero]
        # [$selection get {x y z}] returns a list of {x y z} 
        #    values (one per atoms) so get each term one by one
        foreach coord [$selection get {x y z}] {
           # sum up the coordinates
           set gc [vecadd $gc $coord]
        }
        # and scale by the inverse of the number of atoms
        return [vecscale [expr 1.0 /[$selection num]] $gc]
}



mol new AAAADEMC_80.psf type psf
mol addfile AAAADEMC_80.pdb type pdb waitfor all

mol delrep  0 top

set Bilayer [atomselect top "resname POPC POPS POPE POPG DPPC DTPC CHOL"]
set Peptide [atomselect top "not resname POPC POPS POPE POPG DPPC DTPC CHOL W SW TW H2O NA CL NA+ CL- ION"]
set All [atomselect top "all"]
set W [atomselect top "resname H2O W SW TW"]
set COG [geom_center $Peptide]

#set Box [pbc get -all]
#$Peptide moveby {10 10 0}

#pbc readxst cphmd_out.xst -molid top -all
pbc wrap -all

axes location off
color Display Background white

# pbc box -color black
display projection Orthographic

display depthcue   off

mol selection resname POPC POPS POPE POPG DTPC DPPC CHOL
mol color Charge
mol representation QuickSurf 1.200000 0.300000 1.000000 2
mol addrep top
mol on top

render snapshot snapshot_top_bilayer.png
rotate x by -90
render snapshot snapshot_side_bilayer.png
rotate x by 90

mol selection resname ALAC ALAN ALA VALC VALN VAL ILEC ILEN ILE LEUC LEUN LEU METC METN MET CYSC CYSN CYS
mol color ColorID 3 ; orange lipophilic
mol representation VDW 1.000000 12.00000
mol addrep top
mol on top

mol selection resname PROC PRON PRO
mol color ColorID 2 ; grey proline only?
mol representation VDW 1.000000 12.00000
mol addrep top
mol on top

mol selection resname SERC SERN SER THRC THRN THR ASNC ASNN ASN GLNC GLNN GLN HSEC HSEN HSE
mol color ColorID 7 ; green Polar
mol representation VDW 1.000000 12.00000
mol addrep top
mol on top

mol selection resname ASPC ASPN ASP GLUC GLUN GLU
mol color ColorID 1 ; red negative
mol representation VDW 1.000000 12.00000
mol addrep top
mol on top

mol selection resname PHEC PHEN PHE TYRC TYRN TYR TRPC TRPN TRP
mol color ColorID 11 ; purple aromatic
mol representation VDW 1.000000 12.00000
mol addrep top
mol on top

mol selection resname LYSC LYSN LYS ARGC ARGN ARG
mol color ColorID 0 ; blue positive
mol representation VDW 1.000000 12.00000
mol addrep top
mol on top

mol selection name BAS BB
mol color ColorID 16 ; black back bone
mol representation VDW 1.000000 12.00000
mol addrep top
mol on top

#display resize 500 1000


render snapshot snapshot_top.png
rotate x by -90
render snapshot snapshot_side.png

rotate x by -90
render snapshot snapshot_bottom.png



rotate x by 90
mol delrep 0 top ; removes the first thing, in this case the bilayer
mol selection resname H2O W SW TW
mol color ColorID 17
mol representation VDW 1.000000 12.00000
mol addrep top
mol on top
render snapshot snapshot_water.png


#display update
exit
