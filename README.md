# pep-photos
"""
Photograph.py

Created on Mon Jan  8 22:54:47 2024

@author: Orignally rkb19187 modified by pfb19164

Purpose
-------
Batch-generate VMD snapshot panels for PFPeptoids systems.

What it does
------------
1. Scans the current PFPeptoids folder layout:
   Boxes/2.1/{bilayer}/run_x/{molecule}_{nmol}

2. Chooses trajectory source with strict logic:
   - Prefer Relax.gro + Relax.xtc if they exist
   - Only fall back to Pull.gro + Pull.xtc if:
       ALLOW_PULL_FALLBACK = True
     and Relax files do not exist

3. Checks that the chosen trajectory has reached a minimum final time:
   - Relax: MIN_RELAX_TIME_PS
   - Pull : MIN_PULL_TIME_PS

4. Writes a system-specific Tcl file based on Photograph.tcl

5. Runs VMD to render:
   snapshot_top_bilayer.png
   snapshot_side_bilayer.png
   snapshot_top.png
   snapshot_side.png
   snapshot_bottom.png
   snapshot_water.png

6. Trims whitespace around those PNGs using ImageMagick `convert`
   so the panels pack tightly and consistently

7. Creates:
   a) One horizontal stitched image per system
   b) One vertical stack per molecule/run combining all bilayers

Why trimming is used
--------------------
VMD snapshots often contain large white borders/background margins.
Trimming removes empty whitespace so:
- stitched panels align better
- final images waste less space
- output looks much cleaner for reports/slides

Notes
-----
- Generated Tcl files are stored alongside each system's photo outputs
  inside the analysis/Photos tree, not inside the raw simulation folders.
- If you want Pull fallback, set ALLOW_PULL_FALLBACK = True manually.
- This script expects ImageMagick `convert` and VMD to be available.
"""
