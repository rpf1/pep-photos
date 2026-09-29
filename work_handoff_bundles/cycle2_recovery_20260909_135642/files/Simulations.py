# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:20:36 2025

@author: bwb16179
"""
# remember if you're having issues with submitting jobs, especially to do with C-rescale, it's likely you have NOT module loaded the gcc version and instead have the intel version (which is also the default)
# before running, make sure this spyder terminal has the correct version of gromacs runnings, then run it

import inspect
import Bio.PDB
from PeptideBuilder import Geometry
import PeptideBuilder
import os
import time
import shutil
import argparse
import pandas
import math
import sys
import glob
import datetime
import io
import re
import json
import numpy as np
import peptideutils as pu
import MDAnalysis as mda
from sklearn.metrics import euclidean_distances
from inspect import currentframe, getframeinfo
from collections import OrderedDict
import subprocess

import matplotlib.pyplot as plt

# Our independent modules
import Martinoid
import hpctools

# Our local modules
try:
    from ._CBox import *
except:
    from _CBox import *

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# ####
# # debugging helpers
# 
# print("\n--- IMPORT SOURCE CHECK ---")
# print("Simulations.py =", os.path.abspath(__file__))
# 
# try:
#     import _CBox
#     print("_CBox.py       =", inspect.getfile(_CBox))
# except Exception as e:
#     print("_CBox inspect failed:", e)
# 
# print("mkpdb source   =", inspect.getsourcefile(mkpdb))
# print("vmd_atomistic source =", inspect.getsourcefile(vmd_atomistic))
# print("replace_in_file2 source =", inspect.getsourcefile(replace_in_file2))
# print("--------------------------\n")
# 
# print("dotenv loaded CMLROOT =", os.environ.get("CMLROOT"))
# print("current cwd after load_dotenv =", os.getcwd())
# ####
# =============================================================================


# from PullAnalysis import *
# from CPHMD_restart import *

def pprint(string):
    cf = currentframe()
    print(f"Line: {cf.f_back.f_lineno} - {string}")


def CleanJSON(json):
    json = json.replace("\n", "")
    # get rid of trailing comma which json doesn't like
    clean = json.replace("},}", "}}")
    return clean


def getAPL(gro, xtc, L=8):
    folder = os.path.dirname(gro)
    traj = mda.Universe(gro, xtc, in_memory=True)
    jobtype = os.path.basename(xtc).replace(".xtc", "")
    # PC = traj.select_atoms("resname POPC POPS DPPC")
    PO4 = traj.select_atoms("name PO4")
    nLipids = len(PO4)
    APL = pandas.DataFrame(columns=["APL", "zPL",  "lipids/2"])

    for frame in traj.trajectory:
        PO4.wrap()
        x = PO4.wrap()[:, 0].max() - PO4.wrap()[:, 0].min()
        y = PO4.wrap()[:, 1].max() - PO4.wrap()[:, 1].min()
        z = PO4.wrap()[:, 2].max() - PO4.wrap()[:, 2].min()
        APL.loc[np.ceil(frame.time)] = [
            (x*y)/(nLipids/2), z/nLipids, nLipids/2]
    return APL


class Simulations:

    # NOTE that Na is Nx, this is because of the way martinoid works -> Ng is not described.
    peptoid_resnames = "Nx Nab Nd Ne Nf Nfe NfeB4 NfeC4 Nfex Nfn Nfnap Ni Nk Nke Nkeqm Nl Nm NmO Nn Nq Nr Ns Nse Nt Nv Nw Nwe Ny"   #remember that "Nx" is "Na"
    peptide_resnames = "ALA ALAC ALAN CYS CYSC CYSN ASP ASPC ASPN GLU GLUC GLUN PHE PHEC PHEN GLY GLYC GLYN HIS HISC HISN HSE HSEC HSEN HSP HSPC HSPN ILE ILEC ILEN LYS LYSC LYSN LEU LEUC LEUN ASN ASNC ASNN MET METC METN PRO PROC PRON GLN GLNC GLNN ARG ARGC ARGN SER SERC SERN THR THRC THRN VAL VALC VALN TRP TRPC TRPN TYR TYRC TYRN"
    
    @property
    def pull_mdp(self):
        return self.bilayer_data["pulling_mdp"]

    @property
    def relax_mdp(self):
        return self.bilayer_data["relax_mdp"]

    @property
    def preptide_min_mdp(self):
        return f"{self.root}/minpreptide.mdp"

    @property
    def preptide_eq_mdp(self):
        return f"{self.root}/1nspreptide.mdp"

    @property
    def boxes(self):
        if str(self.FF) == "3":
            folder = f"{self.root}/Boxes/3/{self.water}"
            os.makedirs(folder, exist_ok=True)
            return folder
        elif self.FF == "2.2P":
            print(f"Forcefield {self.FF}, Not implemented")
            sys.exit()
        elif self.FF == "2.1":
            folder = f"{self.root}/Boxes/2.1"
            os.makedirs(folder, exist_ok=True)
            return folder
        else:
            print(f"Forcefield {self.FF}, Not implemented")
            sys.exit()

    @property
    def water_box(self):
        if self.water == "TW":
            return "/users/rkb19187/Documents/TW_box/TW_Box_303K.gro"
        elif self.water == "SW":
            return "/users/rkb19187/Documents/SW_box/SW_Box_303K.gro"
        elif self.water == "W":
            return f"{self.root}/water_125A.gro"
        elif self.water == "W_WF":
            return f"{self.root}/water_anti_125A.gro"

    @property
    def peptide_folder(self):
        return f"{self.boxes}/{self.bilayer}/run_{self.run}/{self.peptide}_{self.nmol}"

    @property
    def preptide_folder(self):
        if self.FF == "3":
            return os.path.join(os.path.dirname(self.boxes), "Preptide") + f"/{self.peptide}"
        else:
            return os.path.join(self.boxes, "Preptide") + f"/{self.peptide}"

    @property
    def bilayer_folder(self):
        if self.FF == "3":
            return f"{self.root}/Reference/{self.FF}/{self.water}/{self.bilayer}"
        else:
            return f"{self.root}/Reference/{self.FF}/bilayers/{self.bilayer}"

    @property
    def bilayer_gro(self):
        meta_route = self.bilayer_data["eq_file"]
        meta_route = f"{self.bilayer_folder}/{meta_route}"
        assert os.path.exists(
            meta_route), "Cannot find the reference systems meta.json"
        if os.path.exists(meta_route):
            return meta_route

    @property
    def ntomp(self):
        try:
            return os.environ["OMP_NUM_THREADS"]
        except KeyError:
            return 1

    def internal_checks(self):
        assert os.path.exists(self.bilayer_folder), f"Missing bilayer folder: {self.bilayer_folder}"
        assert os.path.exists(self.pull_mdp), f"Missing pulling mdp: {self.pull_mdp}"
        assert os.path.exists(self.relax_mdp), f"Missing relax mdp: {self.relax_mdp}"
        assert os.path.exists(self.preptide_min_mdp), f"Missing preptide minimisation mdp: {self.preptide_min_mdp}"
        assert os.path.exists(self.preptide_eq_mdp), f"Missing preptide equilibration mdp: {self.preptide_eq_mdp}"
        assert os.path.exists(os.path.join(f"{self.bilayer_folder}", "meta.json")), f"Missing meta.json in: {self.bilayer_folder}"
    
    def martinize(self, peptide):
        print("martinize() sees GMXDATA =", os.environ.get("GMXDATA"))
    
        SS = "H" * len(peptide)
    
        if self.FF == "3":
            cmd = (
                f"martinize2 "
                f"-f {peptide}_aa.pdb "
                f"-o {peptide}.top "
                f"-x {peptide}.pdb "
                f"-ff martini3001 "
                f"-ss {SS}"
            )
    
        elif self.FF == "2.2P":
            cmd = (
                f"python {self.root}/martinize.py "
                f"-f {peptide}_aa.pdb "
                f"-name {peptide} "
                f"-o {peptide}.top "
                f"-x {peptide}.pdb "
                f"-ff martini22p "
                f"-ss {SS}"
            )
    
        elif self.FF == "2.1":
            cmd = (
                f"python {self.root}/martinize.py "
                f"-f {peptide}_aa.pdb "
                f"-name {peptide} "
                f"-o {peptide}.top "
                f"-x {peptide}.pdb "
                f"-ff martini21 "
                f"-ss {SS}"
            )
    
        else:
            raise ValueError(f"Unknown force field: {self.FF}")
    
        print(cmd)
        os.system(cmd)

    def Preptide(self):
        # Check if it is already done
        if os.path.exists(f"{self.preptide_folder}/{self.peptide}.gro"):
            print("Preptide for:", self.peptide, "already done")
            return 1

        os.makedirs(f"{self.preptide_folder}", exist_ok=True)
        os.chdir(f"{self.preptide_folder}")

        if self.peptoid:
            peptoid = Martinoid.Martinoid(sequence=self.peptide, N_ter_charged=True, C_ter_charged=False, C_ter_protect=True, SS="linear")  # SS = "linear" or "helical"
        else:
            if not os.path.exists(f"{self.peptide}.itp"):
                print(os.path.abspath("."))
                mkpdb(self.peptide)
                shutil.copy(self.root+"/structure_gen.tcl",
                            self.peptide+".pgn")

                replace_in_file2(self.peptide+".pgn",
                                 "self.peptide", self.peptide)
                replace_in_file2(self.peptide+".pgn", "DIR", self.root)
                replace_in_file2(self.peptide+".pgn", "HIS", "HSE")
                os.system(vmd_atomistic(self.vmd, self.peptide))
                os.system(
                    "sed -i \"s/{}/{}/g\" {}_aa.pdb".format("HSE", "HIS", self.peptide))
                os.remove(self.peptide+".pgn")
                print("DEBUG GMXDATA =", os.environ.get(
                    "GMXDATA"))  # ADDED FOR DEBUGGING
                # ADDED FOR DEBUGGING
                print("DEBUG PATH =", os.environ.get("PATH"))
                # ADDED FOR DEBUGGING
                print("DEBUG cwd =", os.getcwd())
                self.martinize(self.peptide)
                try:
                    os.rename(f"{self.peptide}_A.itp", f"{self.peptide}.itp")
                except:
                    os.rename(f"molecule_0.itp", f"{self.peptide}.itp")

        # local min of peptoid/peptide
        if not os.path.exists(f"{self.peptide}_solomin.gro"):
            print("BBB:", self.root)

            if self.water == 'W_WF':
                replace = "W"
            else:
                replace = self.water

            # cmd = f"gmx_mpi insert-molecules -f {self.water_box} -ci {self.peptide}.pdb -ip {self.root}/center_125A.dat -radius 0.2 -replace W -o Solo_water"
            cmd = f"gmx_mpi insert-molecules -f {self.water_box} -ci {self.peptide}.pdb -radius 0.2 -replace {replace} -nmol 1 -o Solo_water"
            print(cmd)
            print("AAA:", os.path.abspath("."))
            os.system(cmd)
            indie_topol = readin(f"{self.root}/topol{self.FF}.top")
            indie_topol = indie_topol.replace("ROOT", self.root)
            indie_topol = indie_topol.replace(
                "molecule_0.itp", f"{self.peptide}.itp")
            indie_topol = indie_topol.replace("EXTRAITPS", "")
            indie_topol = indie_topol.split("[ molecules ]\n")
            print(indie_topol)
            if self.water == 'W_WF':
                water_nums = count_water("Solo_water.gro")
                indie_topol[1] = "W" + " " + str(water_nums['W']) + "\n"
                indie_topol[1] = indie_topol[1] + "WF" + \
                    " " + str(water_nums['WF']) + "\n"
            else:
                indie_topol[1] = self.water + " " + \
                    str(count_water("Solo_water.gro"))+"\n"
            if self.peptoid:
                indie_topol[1] = indie_topol[1] + f"Peptoid       1"
            elif self.FF == "3":
                indie_topol[1] = indie_topol[1] + f"molecule_0       1"
            else:
                indie_topol[1] = indie_topol[1] + f"{self.peptide}_A       1"
            indie_topol = "[ molecules ]\n".join(indie_topol)
            indie_topol = indie_topol.replace("Protein", f"{self.peptide}")
            ###############################################
            with open("topol_solo.top", 'w') as solo_topol:
                solo_topol.write(indie_topol)
            print(os.path.abspath("."))
            os.system(f"gmx_mpi grompp -f {self.preptide_min_mdp} -c Solo_water.gro -o {self.peptide}_solomin -p topol_solo.top -maxwarn 1")
            # sys.exit()
            if self.peptoid:
                min_cmd = f"mpirun -np 1 gmx_mpi mdrun -v -ntomp {self.ntomp} -tableb {self.root}/angle5_a0.xvg -deffnm {self.peptide}_solomin"
            else:
                min_cmd = f"mpirun -np 1 gmx_mpi mdrun -v -ntomp {self.ntomp} -deffnm {self.peptide}_solomin"
            os.system(min_cmd)
            # sys.exit()
            cdel(glob.glob("*#*"))
        else:
            print(f"Found: {self.peptide}_solomin.gro")
        # if self.peptide == "VCVYWWRT": sys.exit()
        if not os.path.exists(f"{self.peptide}.gro"):
            minE = parsemin(f"{self.peptide}_solomin.log")
            if minE > 1000.0:
                print("SoloMin: Failed, minE > 1000.0")
                # DELETE EVERYTHING IN PREPTIDE FOLDER IF IT FAILS
                for file in glob.glob("*"):
                    os.remove(file)
                self.Preptide()
                os.chdir(self.root)
                return False
            else:
                ns = f"gmx_mpi grompp -f {self.preptide_eq_mdp} -c {self.peptide}_solomin.gro -p topol_solo.top -o {self.peptide}_soloeq -maxwarn 2"
                os.system(ns)
                if not os.path.exists(f"{self.peptide}_soloeq.tpr"):
                    print(f"SoloMin: Failed, {self.peptide}_soloeq.tpr not made")
                    os.chdir(self.root)
                    return False

                if self.peptoid:
                    run = f"mpirun -np 1 gmx_mpi mdrun -v -ntomp {self.ntomp} -tableb {self.root}/angle5_a0.xvg -deffnm {self.peptide}_soloeq"
                else:
                    run = f"mpirun -np 1 gmx_mpi mdrun -v -ntomp {self.ntomp} -deffnm {self.peptide}_soloeq"
                os.system(run)
                XtractPep(f"{self.peptide}_soloeq.gro", f"{self.peptide}.gro")
                cmd = f"gmx_mpi editconf -f {self.peptide}.gro -o {self.peptide}.gro -c -box 0 0 0"
                os.system(cmd)
                cdel(glob.glob("*#*"))
                os.chdir(self.root)
                return True
        else:
            print(f"Found: {self.peptide}.gro")
        os.chdir(self.root)

    def InsertionPositions(self):
        u = mda.Universe(self.bilayer_gro)
        Membrane = u.select_atoms(f"resname {self.phospholipids} {self.OuterMembrane} or name BB SC1 SC2 SC3")  # name BB SC1 ... to capture membrane proteins
        PO4 = Membrane.select_atoms("name PO4")
        # plt.scatter(PO4.positions[:,0], PO4.positions[:,2])
        W = u.select_atoms("resname W WF SW PW TW H2O")
        # Account for the much fatter E. Coli outer membrane
# =============================================================================
#         if self.bilayer == "GramNegative_OuterMembrane":
#             z_max = Membrane.positions[:,2].mean()+65
#             z_min = Membrane.positions[:,2].mean()+55
#         else:
#             z_max = Membrane.positions[:,2].mean()+30 # This isnt going to work for the fatter bacterial outer membrane
#             z_min = Membrane.positions[:,2].mean()+20
# =============================================================================

        z_min = Membrane.positions[:, 2].mean(
        )+self.bilayer_data["InsertionPositions"][0]
        # This isnt going to work for the fatter bacterial outer membrane
        z_max = Membrane.positions[:, 2].mean(
        )+self.bilayer_data["InsertionPositions"][1]

        x_max = W.positions[:, 0].max()
        y_max = W.positions[:, 1].max()

        X = np.linspace(0, x_max, 200)
        Y = np.linspace(0, y_max, 200)
        Z_top = np.linspace(z_min, z_max, 20)
        self.min_prot_dist = 3  # A
        positions = np.ndarray((0, 3))
        FailedAttempts = 0

        npositions = self.nmol
        if self.Rx:
            npositions += self.nmol

        while positions.shape[0] < npositions:
            # print(positions.shape, "/", nmol)
            xyz = np.array([np.random.choice(X), np.random.choice(Y), 0])
            z = np.random.choice(Z_top)
            xyz[2] = z
            if positions.shape[0] == 0:
                positions = np.vstack((positions, xyz))
                print(True)
            else:
                d = euclidean_distances(
                    xyz.reshape(1, -1), positions).flatten()
                if d.min() < self.min_prot_dist:
                    FailedAttempts += 1
                    if FailedAttempts > 1000:
                        positions = np.ndarray((0, 3))
                        FailedAttempts = 0
                    continue
                else:
                    positions = np.vstack((positions, xyz))
                    FailedAttempts = 0
# =============================================================================
#             if self.Rx:
#                 if positions.shape[0] == npositions/2:
#                     Z_top -= Membrane.positions[:,2].max()
# =============================================================================
# =============================================================================
#         plt.scatter(positions[:,0], positions[:,2])
#         plt.ylim(0, Membrane.positions[:,2].max())
#         plt.show()
# =============================================================================

        positions /= 10
        with open("positions.dat", 'w') as pos:
            for i in range(self.nmol):
                pos.write(" ".join([str(x.round(1))
                          for x in list(positions[i])]))
                pos.write("\n")
        if self.Rx:
            # diene = open("diene_positions.dat", 'w')
            Dimethyl_maleate = open("dieneophile_positions.dat", 'w')
            for i in range(self.nmol, npositions):
                Dimethyl_maleate.write(
                    " ".join([str(x.round(1)) for x in list(positions[i])]))
                Dimethyl_maleate.write("\n")

    def reset_box(self):
        for file in glob.glob(f"{self.peptide_folder}/*"):
            print("reset_box removing:", file)
            os.remove(file)

# =============================================================================
#     def WriteTopol(self):
#         print("Writing topology to:", os.path.abspath("topol.top"))
#         shutil.copy(f"{self.root}/topol{self.FF}.top", "topol.top")
#
#         U = mda.Universe(self.bilayer_gro)
#         PO4 = U.select_atoms("name PO4 ROH PA4 GM1 AM1 GL1") # Need ROH for CHOL since it doesnt have a PO4, PA4 for CDL1, GM1+AM1 for the mammalian plasma membrane
#         #Need extra selections to help get all the mammalian plasma membrane lipids
#         #PO4 = PO4 + U.select_atoms("resname DPCE DXCE PNCE XNCE and name AM1")
#         print("Bilayer composition:")
#         composition = OrderedDict()
#         # Account for membrane proteins
#         if self.bilayer in self.MembraneProteins:
#             for protein in self.MembraneProteins[self.bilayer]:
#                 composition[protein] = self.MembraneProteins[self.bilayer][protein]
#         n = 0
#         self.PO4 = PO4
#         useded_resids = [] # dont want to count the same phospholipd twice
#         for i, res in enumerate(PO4):
#             if res.resid in useded_resids:
#                 continue
#             else:
#                 useded_resids.append(res.resid)
#             if i == 0:
#                 composition[f"{res.resname}-{n}"] = 1
#             else:
#                 if list(composition.keys())[-1].split("-")[0] == res.resname:
#                     composition[f"{res.resname}-{n}"] += 1
#                 else:
#                     n+=1
#                     composition[f"{res.resname}-{n}"] = 1
#         print(composition)
#
#         if self.bilayer == "GramNegative_OuterMembrane":
#             composition_txt = "LPSR1O62      336\n"
#         else:
#             composition_txt = ""
#         for resname in composition:
#             num = composition[resname]
#             print(resname, num)
#             res = resname.split("-")[0]
#             composition_txt = composition_txt + f"{res}     {num}\n"
#
#         EXTRAITPS = ""
#         for itp in self.bilayer_data["itps"]:
#             if itp[:2] == "./":
#                 itp = os.path.join(self.bilayer_folder, itp)
#             EXTRAITPS = EXTRAITPS + '#include "' + itp + '" \n'
#         replace_in_file2("topol.top", "EXTRAITPS", EXTRAITPS)
#         replace_in_file2("topol.top", "BILAYERINFO", composition_txt)
#         replace_in_file2("topol.top", '#include "Protein.itp"', f'#include "{self.preptide_folder}/{self.peptide}.itp"')         ## update *itp name are required...
#         replace_in_file2("topol.top", "ROOT", self.root)         ## update the root dir name
#
#         if self.Rx:
#             nwaters = count_water("Box.gro")
#             replace_in_file2("topol.top", "WATER", self.water + "      " + str(nwaters-2000) + "\nWF       2000")
#         else:
#             replace_in_file2("topol.top", "WATER", self.water + "      " + str(count_water("Box.gro")))
#         if self.peptoid:
#             replace_in_file2("topol.top", "ANALYTE", f"Peptoid    {self.nmol}")
#         elif self.FF == "3":
#             replace_in_file2("topol.top", "ANALYTE", f"molecule_0    {self.nmol}")
#         else:
#             replace_in_file2("topol.top", "ANALYTE", f"{self.peptide}_A    {self.nmol}")
#
#         if self.Rx:
#             #replace_in_file2("topol.top", f"{self.peptide}_A    {self.nmol}", f"{self.peptide}_A    {self.nmol}\ndieneophile 80\ndiene 80\n")
#             replace_in_file2("topol.top", f"{self.peptide}_A    {self.nmol}", f"{self.peptide}_A    {self.nmol}\nDimethyl_maleate 80\n")
# =============================================================================

# =============================================================================
# =============================================================================
# #     Ross Fix; added ability to write ions to the topology, avoiding error
# #     with mismatching coordinates at the genion stage
# =============================================================================
# =============================================================================

    def WriteTopol(self):
        print("Writing topology to:", os.path.abspath("topol.top"))
        shutil.copy(f"{self.root}/topol{self.FF}.top", "topol.top")

        # Load BOTH the reference bilayer AND the current box
        U_bilayer = mda.Universe(self.bilayer_gro)
        U_box = mda.Universe("Box.gro")

        # Get lipid composition from bilayer
        PO4 = U_bilayer.select_atoms("name PO4 ROH PA4 GM1 AM1 GL1")

        print("Bilayer composition:")
        composition = OrderedDict()

        # Account for membrane proteins
        if self.bilayer in self.MembraneProteins:
            for protein in self.MembraneProteins[self.bilayer]:
                composition[protein] = self.MembraneProteins[self.bilayer][protein]

        n = 0
        self.PO4 = PO4
        used_resids = []

        # Count lipids
        for i, res in enumerate(PO4):
            if res.resid in used_resids:
                continue
            else:
                used_resids.append(res.resid)
            if i == 0:
                composition[f"{res.resname}-{n}"] = 1
            else:
                if list(composition.keys())[-1].split("-")[0] == res.resname:
                    composition[f"{res.resname}-{n}"] += 1
                else:
                    n += 1
                    composition[f"{res.resname}-{n}"] = 1

        print(composition)

        # Build composition text
        composition_txt = ""
        for resname in composition:
            num = composition[resname]
            res = resname.split("-")[0]
            composition_txt = composition_txt + f"{res}     {num}\n"

        # Include extra ITPs
        EXTRAITPS = ""
        for itp in self.bilayer_data["itps"]:
            if itp[:2] == "./":
                itp = os.path.join(self.bilayer_folder, itp)
            EXTRAITPS = EXTRAITPS + '#include "' + itp + '" \n'

        replace_in_file2("topol.top", "EXTRAITPS", EXTRAITPS)
        replace_in_file2("topol.top", "BILAYERINFO", composition_txt)
        replace_in_file2("topol.top", '#include "Protein.itp"', f'#include "{self.preptide_folder}/{self.peptide}.itp"')
        replace_in_file2("topol.top", "ROOT", self.root)

        # Count water in current box
        water_counts = count_water("Box.gro")
        n_na, n_cl = count_ions_in_box("Box.gro")

# =============================================================================
#         # *** FIX: Count ions from the current box ***
#         try:
#             NA_atoms = U_box.select_atoms("resname NA+ NA NAP")
#             CL_atoms = U_box.select_atoms("resname CL- CL CLM")
#             n_na = len(NA_atoms)
#             n_cl = len(CL_atoms)
#             print(f"Found {n_na} NA+ ions and {n_cl} CL- ions in Box.gro")
#         except:
#             n_na = 0
#             n_cl = 0
#             print("No ions found in Box.gro")
# =============================================================================

        # Build water + ions section
        water_ion_section = ""
        for water_type, count in water_counts.items():
            water_ion_section += f"{water_type}      {count}\n"
        if n_na > 0:
            water_ion_section += f"NA+      {n_na}\n"
        if n_cl > 0:
            water_ion_section += f"CL-      {n_cl}\n"

        replace_in_file2("topol.top", "WATER", water_ion_section.strip())

        # Add peptide
        if self.peptoid:
            replace_in_file2("topol.top", "ANALYTE", f"Peptoid    {self.nmol}")
        elif self.FF == "3":
            replace_in_file2("topol.top", "ANALYTE",
                             f"molecule_0    {self.nmol}")
        else:
            replace_in_file2("topol.top", "ANALYTE", f"{self.peptide}_A    {self.nmol}")

        if self.Rx:
            replace_in_file2("topol.top", f"{self.peptide}_A    {self.nmol}",
                             f"{self.peptide}_A    {self.nmol}\nDimethyl_maleate 80\n")

    def load_bilayer_metadata(self):
        json_file = os.path.join(f"{self.bilayer_folder}", "meta.json")
        with open(json_file, 'r') as jin:
            self.bilayer_data = json.load(jin)
            for k, v in self.bilayer_data.items():
                if "ROOT" in v:
                    self.bilayer_data[k] = v.replace("ROOT", self.root)
                elif k == 'itps':
                    for i, x in enumerate(v):
                        self.bilayer_data[k][i] = x.replace("ROOT", self.root)
        print(self.bilayer_data)


# =============================================================================
#     def genions(self):
#         #######
#         # print(self.preptide_folder)
#         cmd1 = f"gmx_mpi grompp -f {self.root}/min.mdp -p topol.top -c Box.gro -o Min -maxwarn 2"                   ## as below
#         cmd2 = "echo W | gmx_mpi genion -s Min.tpr -p topol.top -nname CL -pname NA -neutral -o Box.gro"            ## most reliable way to update ion content
#         os.system(cmd1)                                                                                             ## Na is now Nx
#         assert os.path.exists("Min.tpr"), print("Failed to to run genion")
#         os.system(cmd2)
#         #######
#         if self.FF != "3":
#             replace_in_file2("topol.top", "NA ","NA+ ")
#             replace_in_file2("topol.top", "CL ","CL- ")
# =============================================================================

    def genions(self):
        """Add ions to neutralize system and reach physiological salt concentration"""

        # First check if ions already present
        U = mda.Universe("Box.gro")
        existing_na = len(U.select_atoms("resname NA+ NA NAP"))
        existing_cl = len(U.select_atoms("resname CL- CL CLM"))

        if existing_na > 0 or existing_cl > 0:
            print(f"WARNING: Box.gro already contains {existing_na} NA+ and {existing_cl} CL- ions")
            print(
                "Skipping genion step - ions already in topology from reference bilayer")
            return

        # Standard genion procedure
        cmd1 = f"gmx_mpi grompp -f {self.root}/min.mdp -p topol.top -c Box.gro -o Min -maxwarn 2"
        os.system(cmd1)

        if not os.path.exists("Min.tpr"):
            print("Failed to create Min.tpr for genion")
            return False

        if self.water == 'W_WF':
            replace = "W"
        else:
            replace = self.water

        # Neutralize and add 0.15 M salt
        cmd2 = f"echo {replace} | gmx_mpi genion -s Min.tpr -p topol.top -nname CL -pname NA -neutral -conc 0.15 -o Box.gro"
        os.system(cmd2)

        # Fix ion naming for Martini 2.x
        if self.FF != "3":
            replace_in_file2("topol.top", "NA ", "NA+ ")
            replace_in_file2("topol.top", "CL ", "CL- ")

    def SetupBox(self):
        print(self.bilayer_data)
        os.makedirs(self.peptide_folder, exist_ok=True)
        os.chdir(self.peptide_folder)
        peptide_gro = f"{self.preptide_folder}/{self.peptide}.gro"
        if not os.path.exists(peptide_gro):
            self.Preptide()
            os.chdir(self.peptide_folder)
        print("Using:", peptide_gro)
        nBB = count_peptides(peptide_gro)
        fresh = True
        if os.path.exists("Box.gro"):
            if count_peptides("Box.gro") == nBB*self.nmol:
                print("Box with correct number of peptides already setup.")
                return 1

        MaxAttempts = 25
        for i in range(MaxAttempts):
            print("Attempt:", i)
            self.InsertionPositions()
            cmd = f"gmx_mpi insert-molecules -f {self.bilayer_gro} -ci {peptide_gro} -replace W -ip positions.dat -try 1000 -dr 5 5 2 -o Box.gro"
            os.system(cmd)
            print(self.peptide, self.nmol)
            # It may fail completley to even make a bad box if the preptide went wrong, remove preptide and try again
            if not os.path.exists("Box.gro"):
                os.remove(f"{self.root}/{self.peptide}/{self.peptide}.gro")
                os.remove(
                    f"{self.root}/{self.peptide}/{self.peptide}_solomin.gro")
                self.Preptide()
                self.SetupBox()
                return 1
            # This is adjusted to account for membrane proteins
            if count_peptides("Box.gro") - count_peptides(self.bilayer_gro) != nBB*self.nmol:
                print("Box.gro does not contain self.nmol * BB number of beads!!")
                print("count_peptides Box.gro =", count_peptides("Box.gro"))
                print("count_peptides self.bilayer_gro =",
                      count_peptides(self.bilayer_gro))
                print("nBB*self.nmol =", nBB*self.nmol)
                print("Trying again...")
                continue
            else:
                break
        if count_peptides("Box.gro") != nBB*self.nmol:
            print("\n"*5)
            print("Box.gro does not contain self.nmol * BB number of beads!!")

# =============================================================================
# =============================================================================
# #         As far as I can figure, self.Rx corresponds to reactive susbtrates
# #         and thus can be largely left alone here as it should default to False
# =============================================================================
# =============================================================================

        if self.Rx:
            cmd = f"gmx_mpi insert-molecules -f Box.gro -ci /users/rkb19187/Desktop/BioTech/DielsAlderCphMD/Dimethyl_maleate.gro -replace W -ip dieneophile_positions.dat -try 1000 -dr 5 5 2 -o Box.gro"
            os.system(cmd)
            # cmd = f"gmx_mpi insert-molecules -f Box.gro -ci /users/rkb19187/Desktop/BioTech/DielsAlderCphMD/Diene.gro -nmol 40 -replace W -ip diene_positions.dat -try 1000 -dr 5 5 2 -o Box.gro"
            # os.system(cmd)
            print("Added molecules to Box")

        # Setup topol
        self.WriteTopol()

        self.genions()

    def MinBox(self, one_shot=False):
        if parsemin(f"Min.log") > 200.0:
            pass
        else:
            print("Already minimized")
            return 1
        cmd = f"gmx_mpi grompp -f {self.root}/min.mdp -p topol.top -c Box.gro -o Min -maxwarn 2"
        os.system(cmd)
        assert os.path.exists("Min.tpr"), print("Failed to make Min.tpr")

        if self.peptoid:
            cmd = f"mpirun -np 1 gmx_mpi mdrun -v -ntomp {self.ntomp} -tableb {self.root}/angle5_a0.xvg -deffnm Min"
        else:
            cmd = f"mpirun -np 1 gmx_mpi mdrun -ntomp {self.ntomp} -deffnm Min"
        os.system(cmd)
        if not os.path.exists("Min.gro"):
            print("Failed to make Min.gro")
            sys.exit()

# =============================================================================
#     def SpecialIndex(self):
#         cmd = "echo q | gmx_mpi make_ndx -f Box.gro -o index.ndx"
#         os.system(cmd)
#         original   = readin("index.ndx")
#         u          = mda.Universe("Box.gro")
#         groProtein = u.select_atoms(f"resname {self.peptide_resnames}")       ## account for break in relation between resname and peptide name
#         groPOP     = u.select_atoms(f"resname {self.phospholipids}")   ## We do not want to include the outer membrane here
#         self.groPOP = groPOP
#
#         print(len(groProtein))
#         #cutoff = groPOP.positions[:,2].mean()
#         #print("Cutoff:", cutoff)
#         TopHalf = []
#         BottomHalf = []
#         phospholipids_indices = list()
#         for bead in groProtein:
#             TopHalf.append(bead.id)
# # =============================================================================
# #             if bead.position[2] >= cutoff:
# #                 TopHalf.append(bead.id)
# #             else:
# #                 BottomHalf.append(bead.id)
# #         for bead in groPOP:
# #             if self.Rx:
# #                 if abs(bead.position[2] - groPOP.positions[:,2].mean()) > 50: # We just want use the bilayer that appears in the middle of the box
# #                     continue
# # =============================================================================
#             phospholipids_indices.append(bead.id)
#         print("TopHalf:", len(TopHalf))
#         assert len(phospholipids_indices) > groPOP.positions.shape[0]*0.3
#
#         print("phospholipids_indices:", len(phospholipids_indices))
#         index = open("SpecialIndex.ndx", 'w')
#         index.write(original)
#         index.write("[ TopHalf ]\n")
#         i = 0
#         for ndx in TopHalf:
#             index.write(str(ndx)+" ")
#             i+=1
#             if i % 15==0:
#                 index.write("\n")
#         #a="""
#         index.write("\n[ Bilayer ]\n")
#         i = 0
#         for ndx in phospholipids_indices:
#             index.write(str(ndx)+" ")
#             i+=1
#             if i % 15==0:
#                 index.write("\n")
#         #"""
#         index.write("\n\n")
#         index.close()
# =============================================================================

    # Ross Fix for SpecialIndex error due to assrtion error

    def SpecialIndex(self):
        """Create custom index groups for pulling simulations"""

        # Create base index
        cmd = "echo q | gmx_mpi make_ndx -f Box.gro -o index.ndx"
        os.system(cmd)
        original = readin("index.ndx")

        # Load structure
        u = mda.Universe("Box.gro")

        # Select peptides (using peptide residue names)
        if self.peptoid:
            groProtein = u.select_atoms(f"resname {self.peptoid_resnames}")
        else:
            groProtein = u.select_atoms(f"resname {self.peptide_resnames}")

        # Auto-detect lipids using headgroup beads
        headgroup_beads = u.select_atoms("name PO4 ROH PA4 GM1 AM1 GL1")

        if len(headgroup_beads) == 0:
            # Fallback to hardcoded list
            print("No headgroup beads found, using hardcoded lipid list")
            groPOP = u.select_atoms(f"resname {self.phospholipids}")
        else:
            # Use detected lipids
            lipid_resnames = set(headgroup_beads.residues.resnames)
            print(f"Detected lipids: {', '.join(sorted(lipid_resnames))}")
            groPOP = u.select_atoms(f"resname {' '.join(lipid_resnames)}")

        self.groPOP = groPOP

        print(f"Peptide atoms: {len(groProtein)}")
        print(f"Lipid atoms: {len(groPOP)}")

        # Build TopHalf index (all peptide atoms)
        TopHalf = [bead.id for bead in groProtein]

        # Build Bilayer index (lipids, with optional filtering for Rx systems)
        phospholipids_indices = []

        if self.Rx:
            # Filter lipids: only include those near membrane center
            # (for reactive systems with multiple bilayers)
            membrane_center_z = groPOP.positions[:, 2].mean()
            print(f"Membrane center Z: {membrane_center_z:.1f} Å")

            for bead in groPOP:
                # Within 50 Å of center
                if abs(bead.position[2] - membrane_center_z) <= 50:
                    phospholipids_indices.append(bead.id)

            print(f"Filtered lipids (within 50 Å of center): {len(phospholipids_indices)}")
        else:
            # Include all lipids
            phospholipids_indices = [bead.id for bead in groPOP]

        # Validate selections
        if len(TopHalf) == 0:
            raise ValueError(
                "ERROR: No peptide atoms found for TopHalf group!")

        if len(phospholipids_indices) == 0:
            raise ValueError(
                "ERROR: No phospholipids found for Bilayer group!")

        # For Rx systems, validate we got enough lipids after filtering
        if self.Rx:
            min_expected = groPOP.positions.shape[0] * 0.3
            if len(phospholipids_indices) <= min_expected:
                print(f"WARNING: Only {len(phospholipids_indices)} lipids after filtering")
                print(f"Expected at least {min_expected:.0f} (30% of total)")

        # Write index file
        with open("SpecialIndex.ndx", 'w') as index:
            index.write(original)

            index.write("[ TopHalf ]\n")
            for i, ndx in enumerate(TopHalf):
                index.write(f"{ndx} ")
                if (i + 1) % 15 == 0:
                    index.write("\n")

            index.write("\n[ Bilayer ]\n")
            for i, ndx in enumerate(phospholipids_indices):
                index.write(f"{ndx} ")
                if (i + 1) % 15 == 0:
                    index.write("\n")

            index.write("\n\n")

        print(f"  Created SpecialIndex.ndx:")
        print(f"  [ TopHalf ]: {len(TopHalf)} atoms")
        print(f"  [ Bilayer ]: {len(phospholipids_indices)} atoms")

    def copy_mdp(self, src, dst):
        shutil.copy(src, dst)
        # if "POPC" not in self.bilayer:
        # replace_in_file2(dst, "POPC", self.composition[self.bilayer.upper()])
        if "Pull" in dst:
            nsteps = np.ceil((self.pull_time/self.dt)*1000).astype(np.int64)
        elif "Relax" in dst:
            nsteps = np.ceil((self.relax_time/self.dt)*1000).astype(np.int64)
        replace_in_file2(dst, "dt                       = 0.025",
                         f"dt                       = {self.dt}")
        replace_in_file2(dst, "nsteps                   = 4000000",
                         f"nsteps                   = {nsteps}")

    def SetupPulling(self):
        replace_in_file2("topol.top", "_HHHHHHHH", "")
        self.copy_mdp(self.pull_mdp, "Pull.mdp")
        if not os.path.exists("Pull.tpr"):
            cmd = f"gmx_mpi grompp -f Pull.mdp -p topol -c Min -n SpecialIndex.ndx -o Pull -maxwarn 2"
            os.system(cmd)
        else:
            print("Found: Pull.tpr")
            return 1

    def SetupRelax(self):
        replace_in_file2("topol.top", "_HHHHHHHH", "")
        self.copy_mdp(self.relax_mdp, "Relax.mdp")
        if not os.path.exists("Relax.tpr"):
            cmd = f"gmx_mpi grompp -f Relax.mdp -p topol -c PullMinCOM.gro -o Relax -maxwarn 1"
            os.system(cmd)
        else:
            print("Found: Relax.tpr")
            return 1

    def check_complete(self, Job):
        path = f"{Job}.log"
        if os.path.exists(path):
            content = readin(path)
            if "Finished mdrun" in content:
                return True
            else:
                return False
        else:
            return False

    def SubmitJob(self, Job, restart=False):
        if not os.path.exists(f"{Job}.tpr"):
            print(f"No {Job}.tpr job file!")
            return 0
        if self.check_complete(Job):
            print(f"Simulations.py > SubmitJob > check_complete: Job complete!")
            return 0
        if os.path.exists("Pull_pullx.xvg") and Job.upper() == "PULL":
            print("Found: Pull_pullx.xvg, Submitting a restart.")
            restart = True

        shame = f"{Job}.sh" if not restart else f"{Job}_restart.sh"
        if self.GPU:
            shutil.copy(self.root+"/gpu.sh", shame)
        else:
            shutil.copy(self.root+"/cpu.sh", shame)
        replace_in_file2(shame, "PEPTIDE_bilayer", self.job_name(Job))

        if restart:
            if Job.upper() == "PULL":
                replace_in_file2(shame, "-deffnm Pull",
                                 f"-deffnm {Job} -cpi {Job} -noappend")
            else:
                replace_in_file2(shame, "-deffnm Pull",
                                 f"-deffnm {Job} -cpi {Job} -append")
        else:
            replace_in_file2(shame, "-deffnm Pull", f"-deffnm {Job}")

        if not self.peptoid:
            replace_in_file2(shame, "-tableb ROOT/angle5_a0.xvg", "")

        replace_in_file2(shame, "PEPTIDE", self.peptide)
        replace_in_file2(shame, "NMOL", str(self.nmol))
        replace_in_file2(shame, "ROOT", self.root)
        replace_in_file2(shame, "CONDA_ENV", os.environ["CONDA_ENV"])

        # Auto do relax after pulling
        if "PULL" in Job.upper():
            if not self.GPU:
                replace_in_file2(shame, "--gpu", "")
            if not self.run:
                replace_in_file2(shame, "--reproduce re", "")
        # elif "RELAX" in Job.upper():
        #     replace_in_file2(shame, "/opt/software/scripts/job_epilogue.sh ", f"""/opt/software/scripts/job_epilogue.sh""")

        replace_in_file2(shame, "CPUS", str(self.ntmpi))
        cmd = "sbatch "+shame
        print(cmd, "for", self.peptide)
        os.system(cmd)

    def job_name(self, typ):
        return f"{self.peptide}_{self.nmol}-{typ}-{self.bilayer}-{self.water}_run{self.run}-FF{self.FF}"

    def CheckQ(self):
        for Job in ["Pull", "Relax"]:
            JobName = self.job_name(typ=Job)
            if self.hpc.IsRunning(JobName):
                print(f"{JobName} already in queue")
                return True
        return False

    def CheckCpHMDQ(self, peptide, pH=7):
        job = f"{peptide}pH{pH}"
        if self.hpc.IsRunning(job, exact=False):
            print(f"{job} already in queue")
            return True
        return False
    

    def _XtractFrame(self, gro, xtc, frame, fname):
        U = mda.Universe(gro, xtc)
        U.trajectory[frame]  # Switches active frame
        atoms = U.select_atoms("all")
        atoms.write(fname)

    def CheckFolder(self):
        pull_file = os.path.join(self.root, f"{self.boxes}/{self.bilayer}/run_{self.run}/{self.peptide}_{self.nmol}/Pull.log")
        relax_file = os.path.join(self.root, f"{self.boxes}/{self.bilayer}/run_{self.run}/{self.peptide}_{self.nmol}/Relax.log")
        if not os.path.exists(pull_file):
            print("Couldnt find:", pull_file)
            return "Pull"
        pull_log = readin(pull_file)\

        if "Fatal error:" in pull_log:
            return "Pull-failed"
        ######################################################
        if "           Step" in pull_log:
            last_index = None
            data = pull_log.split('\n')
            for index, line in enumerate(data):
                if "           Step" in line:
                    if index + 1 < len(data):
                        values = (data[index + 1]).split()
                        nums = [float(val) for val in values]
            ######################################################
            last_time = nums[-1]
            if (float(last_time)/1000) < self.PullTime:
                return "Pull-restart"
        else:
            return "Pull-failed"

        if not os.path.exists(relax_file):
            return "Relax"
        relax_log = readin(relax_file)
        self.relax_file = relax_file
        self.relax_log = relax_log
        if "Fatal error:" in relax_log:
            return "Relax-failed"
        last_time = relax_log.split(
            "Step           Time\n")[-1].split("\n")[0].strip().split()[1]
        print("Last time in Relax.log:", (float(last_time)/1000), "ns")
        if (float(last_time)/1000) < self.relax_time:
            return "Relax-restart"

        return "CpHMD"

    def pull_in_targets(self):
        return self.peptide in self.targets.index

    def relax_in_targets(self):
        return self.peptide in self.targets.dropna().index

    def ParseXVG(self, fname):
        xvg = pandas.read_csv(fname, index_col=0,
                              header=None, skiprows=17, sep="\t")
        content = readin(fname)
        title = content.split("@    title")[1].split("\n")[0].split("\"")[1]
        xlabel = content.split("@    xaxis")[1].split("\n")[0].split("\"")[1]
        ylabel = content.split("@    yaxis")[1].split("\n")[0].split("\"")[1]
        return title, xlabel, ylabel, xvg

    def getFx(self, folder):
        # print(folder)
        Pullx = pandas.DataFrame(columns=["Position (nm)"])
        Pullf = pandas.DataFrame(columns=["Force (kJ/mol/nm)"])
        for i, xvg_file in enumerate(glob.glob(folder+"/*pullx*.xvg")):
            title, xlabel, ylabel, xvg = self.ParseXVG(xvg_file)
            xvg.columns = [ylabel]
            Pullx = pandas.concat((Pullx, xvg))
            Pullx.columns = [ylabel]
        Pullx = Pullx.sort_index().dropna()
        for i, xvg_file in enumerate(glob.glob(folder+"/*pullf*.xvg")):
            title, xlabel, ylabel, xvg = self.ParseXVG(xvg_file)
            xvg.columns = [ylabel]
            Pullf = pandas.concat((Pullf, xvg))
            Pullf.columns = [ylabel]
        Pullf = Pullf.sort_index().dropna()
        return Pullf, Pullx

    def StatusHandler(self, Status):
        if "Pull" in Status:
            if Status == "Pull-failed" or Status == "Pull-restart":
                print("Pulling failed with terminal error!")
                print("Deleting and starting again.")
                print("Disabling GPU for MD since sometimes this is causing it to fail.")
                self.GPU = False
                Status = "Pull"
            if Status == "Pull":
                self.Preptide()
                self.reset_box()  # If we are at status = Pull and stuff exists in this folder somethings gone badly wrong
                self.SetupBox()
                self.MinBox()
                self.SpecialIndex()
                self.SetupPulling()
                self.SubmitJob("Pull")
            else:
                print("Unknown status:", status)
        elif "Relax" in Status:
            os.chdir(self.peptide_folder)
            if Status == "Relax":
                # returns empty if file not found
                Pullf, Pullx = self.getFx(".")
                if os.path.exists("Pull_APL.csv"):
                    pprint("Found: Pull_APL.csv")
                    APL = pandas.read_csv(f"Pull_APL.csv", index_col=0)
                    # print(APL)
                else:
                    APL = getAPL(f"Pull.gro", f"Pull.xtc", L=len(self.peptide))
                    APL.to_csv("Pull_APL.csv")
                    assert os.path.exists("Pull_APL.csv")

                frame = np.where(APL["APL"].values == APL["APL"].max())[0][-1]
                self._XtractFrame("Pull.gro", "Pull.xtc",
                                  frame, "PullMinCOM.gro")
                self.SetupRelax()
                self.SubmitJob("Relax")
            elif Status == "Relax-restart":
                print("Restarting pulling")
                print("Disabling GPU for MD since sometimes this is causing it to fail.")
                self.GPU = False
                self.SubmitJob("Relax", restart=False)
            elif Status == "Relax-failed":
                print("Restarting pulling")
                print("Disabling GPU for MD since sometimes this is causing it to fail.")
                self.GPU = False
                self.SubmitJob("Relax", restart=False)
            else:
                print("Unknown status:", Status)
                sys.exit()
        elif Status == "CpHMD":
            os.chdir(self.peptide_folder)

            if not os.path.exists("Relax_APL.csv"):
                APL = getAPL("Relax.gro", "Relax.xtc", L=len(self.peptide))
                APL.to_csv("Relax_APL.csv")
                assert os.path.exists("Relax_APL.csv")

            print("Status is CpHMD-ready.")
            print("Pull and Relax are complete.")
            print("No CpHMD preparation or submission is handled by Simulations.py.")
    
    def __init__(self, sequence, nmol, GPU=False, run=0, cpus=16, peptoid=False, FF="2.1", W="W_WF", pull_time=100, relax_time=100, bilayer="POPC80POPS20", Rx=False):
        # list of non-protein membrane constituents
        self.phospholipids = "POPC POPS DPPC DTPC DPPE POPE DPPS POPI DPPI CHOL DPSM DOPE DOPG CDL1 DOPG PIPC DPCE PIPX POPX PAPC DPSM PNSM PAPS"
        self.OuterMembrane = "ECLIP AKDO AHEP AGLC AGAL BGLC BGLCN BMAN AGALN AGLCN"
# =============================================================================
#         self.composition = dict()
#         for PC in self.phospholipids:
#             self.composition[PC] = PC
#         self.composition["SARS-COV-2"] = "POPC POPS DPPC DTPC DPPE POPE DPPS POPI DPPI CHOL DPSM"
#         self.composition["SARS-Cov-2_Mprotein"] = "POPC POPS DPPC DTPC DPPE POPE DPPS POPI DPPI CHOL DPSM"
#         self.composition["OuterMembrane"] = "DOPE CDL1 DOPG ECLIP AKDO AHEP AGLC AGAL BGLC BGLCN BMAN AGALN"
# =============================================================================

        self.MembraneProteins = dict()
        self.MembraneProteins["SARS-Cov-2_Mprotein"] = {
            "Membrane_M_Protein": 2}
        self.MembraneProteins["OuterMembrane"] = {"LPSR1O62": 84}

        self.username = os.environ["USERNAME"]
        # os.path.join(os.environ["CMLROOT"], "MolecularDynamics")
        self.root = os.environ['CMLROOT']

        ####
        # debugging helpers
        print("\n--- ACTIVE SCRIPT/ENVIRONMENT CHECK ---")
        print("Simulations.py file =", os.path.abspath(__file__))
        print("cwd                 =", os.getcwd())
        print("CMLROOT/self.root    =", self.root)
        print("structure_gen.tcl    =", os.path.join(
            self.root, "structure_gen.tcl"))
        print("martinize.py         =", os.path.join(self.root, "martinize.py"))
        print("topol2.1.top         =", os.path.join(self.root, "topol2.1.top"))
        print("_CBox module file    =", sys.modules.get(
            "_CBox").__file__ if "_CBox" in sys.modules else "package import")
        print("Martinoid file       =", Martinoid.__file__)
        print("hpctools file        =", hpctools.__file__)
        print("CONDA_ENV            =", os.environ.get("CONDA_ENV"))
        print("GMXDATA              =", os.environ.get("GMXDATA"))
        print("PATH                 =", os.environ.get("PATH"))
        print("--------------------------------------\n")
        ####

        self.peptide = sequence
        self.vmd = "vmd"
        self.ntmpi = cpus
        self.nmol = nmol
        self.peptoid = peptoid
        self.FF = str(FF)
        self.water = W.upper()
        if self.water == "TW":
            self.dt = 0.020
        else:
            self.dt = 0.025
        self.hpc = hpctools.hpctools()
        self.bilayer = bilayer
        self.PullTime = 100.0  # ns
        self.pull_time = pull_time
        self.relax_time = relax_time
        self.COM_cutoff = 2.0
        self.APL_cutoff = 72
        self.GPU = GPU
        self.run = run
        self.Rx = Rx

        self.load_bilayer_metadata()
        self.internal_checks()


if __name__ == "__main__":
    # this list is the top 5 and bottom 5 of the PFP and NON-PFP Alex peptide's (https://doi.org/10.1039/D4CP01404A) as determined by APL during the CpHMD stage - there were 5 more best and 5 more worst added from the APL at stage
    # Top 5 &&& Bottom 5 PFP (CpHMD)       = [FGYVLIRI, IGFIRVLY, MFIFFSRF, YCVLRLPF, LSFMRFFF &&& FFMSIRFF, QWCIKSKG, FSVLFFRW, YPKMFAFF, IKKFYYYY]
    # Top 5 &&& Bottom 5 PFP (MD)          = [LFVHKVLT, LVIKVFTH, IKKFYYYY, FFPYGWKG, QFLMKCLL &&& FFMSIRFF, QWCIKSKG, GMFFWGKY, LYGGCKKC, HGLFWWRF]
    # Top 5 &&& Bottom 5 NON-PFP (CpHMD)   = [FGCFFWRW, SMTFWWRI, QHPCWRTA, MIMFYMRY, RVHIRLWI &&& CKKFYYYY, GKKFWWIM, QHPSYYYY, WWGSSKMK, LTKFFFMM]
    # Top 12 PFP (CpHMD)                   = ["FFFLSRIF", "GSGTGSGT", "CFTYFFRV", "VCVYWWRT", "FGYVLIRI", "LSFMRFFF", "SAFWWFRF", "HGLFWWRF", "GIALKIVW", "YCVLRLPF"]

    # From the above sequences, some were separated because some terminal or internal residues are not yet parameterised for CpHMD charge-swapping values, and are in separate lists depending on rejection reason
    #peptorun = ["AKRIRKLIKKIFKKI", "FLPLLASLFSRLF", "FGYVLIRI", "IGFIRVLY", "MFIFFSRF", "LSFMRFFF", "FFMSIRFF", "QWCIKSKG", "FSVLFFRW", "YPKMFAFF", "LFVHKVLT","LVIKVFTH", "FFPYGWKG", "QFLMKCLL","GMFFWGKY", "HGLFWWRF", "FGCFFWRW", "SMTFWWRI", "MIMFYMRY", "RVHIRLWI", "GKKFWWIM", "LTKFFFMM", "FFFLSRIF", "GSGTGSGT", "SAFWWFRF", "GIALKIVW"] #accepted
    
    # rejected_terminus = ["IKKFYYYY", "CKKFYYYY"]
    # rejected_cysteine_or_tyrosine_middle = ["YCVLRLPF", "LYGGCKKC", "QHPCWRTA", "QHPSYYYY", "CFTYFFRV", "VCVYWWRT"]
    # NONE were rejected due to both terminus and middle


    # peptorun = ["FGYVLIRI", "IGFIRVLY", "MFIFFSRF", "YCVLRLPF", "LSFMRFFF", "FFMSIRFF", "QWCIKSKG", "FSVLFFRW", "YPKMFAFF", "IKKFYYYY", "LFVHKVLT", "LVIKVFTH", "FFPYGWKG", "QFLMKCLL", "GMFFWGKY", "LYGGCKKC", "HGLFWWRF", "FGCFFWRW", "SMTFWWRI", "QHPCWRTA", "MIMFYMRY", "RVHIRLWI", "CKKFYYYY", "GKKFWWIM", "QHPSYYYY", "WWGSSKMK", "LTKFFFMM", "FFFLSRIF", "GSGTGSGT", "CFTYFFRV", "VCVYWWRT", "SAFWWFRF", "GIALKIVW", "RRW", "RRWRRW", "RRWRRWRRW", "RRWRRWRRWRRW"] #peptide version of Alex's PPF paper
    # peptorun = ['Nm-Nf-Ni-Nf-Nf-Ns-Nr-Nf', 'Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf', 'Nf-Nf-Nm-Ns-Ni-Nr-Nf-Nf', 'Nf-Ns-Nv-Nl-Nf-Nf-Nr-Nw', 'Ni-Nk-Nk-Nf-Ny-Ny-Ny-Ny', 'Ns-Nm-Nt-Nf-Nw-Nw-Nr-Ni', 'Nm-Ni-Nm-Nf-Ny-Nm-Nr-Ny', 'Nl-Nt-Nk-Nf-Nf-Nf-Nm-Nm', 'Nf-Nf-Nf-Nl-Ns-Nr-Ni-Nf', 'Ns-Na-Nf-Nw-Nw-Nf-Nr-Nf', 'Nr-Nr-Nw', 'Nr-Nr-Nw-Nr-Nr-Nw', 'Nr-Nr-Nw-Nr-Nr-Nw-Nr-Nr-Nw', 'Nr-Nr-Nw-Nr-Nr-Nw-Nr-Nr-Nw-Nr-Nr-Nw']

    # the following peptorun are from the excel sheet of Hamish's tested - this is to compare my results to his
    # peptorun = ["Nfex-Nfnap-Nv", "Nab-Nt-Ni", "Ns-NmO-Nab", "Na-Na-Na", "", "", "", "Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex"] #3 are handpicked top & 3 are middle & 3 are bottom performers. Finally there were some additions from the synthesis side of things:
    # peptorun = ["Nfex-Nfnap-Nv", "Nab-Nt-Ni", "Ns-NmO-Nab", "Na-Na-Na"]

    # this is the first tests of why the peptoids are not running to completion
    # peptorun = ["Nl-Nt-Nk-Nf-Nf-Nf-Nm-Nm"]


    # peptorun = ["YYACAYY", "MFIFFSRF"]  # , "FGYVLIRI", "IGFIRVLY"]
    # peptorun = ["AKRIRKLIKKIFKKI"]

    
    # peptorun = ["Nl-Nt-Nk-Nf-Nf-Nf-Nm-Nm"]
    # peptorun = ["Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nfex-Nfnap-Nv"]
    # peptorun = ["Nke-NfeB4-Nwe", "Nwe-Nke-Nfex", "Nwe-Nke-Nwe", "Nfex-Nfex-Nfex", "Nk-Nfex-Nfex", "Nfex-Nfnap-Nv", "Nab-Nt-Ni", "Ns-NmO-Nab", "Nr-Nfex-Nr", "Nk-Nn-Ns"]
    # peptorun = ["Nl-Nt-Nk-Nf-Nf-Nf-Nm-Nm", 'Nm-Nf-Ni-Nf-Nf-Ns-Nr-Nf', 'Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf', 'Nf-Nf-Nm-Ns-Ni-Nr-Nf-Nf', 'Nf-Ns-Nv-Nl-Nf-Nf-Nr-Nw', 'Ni-Nk-Nk-Nf-Ny-Ny-Ny-Ny', 'Nm-Ni-Nm-Nf-Ny-Nm-Nr-Ny', 'Ns-Na-Nf-Nw-Nw-Nf-Nr-Nf', 'Nf-Nf-Nf-Nl-Ns-Nr-Ni-Nf', 'Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf', 'Ns-Na-Nf-Nw-Nw-Nf-Nr-Nf'] #This is the only peptoids that can be made from the top 5 lists (Alex's peptides I think)

    #######
    # "Na-Na-Na" bugged during pull index selection

    # ---
    # Note to self:
    # pep - True APL - Mordred_prediction - iteration (x out of 8) - overall position (x out of 81)
    # --- TOP
    # "Nfex-Nfnap-Nv" - 78.87332 - 74.27545898 - 3 / 8 - 8 / 81
    # "Nab-Nt-Ni" - 76.3912 - 75.10364407 - 4/8 - 20 / 81
    # "Ns-NmO-Nab" - 73.21905 - 75.00534239 - 4 / 8 - 33 / 81

    # --- MIDDLE
    # "" -  -  -  / 8 -  / 81
    # "" -  -  -  / 8 -  / 81
    # "" -  -  -  / 8 -  / 81

    # --- BOTTOM
    # "Nr-Nfex-Nr" - 67.78938 - 75.08679369 - 6 / 8 - 78 / 81
    # "Na-Na-Na" - 65 - N/A (everything starts from somewhere) - 0 / 8 - 80 / 81
    # "Nk-Nn-Ns" - 62.67451 - 63.18287773 - 1 / 8 - 81 / 81

    # --- Synthesis
    # "" -  -  -  / 8 -  / 81
    # "" -  -  -  / 8 -  / 81
    # "" -  -  -  / 8 -  / 81

    # peptorun = ['Nf-Ng-Ny-Nv-Nl-Ni-Nr-Ni', 'Ni-Ng-Nf-Ni-Nr-Nv-Nl-Ny', 'Nm-Nf-Ni-Nf-Nf-Ns-Nr-Nf', 'Ny-Nc-Nv-Nl-Nr-Nl-Np-Nf', 'Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf', 'Nf-Nf-Nm-Ns-Ni-Nr-Nf-Nf', 'Nq-Nw-Nc-Ni-Nk-Ns-Nk-Ng', 'Nf-Ns-Nv-Nl-Nf-Nf-Nr-Nw', 'Ny-Np-Nk-Nm-Nf-Na-Nf-Nf', 'Ni-Nk-Nk-Nf-Ny-Ny-Ny-Ny', 'Nl-Nf-Nv-Nh-Nk-Nv-Nl-Nt', 'Nl-Nv-Ni-Nk-Nv-Nf-Nt-Nh', 'Nf-Nf-Np-Ny-Ng-Nw-Nk-Ng', 'Nq-Nf-Nl-Nm-Nk-Nc-Nl-Nl', 'Ng-Nm-Nf-Nf-Nw-Ng-Nk-Ny', 'Nl-Ny-Ng-Ng-Nc-Nk-Nk-Nc', 'Nh-Ng-Nl-Nf-Nw-Nw-Nr-Nf', 'Nf-Ng-Nc-Nf-Nf-Nw-Nr-Nw', 'Ns-Nm-Nt-Nf-Nw-Nw-Nr-Ni', 'Nq-Nh-Np-Nc-Nw-Nr-Nt-Na', 'Nm-Ni-Nm-Nf-Ny-Nm-Nr-Ny', 'Nr-Nv-Nh-Ni-Nr-Nl-Nw-Ni', 'Nc-Nk-Nk-Nf-Ny-Ny-Ny-Ny', 'Ng-Nk-Nk-Nf-Nw-Nw-Ni-Nm', 'Nq-Nh-Np-Ns-Ny-Ny-Ny-Ny', 'Nw-Nw-Ng-Ns-Ns-Nk-Nm-Nk', 'Nl-Nt-Nk-Nf-Nf-Nf-Nm-Nm', 'Ng-Ns-Ng-Nt-Ng-Ns-Ng-Nt', 'Nc-Nf-Nt-Ny-Nf-Nf-Nr-Nv', 'Nv-Nc-Nv-Ny-Nw-Nw-Nr-Nt', 'Ns-Na-Nf-Nw-Nw-Nf-Nr-Nf', 'Ng-Ni-Na-Nl-Nk-Ni-Nv-Nw', 'Nf-Nf-Nf-Nl-Ns-Nr-Ni-Nf', 'Ng-Ns-Ng-Nt-Ng-Ns-Ng-Nt', 'Nc-Nf-Nt-Ny-Nf-Nf-Nr-Nv', 'Nv-Nc-Nv-Ny-Nw-Nw-Nr-Nt', 'Nf-Ng-Ny-Nv-Nl-Ni-Nr-Ni', 'Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf', 'Ns-Na-Nf-Nw-Nw-Nf-Nr-Nf', 'Nh-Ng-Nl-Nf-Nw-Nw-Nr-Nf', 'Ng-Ni-Na-Nl-Nk-Ni-Nv-Nw', 'Ny-Nc-Nv-Nl-Nr-Nl-Np-Nf', "Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nr-Nw-Nw", "Nr-Nw-Nw-Nr-Nw-Nw", "Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw", "Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw"] #peptoid version of Alex's PPF paper not all of the peptoids in this list are possible, below are the only ones:
    #peptorun = ['Nm-Nf-Ni-Nf-Nf-Ns-Nr-Nf', 'Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf', 'Nf-Nf-Nm-Ns-Ni-Nr-Nf-Nf', 'Nf-Ns-Nv-Nl-Nf-Nf-Nr-Nw', 'Ni-Nk-Nk-Nf-Ny-Ny-Ny-Ny', 'Nm-Ni-Nm-Nf-Ny-Nm-Nr-Ny', 'Nl-Nt-Nk-Nf-Nf-Nf-Nm-Nm', 'Ns-Na-Nf-Nw-Nw-Nf-Nr-Nf', 'Nf-Nf-Nf-Nl-Ns-Nr-Ni-Nf', 'Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf', 'Ns-Na-Nf-Nw-Nw-Nf-Nr-Nf', "Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex"] #This is the only peptoids that can be made from the top 5 lists + the Nk-Nfex-Nfex moitey
    # problematic runs = ['Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf', 'Ns-Nm-Nt-Nf-Nw-Nw-Nr-Ni', ] #double check that these ones completed - some reruns have been tried
    # peptorun = ["Nm-Nf-Ni-Nf-Nf-Ns-Nr-Nf", "Nf-Nf-Nm-Ns-Ni-Nr-Nf-Nf", "Nf-Ns-Nv-Nl-Nf-Nf-Nr-Nw", "Ni-Nk-Nk-Nf-Ny-Ny-Ny-Ny", "Nm-Ni-Nm-Nf-Ny-Nm-Nr-Ny", "Nl-Nt-Nk-Nf-Nf-Nf-Nm-Nm", "Nf-Nf-Nf-Nl-Ns-Nr-Ni-Nf"]

    # peptorun = ["Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex"] #, "Nr-Nw-Nw", "Nr-Nw-Nw-Nr-Nw-Nw", "Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw", "Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw"] #this is the RWW and KFF trimer motifs at n = 1, 2, 3, 4 (3-12 monomers)
    # peptorun = ["AKRIRKLIKKIFKKI", "FLPLLASLFSRLF", "RRWKWRRKIKKWL", "RRWRRWRRWRRW", "RRWRRWRRW", "RRWRRW", "RRW", "RFR", "LTKFFFMM"]
    # peptorun = ["Nfex-Na-Nfex-Nfex-Na-Nfex", "Nk-Na-Nfex-Nk-Na-Nfex"]

    # peptorun = ["Nw-Nfe-Nk-Nn-Ne-Na-Nfnap-Ny"]
    # peptorun = ["Na-Nfe-Nk-Nn-Ne-Na-Nfnap-Ny", "Nw-Nfe-Nk-Nn-Ne-Na-Nfnap-Na"]
    # peptorun = ["Nw-Nfe-Nk-Nn-Ne-Na-Nfnap-Ny", "Na-Nfe-Nk-Nn-Ne-Na-Nfnap-Ny", "Nw-Nfe-Nk-Nn-Ne-Na-Nfnap-Na"]

# =============================================================================
#     #this is the peptoid runs that just need to complete the Relax Stage
#     peptorun = [
#     "Nf-Nf-Nm-Ns-Ni-Nr-Nf-Nf",
#     "Nf-Ns-Nv-Nl-Nf-Nf-Nr-Nw",
#     "Ni-Nk-Nk-Nf-Ny-Ny-Ny-Ny",
#     "Nk-Nfex-Nfex-Nk-Nfex-Nfex",
#     "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex",
#     "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex",
#     "Nm-Nf-Ni-Nf-Nf-Ns-Nr-Nf",
#     "Nm-Ni-Nm-Nf-Ny-Nm-Nr-Ny",
#     "Ns-Na-Nf-Nw-Nw-Nf-Nr-Nf"
# ]
# =============================================================================

    # peptorun = ["Nab-Nt-Ni", "Nk-Nn-Ns", "Ns-NmO-Nab"] #3-mers needing ran
    # peptorun = ["Nf-Nf-Nf-Nl-Ns-Nr-Ni-Nf"] #8-mer needs finished
    # peptorun = ["Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf"] #strange failure


# =============================================================================
# #this is the peptide runs that just need to complete the Relax Stage
#     peptorun = [
#     "FLPLLASLFSRLF",
#     "IGFIRVLY",
#     "LTKFFFMM",
#     "RFR",
#     "RRW",
#     "RRWKWRRKIKKWL",
#     "RRWRRW",
#     "RRWRRWRRW",
#     "RRWRRWRRWRRW"
# ]
# =============================================================================
    
    #peptorun = ["FLPLLASLFSRLF", "GKKFWWIM", "LTKFFFMM", "GSGTGSGT", "GIALKIVW", "FFFLSRIF", "SAFWWFRF"]
   
    # peptorun = ["Nr-Nr-Nw-Nk-Nw-Nr-Nr-Nk-Ni-Nk-Nk-Nw-Nl", "Nf-Nl-Np-Nl-Nl-Na-Ns-Nl-Nf-Ns-Nr-Nl-Nf", "Nr-Nw-Nw-Nr-Nw-Nw", "Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw", "Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw-Nr-Nw-Nw"] #-> failure at the soloeq stage &"Nf-Nl-Np-Nl-Nl-Na-Ns-Nl-Nf-Ns-Nr-Nl-Nf | ["Nf-Nl-Np-Nl-Nl-Na-Ns-Nl-Nf-Ns-Nr-Nl-Nf"]-> failure as there are no Np peptoid monomers

    # failed to run "Na-Nk-Nr-Ni-Nr-Nk-Nl-Ni-Nk-Nk-Ni-Nf-Nk-Nk-Ni"
    
    
    #peptorun = ["Nk-Nfex-Nfex"] #linear peptoid (short chained) run seperately 
    #peptorun = ["Nl-Ns-Nf-Nm-Nr-Nf-Nf-Nf", "Nl-Nt-Nk-Nf-Nf-Nf-Nm-Nm", "Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex", "Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex"]

    # peptorun =["Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex-Nk-Nfex-Nfex"]



    # =============================================================================
    # PEPTIDE RUN LISTS
    # =============================================================================
    # A peptoid may appear in multiple strategy lists: this is intentional.
    # It must still only be physically simulated once.
    #
    # Historical/stale/unknown lists are kept separately and must NOT be submitted.
    # Known physical duplicates are listed explicitly below.
    # =============================================================================
    

    # # Model 0: original completed dataset
    # model0_original_25 = ["Nke-NfeB4-Nwe", "Nfex-Nfnap-Nv", "Nfnap-NfeC4-NfeC4", "NmO-Nm-Nv", "Nfnap-Nfe-Nm", "NfeC4-Nf-NmO", "Nv-Nm-NmO", "Nfe-Nfnap-Nfe", "Na-Nab-Ni", "Nm-NfeC4-Ni", "Nf-Na-Nfex", "Nf-Nfe-Nfnap", "Nv-Nv-Nl", "Nfex-Ni-Nfnap", "Nfex-Nfex-Nfex", "Nab-Nt-Ni", "Nr-Nfex-Nr", "Nr-Nr-Nw", "Ns-NmO-Nab", "Nwe-Nke-Nfex", "Nwe-Nke-Nwe", "Nw-NfeC4-NfeC4", "Nk-Nfex-Nfex", "Nwe-NfeB4-Nke", "Nfe-Nfe-Nfe"]
    
    # # Cycle 1
    # peptorun_1A = ["Nfe-Nfex-Nk", "Nfe-Nv-Nwe", "Nfex-Nfe-Nk", "Nfex-Nfex-Nk", "Nfex-Nv-Nwe", "Nk-Nfex-Nfe", "Nk-Nl-Nfnap", "Nwe-Nfn-Nl", "Nwe-Ni-Nf", "Na-Nfex-Nfnap", "Nfex-Nfn-Nk", "Nfn-Nfex-Nk", "Ni-Nfex-Nfnap", "Ni-Nfn-Nfnap", "Nk-Nfex-Nf"]
    
    # peptorun_1B = ["Nf-Nfe-Nwe", "Nd-NfeB4-Nwe", "Na-Na-Nd", "Na-Na-Na", "Nf-Nfe-Nm", "Na-Nw-NfeC4", "Na-Nf-Nwe", "Na-Nfe-Nwe", "Na-Nfnap-Nfnap", "Na-NfeB4-NfeB4", "Na-NfeC4-Nk", "Nk-Ni-Nfex", "Nfe-Nfe-Nv", "Na-NfeC4-Nwe", "Na-Nfex-Nfnap"]
    
    # peptorun_1C = ["Nfe-Nfex-Nk", "Nfe-Nv-Nwe", "Nfex-Nfe-Nk", "Nfex-Nfex-Nk", "Nfex-Nv-Nwe", "Na-Nfe-Nwe", "Nf-Nfex-Nk", "Nf-Ni-Nwe", "Nf-Nl-Nwe", "Nf-Nw-Nl", "Nd-Nfe-Nwe", "Nd-Nw-Nfnap", "Nd-Nw-Nwe", "Nd-Nwe-Nfe", "Nd-Nwe-Nwe"]
    
    # peptorun_1D = ["Nfe-Nfex-Nk", "Nfe-Nv-Nwe", "Nfex-Nfe-Nk", "Nfex-Nfex-Nk", "Nfex-Nv-Nwe", "Nk-Nfex-Nfe", "Nk-Nl-Nfnap", "Nwe-Nfn-Nl", "Nwe-Ni-Nf", "Na-Nfex-Nfnap", "Nd-Nfe-Nwe", "Nd-Nw-Nfnap", "Nd-Nw-Nwe", "Nd-Nwe-Nfe", "Nd-Nwe-Nwe"]
    
    # peptorun_1E = ["Nfe-Nfex-Nk", "Nfe-Nv-Nwe", "Nfex-Nfe-Nk", "Nfex-Nfex-Nk", "Nfex-Nv-Nwe", "Nk-Nfex-Nfe", "Nk-Nl-Nfnap", "Na-Nfe-Nwe", "Nf-Nfex-Nk", "Nf-Ni-Nwe", "Nf-Nl-Nwe", "Nd-Nfe-Nwe", "Nd-Nw-Nfnap", "Nd-Nw-Nwe", "Nd-Nwe-Nfe"]
    
    # # Cycle 2
    # peptorun_2A = ["Nk-Nfnap-Nl", "Nk-Nfnap-Ni", "Ni-Nfnap-Nk", "Nfex-Nfnap-Nk", "Nl-Nfnap-Nk", "Nv-Nfnap-Nk", "Nk-Nfnap-Nfe", "Nk-Nw-Nl", "Nk-Nv-Nwe", "Nk-Nfex-Nfnap", "Nfe-Nfnap-Nl", "Nl-Nfnap-Nfe", "Ni-Nfnap-Nl", "Nl-Nv-Nwe", "Nfe-Nw-Nl"]
    
    # peptorun_2B = []  # Old 2B predictions are stale/quarantined.
    
    # peptorun_2D = ["Nn-Nw-Nwe", "Nn-Nwe-Nwe", "Nn-NfeB4-Nwe", "Nwe-Nw-Na", "Nn-Nfnap-Nwe", "Nk-Nw-Nwe", "Nk-Nk-Nwe", "Nk-Nfnap-Nk", "Nwe-Nwe-Na", "Nk-Nw-Nk", "Nse-Na-Nfn", "Nse-Nfn-Nfn", "Nse-Nm-Nfn", "Ne-Nm-Nfn", "Nm-Nse-Nfn"]
    
    # # Historical / noncanonical
    # historical_1B = ["Nf-Nf-Nkeqm"]
    
    # historical_1X_1Y = ["Ne-Nf-Nwe", "Ne-Nfe-Nfnap", "Ne-Nfe-Nwe", "Ne-Nfex-Nfnap", "Ne-Nfnap-Nfe"]
    
    # historical_2B_stale = ["Nf-Nf-Nfnap", "Na-Na-Ne", "Na-Nd-Ni", "Nd-NmO-Nv", "Na-Na-Nkeqm", "Na-Nd-Nfn", "Nd-Nfn-Nk", "Na-Ne-Nwe", "Nf-Nfex-Nfnap", "Nfn-Nf-Nfnap", "Nfe-Na-Nk", "Nk-Nkeqm-Nk", "Nfex-Nfnap-Nfe", "Nkeqm-Nfnap-Nk", "Na-Nfe-Nfnap"]
    
    # # Unknown / not part of the confirmed canonical tripeptoid campaign
    # unknown_peptoids = ["Nk-Nfex-Nfex-Nk-Nfex-Nfex", "NfeC4-Nf-NmO-NfeC4-Nf-NmO", "Nf-Na-Nfes-Nf-Na-Nfes", "Nke-NfeB4-Nwe-Nke-NfeB4-Nwe", "Nke-Nwe-NfeB4-Nke-Nwe-NfeB4"]
    
    # # Known physical duplicates
    # physical_duplicates = {"Nfe-Nfnap-Nfe": "Two completed historical runs; run_0 is preferred.", **{pep: "Accidental Cycle-2 D duplicate: use run_ML_cycle_02, not run_ML_cycle_01." for pep in peptorun_2D}}
 
    
    # Active submission list — intentionally empty by default
    peptorun = []
    
    
    
    # bilayerstorun = ["POPC60CHOL40_W_WF", "POPE75POPG25_W_WF"]#, "POPC80POPS20_W_WF"] #has also been used before -> Alex & Hamish
    # bilayerstorun = ["POPC60CHOL40_W_WF"]
    # bilayerstorun = ["POPE75POPG25_W_WF"]
    # bilayerstorun = ["POPC80POPS20_W_WF"]
    #bilayerstorun = ["POPC60CHOL40_W_WF", "POPC1_W_WF"]
    bilayerstorun = ["POPC60CHOL40_W_WF", "POPE75POPG25_W_WF"]
    
    nmoltorun = [80]
    #nmoltorun = [20, 40, 80, 160] #normally just 80, 213 if comparing to Hamish's work, other nmol for concentration / aggregation studies [20, 40, 80, 160, 213]
    
    for peptide in peptorun:
        for bilayer in bilayerstorun:
            for nmol in nmoltorun:
                print(f"Running: {peptide} with {bilayer}")
                cbox = Simulations(peptide, nmol, GPU=False, run="ML_cycle_01", cpus=8, peptoid=True, bilayer=bilayer)
    
                # =================================================================
                # SAFETY GUARD: leave simulations completely untouched if their
                # Pull or Relax job is currently present in the Slurm queue.
                # =================================================================
                queued_jobs = [
                    line.strip()
                    for line in subprocess.check_output(
                        ["squeue", "-u", cbox.username, "-h", "-o", "%.200j"],
                        text=True
                    ).splitlines()
                ]
                
                pull_job = cbox.job_name("Pull")
                relax_job = cbox.job_name("Relax")
                
                if pull_job in queued_jobs or relax_job in queued_jobs:
                    print(f"SKIPPING ACTIVE JOB: {peptide} / {bilayer} / {nmol}")
                    continue
                # =================================================================
    
                status = cbox.CheckFolder()
                    
                # Running the script to this point will check that a lot of things are set correctly or tell you otherwise -> You can always uncomment the sys.exit() if you want to use it as a test
                # sys.exit()
    
                # Running the next line will execute the peptide/toid prep, and set up and run the Pull simulation
                cbox.StatusHandler(status)


    # =============================================================================
#     sys.exit()
#     if status == 'Pull':
#         print(f"Status is at stage: {status}")
#         cbox.Preptide()
#         cbox.reset_box()
#         cbox.SetupBox()
#         cbox.MinBox()
#         cbox.SpecialIndex()
#         cbox.SetupPulling()
#         cbox.SubmitJob("Pull")
#     elif status == 'Relax':
#         print(f"Status is at stage: {status}")
#         os.chdir(cbox.peptide_folder)
#         cbox.StatusHandler('Relax')
#     elif status == 'CpHMD':
#         pass
# =============================================================================
