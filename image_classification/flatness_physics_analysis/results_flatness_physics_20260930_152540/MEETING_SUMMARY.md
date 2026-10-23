# Flatness ↔ physical-response meeting summary

## Data/QC
- Morphology systems: **395**.
- Strictly matched to nmol=80 master APL data: **393**; unmatched: **2**.
- Morphology QC: **385 PASS**, **10 REVIEW_REQUIRED**.
- Annotation cross-check: **395/395 exact manual-label matches**.

## Do visual curved/flat labels correspond to quantitative morphology?
Inclusive sensitivity definition: curved = curved + bulge_curved; flat = flat_interaction + flat_no_interaction + bulge_flat; pore_formed excluded.
- **height_rms_delta_nm**: curved median 0.219 vs flat 0.094; rank-biserial = 0.731; BH-adjusted p = 3.66e-32 (n=147 curved, 237 flat).
- **excess_area_delta_fraction**: curved median 0.024 vs flat 0.009; rank-biserial = 0.718; BH-adjusted p = 4.53e-31 (n=147 curved, 237 flat).
- **curvature_delta_nm_inv**: curved median 0.046 vs flat 0.029; rank-biserial = 0.477; BH-adjusted p = 1.21e-14 (n=147 curved, 237 flat).
- **thickness_delta_nm**: curved median -0.150 vs flat -0.149; rank-biserial = -0.065; BH-adjusted p = 0.317 (n=147 curved, 237 flat).

## Does projected ΔAPL distinguish visually curved vs flat systems within each primary membrane?
- **bacterial**: curved median ΔAPL 5.611 vs flat 6.009; rank-biserial = -0.023; BH-adjusted p = 0.898.
- **mammalian**: curved median ΔAPL 2.663 vs flat 2.716; rank-biserial = -0.073; BH-adjusted p = 0.723.

## ddAPL versus bacterial–mammalian structural differences (primary matched tripeptoids)
- **height_rms_delta_nm**: Spearman ρ = -0.239, BH-adjusted p = 0.003, n=156 pairs.
- **excess_area_delta_fraction**: Spearman ρ = -0.436, BH-adjusted p = 4.52e-08, n=156 pairs.
- **curvature_delta_nm_inv**: Spearman ρ = -0.368, BH-adjusted p = 4.55e-06, n=156 pairs.
- **thickness_delta_nm**: Spearman ρ = -0.773, BH-adjusted p = 4.31e-31, n=156 pairs.
- **height_asym_delta_nm**: Spearman ρ = -0.240, BH-adjusted p = 0.003, n=156 pairs.
- **excess_asym_delta**: Spearman ρ = -0.369, BH-adjusted p = 4.55e-06, n=156 pairs.
- **curvature_asym_delta_nm_inv**: Spearman ρ = -0.270, BH-adjusted p = 9.17e-04, n=156 pairs.

## Example systems: relatively small ddAPL but large bacterial–mammalian height-RMS difference
- `Nf-Nw-Nl` / `run_ML_cycle_01`: ddAPL=0.121; ΔΔ height RMS=0.304 nm.
- `Ni-Nfex-Nfnap` / `run_ML_cycle_01`: ddAPL=1.697; ΔΔ height RMS=0.302 nm.
- `Nfe-Nfnap-Nk` / `run_ML_cycle_02`: ddAPL=1.591; ΔΔ height RMS=0.294 nm.
- `Nf-Nfe-Nfnap` / `run_0`: ddAPL=-0.558; ΔΔ height RMS=0.291 nm.
- `Nfex-Nfex-Nfex` / `run_0`: ddAPL=-1.614; ΔΔ height RMS=0.266 nm.

## Strong metric relationships (all PASS systems; |ρ| ≥ 0.85)
- excess_asym_delta vs curvature_asym_delta_nm_inv: ρ=0.921 (n=383).
- APL_change vs thickness_delta_nm: ρ=-0.905 (n=383).
- height_rms_delta_nm vs excess_area_delta_fraction: ρ=0.894 (n=383).
- excess_area_delta_fraction vs curvature_delta_nm_inv: ρ=0.863 (n=383).

## Endpoint persistence check
- height_rms: 100 ns endpoint vs 80–100 ns median ρ=0.899 (n=385).
- excess_area: 100 ns endpoint vs 80–100 ns median ρ=0.941 (n=385).
- curvature: 100 ns endpoint vs 80–100 ns median ρ=0.922 (n=385).
- thickness: 100 ns endpoint vs 80–100 ns median ρ=0.979 (n=385).

## Interpretation to discuss
- Quantitative morphology should be treated as complementary structural information, not as a replacement for ddAPL.
- Strong separation of visual curved/flat classes by height RMS/excess area/curvature, alongside weak within-membrane ΔAPL separation, would directly support the argument that projected APL misses some visually obvious structural deformation.
- Thickness is expected to track APL strongly and should not automatically be treated as an independent morphology axis.
- Use the discordant systems as candidate supervisor-slide examples, then inspect their images before making a final qualitative claim.

## Output locations
- `task_0/`: strict merge/QC
- `task_1/`: visual class vs morphology
- `task_2/`: visual class vs ΔAPL
- `task_3/`: matched-pair ddAPL vs structural differences
- `task_4/`: discordant representative systems
- `task_5/`: metric redundancy/correlations
- `task_6/`: membrane-label association and temporal persistence
