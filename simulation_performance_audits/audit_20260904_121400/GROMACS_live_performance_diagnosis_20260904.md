# GROMACS live performance diagnosis — 4 September 2026

## Technical summary

Slurm allocated the requested eight CPUs to every inspected job. The failure is application-side utilisation/progress after allocation, and it is strongly concentrated on `node036`, `node039`, `node042`, and `node043`.

On healthy nodes, comparable eight-CPU jobs consumed 98.9–99.2% of their allocated CPU capacity and reached roughly 69–86 ns in about 8–10.5 hours. On the four affected nodes, most jobs consumed only 40–53% of their allocation and reached roughly 15–22 ns after about 53 hours. Two additional jobs consumed only about 3% of their allocation, consistent with computing briefly and then becoming nearly idle or stalled.

This establishes a real computational incident. It does not yet prove whether the initiating mechanism is GROMACS/OpenMP affinity, node health, interaction with densely packed jobs, or another node-local problem.

## Verified allocation and utilisation evidence

The inspected Slurm records report `NumCPUs=8`, `CPUs/Task=8`, and `AllocTRES=cpu=8` for every job. Therefore the scheduler did not silently allocate only one or two CPUs.

| Node group | Jobs inspected | Typical CPU efficiency | Observed MD progress |
|---|---:|---:|---:|
| `node022`, `node023`, `node030`, `node031`, `node054` | 6 | 98.9–99.2% | 69–86 ns after about 8–10.5 h |
| `node035` | 1 | 83.1% | 75.3 ns after about 10.3 h |
| `node036`, `node039`, `node042`, `node043` | 16 | mostly 40–53%; two near 3% | mostly 4.5–22.1 ns after about 24–53 h |

For an eight-CPU job, CPU efficiency is:

`aggregate CPU time / (elapsed wall time × 8)`

For example, job `8016108` had run for 53.34 hours. Full use of eight CPUs would produce about 426.7 aggregate CPU-hours, but `sstat` reported about 199.1 CPU-hours: 46.7% efficiency, equivalent to an average of 3.73 busy CPUs. Job `8027068`, by contrast, was 99.2% efficient, equivalent to 7.94 busy CPUs.

Jobs `8016391` and `8024748` averaged only 0.27 and 0.24 busy CPUs respectively. Their CPU-hours are approximately what a healthy run would need to generate their recorded 16.8 ns and 10.0 ns, suggesting they computed initially and then spent most of their wall time idle or stalled.

## Interpretation

The earlier GROMACS line `Overriding thread affinity set outside gmx mdrun`, together with `-pin on`, makes affinity conflict a plausible mechanism: GROMACS may replace scheduler-provided placement. The node concentration also makes node-local contention, packing, or health a plausible contributor.

These are hypotheses, not yet proven causes. The available data do prove that the problem is not normal molecular-size variation and is not explained by Slurm allocating only one or two CPUs.

## Immediate operational recommendation

1. Do not let the pathological jobs continue to the three-day limit; at their present progress they cannot finish 100 ns in that allocation.
2. Before restarting, distinguish canonical required simulations from duplicate/noncanonical cycle directories. Do not bulk-resubmit every affected job.
3. Preserve/check the current phase checkpoint (`Pull.cpt` or `Relax.cpt`).
4. Restart one confirmed-required job as a controlled test, temporarily excluding `node036,node039,node042,node043`.
5. Let Slurm own CPU binding: launch the single MPI rank with `srun --cpu-bind=cores`, set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, and use GROMACS `-pin off`. Request physical cores with `--hint=nomultithread` if Archie supports it.
6. After 20–30 minutes, verify both CPU efficiency and simulation-time advancement. Proceed with the remaining required jobs only if the test is normal.
7. Send the live diagnostic and one slow/one healthy GROMACS log to Archie support; the four-node concentration is actionable cluster evidence.

## Runtime comparison

If a normal run takes 12 hours on eight cores, ideal proportional runtimes are 48 hours on two cores and 96 hours on one core. Actual GROMACS scaling will not be perfectly linear, but these are useful order-of-magnitude expectations.

At 10% completion after 48 hours, the projected total is 480 hours (20 days). That throughput is equivalent to only about 0.2 normally performing core, so it is substantially worse than a legitimate one-core run.

## Remaining evidence needed for root-cause proof

- one complete slow `Pull.log` or `Relax.log` plus its `Slurm.log`;
- one complete healthy current log for comparison;
- process/thread affinity or task layout from a running affected job;
- node topology/health confirmation from Archie support.

## Sources

- `live_gromacs_diagnostics_20260904_125818.txt` (provided output)
- `simulation_performance_detailed.csv` (previous audit output)

