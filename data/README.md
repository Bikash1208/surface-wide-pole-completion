# `data/` — regenerated intermediate results (not committed)

The figure and table scripts consume FDTD/post-processing outputs that are **not**
stored in this repository. Regenerate them with the `run_*` scripts (Step 1 in the
top-level README); by default they are written to the repository root.

| File | Produced by | Consumed by |
|---|---|---|
| `paperB_fig3_data.npz` | `run_spectrum_data.py` | Fig. 1 |
| `paperB_multibody_validation_spectra.npz` | `run_multibody_transfer.py` | Fig. 1, Table I |
| `paperB_cutoff_sweep_spectra.npz` + `paperB_cutoff_sweep.csv` | `run_cutoff_sweep.py` | Fig. 2(a) |
| `paperB_2xgrid_reference.npz` | `run_grid_reference.py` | Table I (vs reference) |
| `paperB_fig4_data.npz` | `compute_exact_mie_poles.py extract` | Tables II, III; Fig. 5 marker |
| `paperB_pc1_v2.npz` | `run_estimator_baselines.py` (anchor cache) | Table III |
| `workflow_solver_times.csv`, `paperB_measured_postprocess_times.csv` | `measure_workflow_times.py` | timing numbers (§III) |

Notes:
- Each `.npz` stores `float64` spectra on the manuscript band (1.20–2.60 GHz,
  318 FFT bins) unless its generator states otherwise.
- These files are `.gitignore`-d. Commit them only if you deliberately want
  one-command figure rebuilds without re-running the FDTD solver.
