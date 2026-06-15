# Surface-Wide Common-Pole Ringdown Completion — reproduction code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20698908.svg)](https://doi.org/10.5281/zenodo.20698908)

Reproduction code for the IEEE *Antennas and Wireless Propagation Letters* (AWPL)
manuscript **"Surface-Wide Common-Pole Ringdown Completion for Accelerated FDTD
RCS Computation."**

The method stops a resonant FDTD scattering run early, fits one global set of
natural poles to the stacked Huygens-surface record, and analytically completes
the discarded late-time tail before the near-to-far-field transform. This
repository contains the in-house 3-D Yee FDTD solver and the experiment scripts
that generate every figure and table. Intermediate data are **not committed**;
they are regenerated from the `run_*` scripts (see Step 1 below).

## Requirements

- Python ≥ 3.10
- `numpy`, `scipy`, `numba`, `matplotlib`, `h5py`, `tqdm` (see `requirements.txt`)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The FDTD kernels are Numba-JIT, `parallel=True`, `float32`. The first run of any
solver script is slow (JIT compile); subsequent runs are fast. Do **not** wrap
the scripts in an external process pool — the kernels already use all cores.

## Layout

```
.
├── README.md  VERIFICATION.md  requirements.txt  .gitignore  LICENSE  CITATION.cff
├── constants.py            shared invariants (band, FFT length, geometry)
├── plotting_utils.py       IEEE figure style, palette, dB metrics
├── generate_manuscript_figures.py    dispatcher over the plot_* scripts
│
│   # solver core (validated; unchanged)
├── fdtd_engine.py  tfsf.py  geometry.py  spherical_recorder.py
├── fdtd_extrapolate.py  ntff_rcs.py  mie_sphere.py
├── awpl_pipeline.py        shared run / spectrum / pole-selection helpers
├── compute_exact_mie_poles.py   exact + stratified Mie pole references
│
│   # experiment / data generators  (run_*.py, measure_*.py)
├── run_spectrum_data.py  run_cutoff_sweep.py  run_multibody_transfer.py
├── run_grid_reference.py  run_estimator_baselines.py  measure_workflow_times.py
├── run_cross_permittivity_poles.py  run_cross_permittivity_rcs.py
│
│   # figure scripts  (plot_*.py)
├── plot_fig1_spectra.py  plot_fig2a_pareto.py  plot_fig2b_convergence.py
├── plot_fig3_bistatic.py  plot_fig4_cylinder_poles.py  plot_fig5_pole_scatter.py
├── plot_sensitivity_supplement.py  plot_variance_supplement.py
│
├── data/      where regenerated .npz/.csv land (not committed; see data/README.md)
└── results/   generated figures/tables (git-ignored; see results/README.md)
```

Run every script from the repository root; the `.npz/.csv` produced by the data
scripts are written to / read from the root by relative name.

## Step 1 — regenerate the intermediate data (FDTD; minutes each)

```bash
python run_spectrum_data.py                   # -> paperB_fig3_data.npz            (Fig. 1)
python run_multibody_transfer.py --cut 4000   # -> *_spectra.npz                   (Fig. 1, Table I)
python run_cutoff_sweep.py                    # -> *_spectra.npz + .csv            (Fig. 2a)
python run_grid_reference.py                  # -> Table I (vs reference); ~16x cost
python compute_exact_mie_poles.py extract     # -> paperB_fig4_data.npz            (Tables II/III)
python measure_workflow_times.py              # -> timing CSVs
```

## Step 2 — build the figures (outputs to `results/`)

```bash
python generate_manuscript_figures.py --all          # all figures
python generate_manuscript_figures.py --figures 1 2a # a subset
```

Or run the per-figure scripts directly. Figures 1 and 2(a) only *plot* (they read
the Step-1 data files); Figures 2(b)–5 invoke the FDTD solver themselves.

```bash
python plot_fig1_spectra.py                                              # Fig. 1
python plot_fig2a_pareto.py                                              # Fig. 2(a)
python plot_fig2b_convergence.py   --out results/fig2b_convergence.pdf   # Fig. 2(b)
python plot_fig3_bistatic.py       --out results/fig3_bistatic.pdf       # Fig. 3
python plot_fig4_cylinder_poles.py --out results/fig4_cylinder_poles.pdf # Fig. 4
python plot_fig5_pole_scatter.py   --out results/fig5_pole_scatter.pdf   # Fig. 5
```

> **Fig. 3 / Fig. 4 panel split.** These two scripts emit a single two-panel PDF;
> the manuscript places the panels as separate files (`_1/_2`, `_a/_b`). The
> plotted content is identical — only the file split differs.

## Tables and quoted numbers

| Output | Script(s) |
|---|---|
| Table I (early-stop transfer, vs full + vs reference) | `run_multibody_transfer.py --cut 4000`, then `run_grid_reference.py` |
| Table II (cross-permittivity LOO: f, Q, peak) | `run_cross_permittivity_poles.py` (f, Q), `run_cross_permittivity_rcs.py` (peak) |
| Table III (estimator independence: MPM / ESPRIT / VF) | `run_estimator_baselines.py` |
| Timing (≈75 % solver, 59–65 % end-to-end) | `measure_workflow_times.py` |
| Sensitivity / random-projection ranges (text) | `plot_sensitivity_supplement.py`, `plot_variance_supplement.py` |

See `VERIFICATION.md` for the expected output of each script.

## Configuration

Reference-run parameters are fixed in the scripts (core subset mirrored in
`constants.py`) and match the manuscript:

- Grid Δ = 2 mm; sphere radius 24 mm (RC = 12 cells); 8-cell CPML.
- Full run `N_total` = 16000 steps; 75 % early stop at `N_cut` = 4000.
- Pole fit: window start `T0` = 600, length 2400, pencil order 28, `Q ≥ 3`.
- Huygens surface 24 × 48 nodes (6912 signals); rFFT length 65536 (Δf ≈ 4.40 MHz).
- Drive: Gaussian pulse, 1.8 GHz (sphere) / 1.9 GHz (other bodies), fractional bandwidth 1.4.

Three frequency bands appear **by design — do not unify them**:

- **1.20–2.60 GHz** — RCS RMS evaluation band (figures, tables).
- **0.80–3.00 GHz** — wider Mie pole-search band (`compute_exact_mie_poles.py`).
- **0.90–2.80 GHz** — estimator candidate band (`run_estimator_baselines.py`).

## Notes

- **In-house solver, no proprietary software.** Everything runs on the bundled
  Yee FDTD solver; no commercial EM package is required.
- **Determinism.** Random projections use fixed seeds; FDTD is deterministic.
  Wall-clock timings depend on hardware and are reported as such.
- **Scientific parameters** (grid `Δ`, `N_total`, cut step, pencil order,
  band, fit window) are fixed in the scripts and documented in the manuscript.

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the AWPL letter (details filled on acceptance).
