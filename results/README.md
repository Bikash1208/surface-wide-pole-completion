###### `results/` — generated figures and tables

This directory holds outputs produced by the scripts (PDF/PNG figures, CSV table
data). **It is git-ignored** (see `.gitignore`): these artifacts are fully
reproducible from the scripts and cached `data/`, so they are not committed.

Reproduce everything with the commands in the top-level `README.md`. Each figure
script writes here via its `--out results/<name>.pdf` argument (or the default
output path).
****