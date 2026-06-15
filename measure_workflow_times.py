#!/usr/bin/env python3
"""
measure_workflow_times.py

Measure actual solver runtimes for the workflow-time bar plot.

This script reruns the FDTD time loops for the four bodies used in the paper:

    solid sphere
    hollow shell
    triaxial ellipsoid
    finite cylinder

It writes:
    workflow_solver_times.csv
    paperB_measured_postprocess_times.csv
    paperB_measured_workflow_timing_data.csv
    paperB_measured_workflow_timing.png
    paperB_measured_workflow_timing.pdf

Timing convention:
    The timer starts immediately before the FDTD update loop and stops
    immediately after the loop. Geometry/material setup is not included.
    This matches the timing style used in the existing paperB scripts.

Default behavior:
    - measure full solver time using 16000 steps
    - measure actual early-stop solver time using 4000 steps
    - measure global-pole post-processing on the full solid-sphere record
    - measure per-node MPM on a 64-signal subset and scale to all 6K signals

For a fully measured per-node baseline without subset scaling, use:
    --pernode-full

That can take a long time.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------- shared constants ----------------
EPS_R = 25.0
C0_ = 299792458.0

RC = 12
D = 2e-3
NPML = 8
BUFFER = 4
HY_GAP = 6
TF_GAP = 3

N_FULL = 16000
N_CUT = 4000
T0 = 600
ORDER = 28
NTH, NPH = 24, 48
N_SUB = 64
N_REP_POST = 3

COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

BODIES = {
    "solid sphere": None,
    "hollow shell": None,
    "ellipsoid": {"semi": np.array([14, 11, 8]) * 2e-3},
    "cylinder": {"radius": 16e-3, "height": 40e-3},
}


@dataclass
class SolverTiming:
    body: str
    full_s: float
    cut_s: float
    full_steps: int
    cut_steps: int
    grid: str


@dataclass
class PostTiming:
    pernode_s: float
    global_pole_s: float
    global_pole_est_s: float
    global_residue_s: float
    pernode_mode: str
    n_signals: int


# ---------------- geometry helpers ----------------
def grid_for_sphere() -> Tuple[int, int, np.ndarray, int, int, float]:
    half = RC + HY_GAP + BUFFER + NPML
    ng = 2 * half + (2 * half) % 2
    cc = ng // 2
    center = np.array([cc, cc, cc]) * D
    lo, hi = cc - RC - TF_GAP, cc + RC + TF_GAP
    rsph = (RC + HY_GAP) * D
    return ng, cc, center, lo, hi, rsph


def build_mask_general(body: Dict[str, np.ndarray], sim, center: np.ndarray, d: float):
    from geometry import grid_coords

    x, y, z = grid_coords(sim)
    X, Y, Z = np.meshgrid(x - center[0], y - center[1], z - center[2], indexing="ij")
    if "semi" in body:
        s = body["semi"]
        return (X / s[0]) ** 2 + (Y / s[1]) ** 2 + (Z / s[2]) ** 2 <= 1.0

    r, h = body["radius"], body["height"]
    return (X**2 + Y**2 <= r**2) & (np.abs(Z) <= h / 2)


def amax_of(body: Dict[str, np.ndarray]) -> float:
    if "semi" in body:
        return float(max(body["semi"]))
    return float(max(body["radius"], body["height"] / 2))


# ---------------- FDTD runs ----------------
def run_solid_sphere(n_steps: int, keep_record: bool = False):
    from fdtd_engine import FDTD
    from geometry import sphere_mask
    from spherical_recorder import SphericalHuygensRecorder
    from tfsf import PlaneWaveBox, gaussian_modulated

    ng, cc, center, lo, hi, rsph = grid_for_sphere()

    sim = FDTD(ng, ng, ng, D, D, D, npml=NPML)
    sim.set_material(sphere_mask(sim, center, RC * D), eps_r=EPS_R)
    sim.finalize_material()

    wf = gaussian_modulated(1.8e9, 1.4 * 1.8e9, sim.dt)
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, wf)
    rec = SphericalHuygensRecorder(sim, center, rsph, NTH, NPH, n_steps=n_steps)

    t0 = time.perf_counter()
    for n in range(n_steps):
        sim.update_H()
        src.correct_H(sim, n)
        sim.update_E()
        src.correct_E(sim, n)
        rec.record(sim, n)
    elapsed = time.perf_counter() - t0

    if keep_record:
        return sim.dt, rec, wf, elapsed, f"{ng}^3"
    return elapsed, f"{ng}^3"


def run_hollow_shell(n_steps: int):
    from fdtd_engine import FDTD
    from geometry import sphere_mask
    from spherical_recorder import SphericalHuygensRecorder
    from tfsf import PlaneWaveBox, gaussian_modulated

    ng, cc, center, lo, hi, rsph = grid_for_sphere()
    a_out = 0.024
    a_core = 0.012

    sim = FDTD(ng, ng, ng, D, D, D, npml=NPML)
    shell = sphere_mask(sim, center, a_out) & ~sphere_mask(sim, center, a_core)
    sim.set_material(shell, eps_r=EPS_R)
    sim.finalize_material()

    wf = gaussian_modulated(1.9e9, 1.4 * 1.9e9, sim.dt)
    src = PlaneWaveBox(sim, lo, hi, lo, hi, lo, hi, wf)
    rec = SphericalHuygensRecorder(sim, center, rsph, NTH, NPH, n_steps=n_steps)

    t0 = time.perf_counter()
    for n in range(n_steps):
        sim.update_H()
        src.correct_H(sim, n)
        sim.update_E()
        src.correct_E(sim, n)
        rec.record(sim, n)
    elapsed = time.perf_counter() - t0

    return elapsed, f"{ng}^3"


def run_general_body(name: str, body: Dict[str, np.ndarray], n_steps: int):
    from fdtd_engine import FDTD
    from spherical_recorder import SphericalHuygensRecorder
    from tfsf import PlaneWaveBox, gaussian_modulated

    d = D
    am = int(round(amax_of(body) / d))
    hy = int(round(6 * D / d))
    buf = int(round(4 * D / d))
    tg = int(round(3 * D / d))
    half = am + hy + buf + NPML
    ng = 2 * half + (2 * half) % 2
    cc = ng // 2
    center = np.array([cc, cc, cc]) * d

    sim = FDTD(ng, ng, ng, d, d, d, npml=NPML)
    sim.set_material(build_mask_general(body, sim, center, d), eps_r=EPS_R)
    sim.finalize_material()

    wf = gaussian_modulated(1.9e9, 1.4 * 1.9e9, sim.dt)
    src = PlaneWaveBox(
        sim,
        cc - am - tg,
        cc + am + tg,
        cc - am - tg,
        cc + am + tg,
        cc - am - tg,
        cc + am + tg,
        wf,
    )
    rec = SphericalHuygensRecorder(sim, center, (am + hy) * d, NTH, NPH, n_steps=n_steps)

    t0 = time.perf_counter()
    for n in range(n_steps):
        sim.update_H()
        src.correct_H(sim, n)
        sim.update_E()
        src.correct_E(sim, n)
        rec.record(sim, n)
    elapsed = time.perf_counter() - t0

    return elapsed, f"{ng}^3"


def measure_solver_times(measure_cut: bool, keep_solid_full_record: bool):
    rows: List[SolverTiming] = []
    solid_record = None

    for body_name in BODIES:
        print(f"\n=== measuring {body_name}: full {N_FULL} steps ===", flush=True)

        if body_name == "solid sphere":
            if keep_solid_full_record:
                dt, rec, wf, full_s, grid = run_solid_sphere(N_FULL, keep_record=True)
                solid_record = (dt, rec, wf)
            else:
                full_s, grid = run_solid_sphere(N_FULL, keep_record=False)
        elif body_name == "hollow shell":
            full_s, grid = run_hollow_shell(N_FULL)
        else:
            full_s, grid = run_general_body(body_name, BODIES[body_name], N_FULL)

        print(f"  full: {full_s:.3f} s, grid {grid}", flush=True)

        if measure_cut:
            print(f"=== measuring {body_name}: cut {N_CUT} steps ===", flush=True)
            if body_name == "solid sphere":
                cut_s, _ = run_solid_sphere(N_CUT, keep_record=False)
            elif body_name == "hollow shell":
                cut_s, _ = run_hollow_shell(N_CUT)
            else:
                cut_s, _ = run_general_body(body_name, BODIES[body_name], N_CUT)
            print(f"  cut : {cut_s:.3f} s", flush=True)
        else:
            cut_s = full_s * (N_CUT / N_FULL)
            print(f"  cut : {cut_s:.3f} s estimated from step fraction", flush=True)

        rows.append(
            SolverTiming(
                body=body_name,
                full_s=full_s,
                cut_s=cut_s,
                full_steps=N_FULL,
                cut_steps=N_CUT,
                grid=grid,
            )
        )

    return rows, solid_record


# ---------------- post-processing timing ----------------
def measure_postprocessing_from_record(rec, repeats: int, pernode_full: bool) -> PostTiming:
    from fdtd_extrapolate import mpm_poles

    full = {c: np.asarray(rec.data[c], np.float32) for c in COMPS}
    Yall = np.concatenate([full[c] for c in COMPS], axis=1).astype(np.float64)
    K6 = Yall.shape[1]
    Yw = Yall[T0:N_CUT]
    W = Yw.shape[0]

    global_pole_times = []
    global_residue_times = []
    pernode_times = []

    for rep in range(repeats):
        # global path: PC1 by power iteration + one MPM
        t0 = time.perf_counter()
        v = np.random.default_rng(rep).normal(size=Yw.shape[1])
        for _ in range(12):
            u = Yw @ v
            v = Yw.T @ u
            v /= np.linalg.norm(v)
        pc1 = Yw @ v
        z, _ = mpm_poles(pc1, ORDER)
        t_pole = time.perf_counter() - t0

        # residue solve
        t0 = time.perf_counter()
        Vinv = np.linalg.pinv(z[None, :] ** np.arange(W)[:, None])
        _ = Vinv @ Yw
        t_res = time.perf_counter() - t0

        global_pole_times.append(t_pole)
        global_residue_times.append(t_res)

        # per-node MPM
        if pernode_full:
            idx = np.arange(K6)
            mode = "full measured over all signals"
        else:
            idx = np.random.default_rng(rep).choice(K6, min(N_SUB, K6), replace=False)
            mode = f"measured on {len(idx)} signals and scaled to {K6}"

        t0 = time.perf_counter()
        for j in idx:
            mpm_poles(Yw[:, j], ORDER)
        raw = time.perf_counter() - t0
        scaled = raw if pernode_full else raw * K6 / len(idx)
        pernode_times.append(scaled)

    gpole = float(np.mean(global_pole_times))
    gres = float(np.mean(global_residue_times))
    pnode = float(np.mean(pernode_times))

    print("\nPost-processing timing")
    print(f"  6K = {K6}, W = {W}, M = {ORDER}, repeats = {repeats}")
    print(f"  global pole estimation: {gpole:.6g} s")
    print(f"  global residue solve  : {gres:.6g} s")
    print(f"  global total          : {gpole + gres:.6g} s")
    print(f"  per-node MPM          : {pnode:.6g} s ({mode})")

    return PostTiming(
        pernode_s=pnode,
        global_pole_s=gpole + gres,
        global_pole_est_s=gpole,
        global_residue_s=gres,
        pernode_mode=mode,
        n_signals=K6,
    )


# ---------------- saving and plotting ----------------
def save_solver_csv(rows: List[SolverTiming], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["body", "solver_full_s", "solver_cut_s", "full_steps", "cut_steps", "grid"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "body": r.body,
                    "solver_full_s": f"{r.full_s:.6g}",
                    "solver_cut_s": f"{r.cut_s:.6g}",
                    "full_steps": r.full_steps,
                    "cut_steps": r.cut_steps,
                    "grid": r.grid,
                }
            )


def save_post_csv(post: PostTiming, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pernode_MPM_s",
                "global_total_s",
                "global_pole_estimation_s",
                "global_residue_solve_s",
                "pernode_mode",
                "n_signals",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pernode_MPM_s": f"{post.pernode_s:.6g}",
                "global_total_s": f"{post.global_pole_s:.6g}",
                "global_pole_estimation_s": f"{post.global_pole_est_s:.6g}",
                "global_residue_solve_s": f"{post.global_residue_s:.6g}",
                "pernode_mode": post.pernode_mode,
                "n_signals": post.n_signals,
            }
        )


def build_workflow_rows(solver_rows: List[SolverTiming], post: PostTiming):
    rows = []
    for r in solver_rows:
        red = r.full_s + post.pernode_s
        blue = r.full_s + post.global_pole_s
        green = r.cut_s + post.global_pole_s
        gain = red / green
        rows.append(
            {
                "body": r.body,
                "solver_full_s": r.full_s,
                "solver_cut_s": r.cut_s,
                "full_run_plus_pernode_MPM_s": red,
                "full_run_plus_global_pole_s": blue,
                "early_stop_plus_global_pole_s": green,
                "gain_vs_pernode": gain,
            }
        )
    return rows


def save_workflow_csv(rows: List[Dict[str, float]], path: Path) -> None:
    fields = [
        "body",
        "solver_full_s",
        "solver_cut_s",
        "full_run_plus_pernode_MPM_s",
        "full_run_plus_global_pole_s",
        "early_stop_plus_global_pole_s",
        "gain_vs_pernode",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (f"{v:.6g}" if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )


def plot_workflow(rows: List[Dict[str, float]], out_png: Path, out_pdf: Path) -> None:
    labels = {
        "solid sphere": "solid\nsphere",
        "hollow shell": "hollow\nshell",
        "ellipsoid": "ellipsoid",
        "cylinder": "cylinder",
    }

    x = np.arange(len(rows))
    width = 0.24

    red = [r["full_run_plus_pernode_MPM_s"] for r in rows]
    blue = [r["full_run_plus_global_pole_s"] for r in rows]
    green = [r["early_stop_plus_global_pole_s"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.bar(x - width, red, width, label="full run + per-node MPM")
    ax.bar(x, blue, width, label="full run + global-pole")
    ax.bar(x + width, green, width, label="early stop + global-pole")

    ax.set_yscale("log")
    ax.set_ylabel("workflow time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[r["body"]] for r in rows])
    ax.grid(axis="y", which="major", alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    for xi, row, val in zip(x, rows, green):
        ax.text(
            xi + width,
            val * 1.15,
            f"×{row['gain_vs_pernode']:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def print_summary(solver_rows: List[SolverTiming], post: PostTiming, workflow_rows):
    print("\n==================== MEASURED WORKFLOW SUMMARY ====================")
    print(f"post-processing: per-node {post.pernode_s:.3f} s | global {post.global_pole_s:.3f} s")
    print("body            full solver   cut solver    red workflow   blue workflow  green workflow  gain")
    for r, w in zip(solver_rows, workflow_rows):
        print(
            f"{r.body:15s} "
            f"{r.full_s:11.3f} "
            f"{r.cut_s:11.3f} "
            f"{w['full_run_plus_pernode_MPM_s']:13.3f} "
            f"{w['full_run_plus_global_pole_s']:13.3f} "
            f"{w['early_stop_plus_global_pole_s']:14.3f} "
            f"×{w['gain_vs_pernode']:.1f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Measure actual workflow timings and create Fig. 3.")
    p.add_argument("--no-measure-cut", action="store_true",
                   help="Do not run 4000-step cut simulations; estimate cut time from step fraction.")
    p.add_argument("--pernode-full", action="store_true",
                   help="Run per-node MPM on all 6K signals instead of subset scaling.")
    p.add_argument("--post-repeats", type=int, default=N_REP_POST,
                   help=f"Post-processing repeats. Default: {N_REP_POST}.")
    p.add_argument("--out-prefix", default="paperB_measured_workflow_timing",
                   help="Output prefix for CSV/PNG/PDF files.")
    return p.parse_args()


def main():
    args = parse_args()

    print("Machine:")
    print(" ", platform.platform())
    print(" ", platform.processor() or "(processor not reported)")
    print(" ", "Python", platform.python_version(), "| NumPy", np.__version__)
    print("Timing loop only; setup/material construction is not included.\n")

    solver_rows, solid_record = measure_solver_times(
        measure_cut=not args.no_measure_cut,
        keep_solid_full_record=True,
    )

    if solid_record is None:
        raise RuntimeError("Solid-sphere full record was not retained; cannot measure post-processing.")

    _, rec, _ = solid_record
    post = measure_postprocessing_from_record(
        rec=rec,
        repeats=args.post_repeats,
        pernode_full=args.pernode_full,
    )

    prefix = Path(args.out_prefix)
    solver_csv = Path("workflow_solver_times.csv")
    post_csv = Path("paperB_measured_postprocess_times.csv")
    workflow_csv = prefix.with_suffix(".csv")
    out_png = prefix.with_suffix(".png")
    out_pdf = prefix.with_suffix(".pdf")
    meta_json = prefix.with_suffix(".json")

    save_solver_csv(solver_rows, solver_csv)
    save_post_csv(post, post_csv)

    workflow_rows = build_workflow_rows(solver_rows, post)
    save_workflow_csv(workflow_rows, workflow_csv)
    plot_workflow(workflow_rows, out_png, out_pdf)

    meta = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "timing_convention": "FDTD update loop plus recorder only; setup excluded.",
        "constants": {
            "N_FULL": N_FULL,
            "N_CUT": N_CUT,
            "T0": T0,
            "ORDER": ORDER,
            "NTH": NTH,
            "NPH": NPH,
            "N_SUB": N_SUB,
            "post_repeats": args.post_repeats,
            "pernode_full": args.pernode_full,
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    print_summary(solver_rows, post, workflow_rows)
    print("\nSaved:")
    print(f"  {solver_csv}")
    print(f"  {post_csv}")
    print(f"  {workflow_csv}")
    print(f"  {out_png}")
    print(f"  {out_pdf}")
    print(f"  {meta_json}")


if __name__ == "__main__":
    main()
