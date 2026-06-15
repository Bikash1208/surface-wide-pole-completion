#!/usr/bin/env python3
"""
run_cross_permittivity_rcs.py  --  RUN THIS YOURSELF (no sandbox).

v1 RCS reconstruction looked bad (2-4% peak, -4 dB level, 8-10 dB RMS), but that mixed
THREE errors. This version separates them so we can see what's actually the ROM vs what's
the solver/model:

  (A) FDTD-pole-f  vs  Mie-peak-f   offset  -> the solver's grid-dispersion gap (NOT ROM)
  (B) anchor-fit RMS (Lorentzian vs Mie, no interpolation) -> the MODEL inadequacy (NOT ROM)
  (C) LOO RMS minus anchor RMS -> the actual ROM interpolation error

Fixes for the model: NARROW window (+/- a few linewidths) and PEAK-calibrate |C|^2 =
sigma_peak * a_m^2 (so the peak matches by construction; LOO then tests interpolation of
sigma_peak and a_m, w_m).

NOTE: comparing to Mie still carries the solver gap (A). For the publishable figure the
ground truth should be FDTD-DIRECT backscatter (NTFF) at the held-out eps_r -- that cancels
(A) and isolates the ROM. Wire your NTFF at the HOOK to switch ground truth.
"""
import numpy as np
from run_cross_permittivity_poles import extract_dipole, radius
from mie_sphere import mie_backscatter

EPS_LIST = [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0]
NF = 240; N_LW = 4.0                                  # window = +/- N_LW linewidths


def mie_spec(eps_r, f):
    return np.array([mie_backscatter(eps_r, radius, ff) for ff in f])


def anchor(eps_r):
    sd = extract_dipole(eps_r)[0]
    a_m, w_m = abs(sd.real), abs(sd.imag); f_m = w_m / (2*np.pi)
    df = N_LW * a_m / (2*np.pi)                       # +/- N_LW linewidths (HWHM=a_m in w)
    f = np.linspace(f_m - df, f_m + df, NF)
    # ---- HOOK: ground truth. Mie here; swap FDTD-NTFF backscatter for the real figure ----
    sig = mie_spec(eps_r, f)
    sig_peak = sig.max(); f_mie_peak = f[np.argmax(sig)]
    C2 = sig_peak * a_m**2                            # peak-calibrated amplitude
    fit = C2 / ((2*np.pi*f - w_m)**2 + a_m**2)
    rms_anchor = np.sqrt(np.mean((10*np.log10(fit+1e-30) - 10*np.log10(sig+1e-30))**2))
    return dict(eps=eps_r, a=a_m, w=w_m, f_m=f_m, C2=C2, sig_peak=sig_peak,
                f_mie_peak=f_mie_peak, f=f, sig=sig, rms_anchor=rms_anchor)


def main():
    print("building anchors (one FDTD each)...", flush=True)
    eps = np.array(EPS_LIST); D = [anchor(er) for er in eps]
    aA = np.array([d["a"] for d in D]); wA = np.array([d["w"] for d in D])
    spA = np.array([d["sig_peak"] for d in D])

    print("\n(A) solver gap + (B) model fit, per anchor:")
    print(" eps |  f_pole(GHz)  f_miePk |  pole-vs-Mie | anchor-fit RMS(dB)")
    for d in D:
        off = 100*abs(d["f_m"]-d["f_mie_peak"])/d["f_mie_peak"]
        print(f" {d['eps']:4.0f} |  {d['f_m']/1e9:8.3f}  {d['f_mie_peak']/1e9:7.3f} | "
              f"  {off:5.2f}%     |   {d['rms_anchor']:5.2f}")

    print("\n(C) LEAVE-ONE-OUT (peak-calibrated, narrow window):")
    print(" held eps | f_peak err | peak-level err | LOO RMS | (LOO-anchor) RMS")
    for i in range(1, len(eps)-1):
        keep = [k for k in range(len(eps)) if k != i]; deg = min(2, len(keep)-1)
        a_p = np.polyval(np.polyfit(eps[keep], aA[keep], deg), eps[i])
        w_p = np.polyval(np.polyfit(eps[keep], wA[keep], deg), eps[i])
        sp_p = np.exp(np.polyval(np.polyfit(eps[keep], np.log(spA[keep]), deg), eps[i]))
        C2_p = sp_p * a_p**2
        f = D[i]["f"]; st = D[i]["sig"]
        sp = C2_p / ((2*np.pi*f - w_p)**2 + a_p**2)
        fpk_t = f[np.argmax(st)]; fpk_p = f[np.argmax(sp)]
        lvl = 10*np.log10(sp.max()/st.max())
        rms = np.sqrt(np.mean((10*np.log10(sp+1e-30)-10*np.log10(st+1e-30))**2))
        print(f"  {eps[i]:6.1f}  |  {100*abs(fpk_p-fpk_t)/fpk_t:6.2f}%  |  {lvl:+6.2f} dB    |"
              f"  {rms:5.2f}  |   {rms - D[i]['rms_anchor']:+5.2f}")

    print("\nDECODE:")
    print("  (A) pole-vs-Mie % ~ the f_peak err  => the peak shift is SOLVER grid dispersion,")
    print("      not the ROM. Compare to FDTD-direct (NTFF) to remove it.")
    print("  (B) anchor-fit RMS ~ the LOO RMS    => the dB RMS is single-Lorentzian MODEL")
    print("      inadequacy over the window, not interpolation.")
    print("  (C) (LOO - anchor) RMS small + peak-level <~1 dB => the ROM INTERPOLATION is")
    print("      clean; the residual is solver+model, addressable separately.")


if __name__ == "__main__":
    main()
