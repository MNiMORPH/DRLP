"""
A base-level drop, and the knickpoint it sends upstream.

This is the model's main use, and the thing that separates it from GRLP: the
disturbance travels rather than spreading. Run it and the incision front marches
up the profile at the celerity the governing equation predicts, while the
headwaters sit untouched for hundreds of thousands of years.

    python examples/base_level_drop.py

Needs nothing but the package. Pass --plot for a figure (needs matplotlib,
`pip install -e ".[figures]"`).
"""
import sys

import numpy as np

from drlp import LongProfile

YEAR = 31556926.0
KYR = 1000 * YEAR


def build():
    return (LongProfile()
            .set_x(dx=500.0, nx=60, x0=500.0)
            .set_A(k_xA=1.0, P_xA=7 / 4.0)
            .set_Q(R=1e-8)
            .set_K(10**-8.5, basis="Q")
            .set_mn(m=0.5, n=1.0)
            .set_uplift_rate(0.3e-3 / YEAR)
            .set_bl(z=0.0)
            .set_steady_state())


def main(plot=False):
    lp = build()
    z0 = lp.z.copy()
    drop = 200.0

    # celerity from the governing equation, not from the solver: at n = 1 the
    # equation is linear advection of z at C = K Q^m.
    C = lp.K * lp.driver**lp.m
    print(f"steady relief          {z0[0] - z0[-1]:8.1f} m")
    print(f"celerity K Q^m         {C.min():.3e} to {C.max():.3e} m/s")
    print(f"base level dropped     {drop:8.1f} m")
    print()
    print(f"{'t [kyr]':>9}  {'outlet dz':>10}  {'headwater dz':>13}  "
          f"{'front [km]':>11}")

    lp.set_bl(z=-drop)
    dt, snapshots = 1e11, (10, 30, 60, 120, 300)
    frames = {}
    for n_frames in snapshots:
        while lp.t < n_frames * dt:
            lp.evolve(nt=1, dt=dt)
        moved = np.abs(lp.z - z0)
        front = lp.x[np.argmax(moved > 1.0)] / 1000.0 if (moved > 1.0).any() \
            else float("nan")
        print(f"{lp.t / KYR:9.1f}  {lp.z[-1] - z0[-1]:10.2f}  "
              f"{lp.z[0] - z0[0]:13.3f}  {front:11.1f}")
        frames[n_frames] = lp.z.copy()

    print()
    print("The outlet takes the full drop. The headwaters have not moved. That")
    print("is advection: the signal has a position, and you can watch it move.")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lp.x / 1000.0, z0, "k--", lw=1, label="initial steady state")
        for n_frames, z in frames.items():
            ax.plot(lp.x / 1000.0, z, lw=1.5,
                    label=f"{n_frames * dt / KYR:.0f} kyr")
        ax.axhline(-drop, color="gray", lw=1, ls=":", label="base level")
        ax.set_xlabel("Downstream distance [km]")
        ax.set_ylabel("Elevation [m]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = "examples/base_level_drop.png"
        fig.savefig(out, dpi=150)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)
