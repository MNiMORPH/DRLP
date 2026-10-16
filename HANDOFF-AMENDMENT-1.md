# Handoff amendment 1 — no network in DRLP

**16 October 2026, from Andy.** Supersedes the last bullet of §6 in
`HANDOFF.md`, which left this open. Written as an amendment because the build
session had already started.

## The decision

**DRLP is a 1D long profile and nothing else. Networks are FluvTree's job.**

- Do **not** build a `Network` class, a confluence rule, or any topology.
  GRLP has one; DRLP does not, and the asymmetry is deliberate.
- Do **not** add `networkx`. The core stays numpy-only, as §3 requires.
- §6's last bullet — "decide whether DRLP needs one… ask before building a
  second graph layer" — is now answered. Do not ask, do not build.

The single-channel sweep is a reverse loop over one array, from the outlet
upstream. That is the whole of it.

## What this implies for the API

FluvTree's sweep visits reaches outlet-first and solves each one against an
already-updated downstream elevation. For the eventual
`processes/stream_power.py` adapter to drive DRLP reach by reach, **the stepper
must accept the downstream boundary elevation as an argument** rather than
assuming it owns base level.

So shape the time step as something an adapter can call per reach:

    step(x, z, Q, K, m, n, U, dt, z_downstream) -> z

with the model object wrapping that for the standalone single-channel case,
where `z_downstream` is base level. Composable without DRLP knowing anything
about topology, and it costs nothing in the 1D case.

Everything else in `HANDOFF.md` stands.
