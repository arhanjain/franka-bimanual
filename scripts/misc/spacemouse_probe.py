#!/usr/bin/env python
"""Diagnose SpaceMouse stickiness by printing raw axis values.

Usage:
    python scripts/spacemouse_probe.py [hidraw_path]   # default /dev/hidraw2

What to look for:
- Push an axis (e.g. left), then RELEASE. Watch the values.
- If they settle to small-but-nonzero at rest (e.g. |v| ~ 0.01-0.03), the
  integrator in spacemouse.py is creeping because it has no deadband.
  The printed "deadband suggestion" is the smallest threshold that would
  have zeroed every resting sample seen so far.
- If a value stays PINNED at full deflection after release (e.g. -0.8) and
  never returns toward 0, the device isn't sending a center report and a
  deadband won't help -- tell Claude and we gate on fresh reports instead.

Reads are drained the same way spacemouse.py does it (one report per read(),
loop until the timestamp stops advancing) so this mirrors the real path.
"""

import sys
import time

import pyspacemouse

HIDRAW = sys.argv[1] if len(sys.argv) > 1 else "/dev/hidraw2"
HZ = 50.0
_MAX_DRAIN = 64
AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def drain(dev):
    """Latest state after draining the HID backlog (mirrors spacemouse.py)."""
    state = dev.read()
    last_t = state.t
    for _ in range(_MAX_DRAIN):
        state = dev.read()
        if state.t == last_t:
            break
        last_t = state.t
    return state


def main() -> int:
    print(f"Opening {HIDRAW} ... (Ctrl-C to stop)")
    dev = pyspacemouse.open_by_path(HIDRAW, nonblocking=True)
    print(f"connected: {dev.describe_connection()}\n")

    # Largest resting residual seen while all axes are 'near zero', used to
    # recommend a deadband. We treat a sample as 'resting' if every axis is
    # under REST_GUESS; this is just for the suggestion, not the readout.
    REST_GUESS = 0.15
    max_resting = 0.0

    period = 1.0 / HZ
    try:
        while True:
            t0 = time.perf_counter()
            s = drain(dev)
            vals = {a: float(getattr(s, a)) for a in AXES}
            btns = list(s.buttons)

            biggest = max(abs(v) for v in vals.values())
            if biggest < REST_GUESS:
                max_resting = max(max_resting, biggest)

            row = "  ".join(f"{a}={vals[a]:+.3f}" for a in AXES)
            flag = "" if biggest < 1e-6 else ("  <-- NONZERO" if biggest < REST_GUESS else "")
            print(
                f"{row}  btn={btns}  | max|rest|={max_resting:.3f} "
                f"deadband>={max_resting + 0.005:.3f}{flag}",
                flush=True,
            )

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        dev.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
