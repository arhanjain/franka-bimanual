"""Direct-RPyC Franka driver for env-frame EE-pose control.

The action interface is an absolute EE *pose* (see envframe_franka.py). Two
actuation paths share this driver, selected by the robot's control mode:

- **Cartesian velocity** (``send_twist``): the robot tracks the pose target by
  streaming short, duration-bounded ``CartesianVelocityMotion`` commands.
  Streaming absolute ``CartesianMotion`` waypoints at teleop rate trips libfranka
  reflexes (velocity/acceleration discontinuity between the overwritten
  stop-at-rest trajectories), so we use the same velocity-tracking scheme the
  original bimanual driver uses for EE teleop.
- **Joint-velocity IK** (``send_jv`` + ``get_state_jac``): the workstation runs the
  sim DLS-IK loop at ~100 Hz, turning the held pose target into the sim DLS-IK
  joint step ``dq`` and streaming it as a joint *velocity* (``dq * ik_hz``) via
  ``JointVelocityMotion`` -- the same duration-windowed primitive the velocity
  paths use, so a converged (zero-velocity) command actively HOLDS the arm.
  Streaming joint *positions* (bare ``JointMotion``) instead does NOT work here:
  franky runs every move through its controller and ``JointMotion`` is a
  point-to-point Ruckig move with no validity window, so streaming it at 100 Hz
  trips "multiple motions" every tick, nothing holds the arm, and it sags. The
  redundancy resolution is still sim's exact ``Jᵀ(JJᵀ+λ²I)⁻¹``, so the joint-space
  trajectory matches sim (closer than the twist path's libfranka-internal IK).
  ``get_state_jac`` returns q + base-frame O_T_EE + base-frame zero Jacobian in
  one round-trip so each IK tick is two RPyC calls per arm (read, then write).

As in the original driver, all motion/state construction lives server-side so
each per-loop op is a single RPyC round-trip per arm, and data crosses the wire
as tuples of native floats (brine encodes those by value; lists become netrefs).
The EE pose crosses the wire as a flat 4x4 homogeneous transform (16 floats,
row-major) so there is no quaternion-order assumption at the franky boundary.
"""

import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import rpyc
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

NUM_JOINTS = 7

DEFAULT_REQUEST_TIMEOUT_S = 5.0
RPYC_TIMEOUT_S = 10

# The FR3 allows one FCI master at a time and tracks it server-side. If a prior
# session died uncleanly (link drop, killed process), the robot keeps holding
# that dead master and rejects new connects with "Connection timeout" until its
# own network timeout frees the lock -- a few to tens of seconds. Retry across
# that window so a quick restart no longer needs a manual ping/wait/rerun.
_CONNECT_RETRIES = 8
_CONNECT_RETRY_BACKOFF_S = 4.0
_CONNECT_RETRY_ERRORS = (
    "Connection timeout",
    "NetworkException",
    "libfranka: Connection",
)


def _short_exc(e: Exception) -> str:
    """Last non-empty line of an exception message.

    RPyC remote exceptions stringify to the whole server-side traceback (~20
    lines); for a transient retryable connect error we only want the final
    ``ErrorType: message`` line so the retry status stays one line.
    """
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
    return lines[-1] if lines else e.__class__.__name__

# Velocity command validity window. Each tick overwrites the previous motion;
# if a tick is dropped the motion decelerates after this window rather than
# running away.
VELOCITY_COMMAND_DURATION_MS = 100
# Conservative (velocity, accel, jerk) scale factors for first bring-up. Raise
# once axis directions and workspace limits are confirmed on hardware.
_EE_RELATIVE_DYNAMICS = (0.4, 0.25, 0.15)
_TORQUE_THRESHOLD = 100.0  # Nm
_FORCE_THRESHOLD = 200.0   # N
# Joint-impedance stiffness + joint-position (velocity, accel, jerk) dynamics
# used by the twist path's internal joint controller when no per-arm override is
# given (joint_ik mode passes sim-matching values from EnvFrameFrankaConfig).
_DEFAULT_JOINT_STIFFNESS = (350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0)
_DEFAULT_JP_RELATIVE_DYNAMICS = (0.4, 0.25, 0.15)

_RECOVERABLE_ERRORS = (
    "UDP receive: Timeout",
    "communication_constrains_violation",
    'current mode ("Reflex")',
    "type of motion cannot change",
    "motion aborted by reflex",
    # Streaming overlapping async motions at the teleop rate can desync after a
    # transient fault/dropped tick, leaving a motion registered so the next
    # move() is rejected forever. reset() (join_motion + recover) clears it.
    "Attempted to start multiple motions",
)

# (q, dq, O_T_EE, twist): q/dq are the 7-vector joint angles/velocities, O_T_EE is
# a 4x4 homogeneous transform and twist is the 6-vector EE velocity (linear,
# angular). O_T_EE and twist are in the arm's own base frame. dq is read for
# joint-space homing (the action interface itself is still EE-pose only).
KinematicSnapshot = tuple[NDArray, NDArray, NDArray, NDArray]

# (q, O_T_EE): one round-trip for the joint-velocity IK loop. q is the 7-vector
# joint angles and O_T_EE the 4x4 base-frame EE transform. The Jacobian is NOT
# fetched from franky (its zero_jacobian is broken here, see get_state_jac); the
# workstation computes it from q via franka_jacobian.zero_jacobian.
IKSnapshot = tuple[NDArray, NDArray]

_SERVER_HELPERS = f"""
import gc as _gc
import threading, numpy as _np
import franky as _fr
import net_franky.cb_robot as _cbm

if not _cbm.state_mutex.acquire(blocking=False):
    _cbm.state_mutex = threading.Lock()
    _cbm.state = None
else:
    _cbm.state_mutex.release()

_DUR = _fr.Duration({VELOCITY_COMMAND_DURATION_MS})
_EE_DYN = _fr.RelativeDynamicsFactor(*{_EE_RELATIVE_DYNAMICS!r})
_ZERO3 = _np.zeros(3)
_ZERO_J = _np.zeros({NUM_JOINTS})

# Handle to the franky.Robot this server process is holding, so a reconnect can
# tear it down before opening a new one. Without this the rpyc server (which
# outlives individual teleop runs) leaks the prior session's libfranka
# connection, and the FR3 -- one master only -- rejects the new connect.
_ACTIVE = {{}}

def _release_existing():
    old = _ACTIVE.pop("robot", None)
    if old is None:
        return
    try:
        old.join_motion()
    except Exception:
        pass
    try:
        old.recover_from_errors()
    except Exception:
        pass
    # Drop our reference and force the franky.Robot destructor to run, which
    # closes the FCI socket so the robot releases the master lock immediately.
    del old
    _gc.collect()

def init_robot(ip, stiffness, dynamics):
    # stiffness: 7-vector joint-impedance K_theta (N*m/rad). dynamics: (vel, accel,
    # jerk) robot-level scale applied to every motion -- the twist path also passes
    # _EE_DYN per-motion, while the joint-velocity path (send_jv) carries no
    # per-motion factor and so runs at exactly this robot-level scale.
    _release_existing()
    r = _cbm.CBRobot(ip)
    r.recover_from_errors()
    r.relative_dynamics_factor = _fr.RelativeDynamicsFactor(*tuple(dynamics))
    r.set_collision_behavior({_TORQUE_THRESHOLD}, {_FORCE_THRESHOLD})
    r.set_joint_impedance(list(stiffness))
    _ACTIVE["robot"] = r
    return r

def get_state(robot):
    with _cbm.state_mutex:
        s = _cbm.state
    s = s.robot_state if s is not None else robot.state
    return (
        tuple(float(x) for x in s.q),
        tuple(float(x) for x in s.dq),
        tuple(float(x) for x in _np.asarray(s.O_T_EE.matrix).ravel()),
        tuple(float(x) for x in s.O_dP_EE_c.linear) + tuple(float(x) for x in s.O_dP_EE_c.angular),
    )

def get_state_jac(robot):
    # One round-trip for an IK tick: joint angles + base-frame EE transform.
    # NOTE: we deliberately do NOT return franky's model.zero_jacobian -- it
    # returns all zeros on this net_franky build (verified on hardware: every
    # frame/state/overload -> norm 0). The workstation computes the geometric
    # Jacobian itself from q (see franka_jacobian.py), so only (q, O_T_EE) cross.
    s = robot.state
    return (
        tuple(float(x) for x in s.q),
        tuple(float(x) for x in _np.asarray(s.O_T_EE.matrix).ravel()),
    )

def send_twist(robot, twist):
    t = _np.asarray(twist, dtype=_np.float64)
    robot.move(_fr.CartesianVelocityMotion(_fr.Twist(t[:3], t[3:]), _DUR, _EE_DYN), asynchronous=True)

def send_jv(robot, vel):
    robot.move(_fr.JointVelocityMotion(_np.asarray(vel, dtype=_np.float64), _DUR), asynchronous=True)

def stop(robot):
    robot.move(_fr.CartesianVelocityMotion(_fr.Twist(_ZERO3, _ZERO3), _DUR, _EE_DYN), asynchronous=False)

def stop_jv(robot):
    robot.move(_fr.JointVelocityMotion(_ZERO_J, _DUR), asynchronous=False)

def reset(robot):
    # join_motion() ends the async control thread and surfaces/clears the stored
    # async exception, re-syncing franky with libfranka. recover_from_errors()
    # alone leaves franky thinking a motion is still active, so the next move()
    # raises "Attempted to start multiple motions" -- forever. Join first.
    try:
        robot.join_motion()
    except Exception:
        pass
    robot.recover_from_errors()

def release():
    # Destroy the server-side franky.Robot so the FR3 releases the FCI master lock
    # NOW, on disconnect, instead of leaking it until the next init_robot. Closing
    # only the rpyc connection leaves _ACTIVE["robot"] alive here and the robot
    # rejects the next connect until its own network timeout frees the lock.
    _release_existing()
"""


class RobotDriver:
    """One arm: one RPyC connection, one robot handle, one set of helpers.

    A classic RPyC connection is NOT thread-safe -- concurrent requests
    interleave on the wire and corrupt the protocol. In twist mode the lerobot
    loop is single-threaded so the wrapper executor's per-arm tasks never
    overlap on one connection. joint_ik mode adds a background 100 Hz IK thread
    that can race ``get_observation`` on the SAME arm's connection, so every RPC
    is serialized through ``_lock``.
    """

    def __init__(self, server_ip: str, robot_ip: str, port: int,
                 stiffness: tuple[float, ...] = _DEFAULT_JOINT_STIFFNESS,
                 dynamics: tuple[float, ...] = _DEFAULT_JP_RELATIVE_DYNAMICS):
        self._lock = threading.Lock()
        self._conn = rpyc.classic.connect(server_ip, port)
        self._conn._config["sync_request_timeout"] = RPYC_TIMEOUT_S
        self._conn.execute(_SERVER_HELPERS)
        ns = self._conn.namespace
        self._rpc_state = ns["get_state"]
        self._rpc_state_jac = ns["get_state_jac"]
        self._rpc_send = ns["send_twist"]
        self._rpc_send_jv = ns["send_jv"]
        self._rpc_stop = ns["stop"]
        self._rpc_stop_jv = ns["stop_jv"]
        self._rpc_reset = ns["reset"]
        self._rpc_release = ns["release"]
        self.robot = self._connect_robot(ns, robot_ip, stiffness, dynamics)

    def _connect_robot(self, ns, robot_ip, stiffness, dynamics):
        """Open the FCI connection, retrying through the FR3's stale-session release window.

        Progress is ONE self-updating status line on a TTY: a spinner that
        animates during the backoff with ``[attempt n/N]``. The status text is
        kept short and clamped to the terminal width so it never wraps (a wrapped
        line breaks the ``\\r`` overwrite and leaves leftover rows). The full
        error is shown only on the final failure. When stderr is not a TTY (logs
        redirected to a file), fall back to one warning per attempt instead.
        """
        last_exc = None
        tty = sys.stderr.isatty()
        spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spin_i = 0

        def status(text: str) -> None:
            # Clamp to terminal width so the line can never wrap (wrapping defeats \r).
            width = shutil.get_terminal_size((80, 24)).columns
            sys.stderr.write("\r\033[2K" + text[: max(1, width - 1)])
            sys.stderr.flush()

        for attempt in range(1, _CONNECT_RETRIES + 1):
            try:
                robot = ns["init_robot"](robot_ip, tuple(stiffness), tuple(dynamics))
                if tty:
                    status(f"✓ connected to {robot_ip} (attempt {attempt}/{_CONNECT_RETRIES})")
                    sys.stderr.write("\n")
                return robot
            except Exception as e:
                last_exc = e
                retryable = any(t in str(e) for t in _CONNECT_RETRY_ERRORS)
                if attempt >= _CONNECT_RETRIES or not retryable:
                    if tty:
                        status(f"✗ failed to connect to {robot_ip} after {attempt} attempt(s)")
                        sys.stderr.write("\n")
                    raise
                if tty:
                    # Animate the spinner across the backoff so it reads as "working",
                    # not frozen. Short text -> no wrap.
                    deadline = time.perf_counter() + _CONNECT_RETRY_BACKOFF_S
                    while time.perf_counter() < deadline:
                        status(f"{spin[spin_i % len(spin)]} connecting to {robot_ip}  "
                               f"[attempt {attempt}/{_CONNECT_RETRIES}, retrying]")
                        spin_i += 1
                        time.sleep(0.1)
                else:
                    logger.warning(
                        "connecting to %s: attempt %d/%d failed (%s); retrying in %.1fs "
                        "(robot likely still holding a prior FCI session)",
                        robot_ip, attempt, _CONNECT_RETRIES, _short_exc(e), _CONNECT_RETRY_BACKOFF_S,
                    )
                    time.sleep(_CONNECT_RETRY_BACKOFF_S)
        raise last_exc

    @property
    def is_alive(self) -> bool:
        return not self._conn.closed

    def get_kinematic_state(self) -> KinematicSnapshot:
        with self._lock:
            q, dq, mat, twist = self._rpc_state(self.robot)
        return np.array(q), np.array(dq), np.array(mat).reshape(4, 4), np.array(twist)

    def get_ik_state(self) -> IKSnapshot:
        """One round-trip: (q[7], O_T_EE[4x4]) for an IK tick (Jacobian computed client-side)."""
        with self._lock:
            q, mat = self._rpc_state_jac(self.robot)
        return np.array(q), np.array(mat).reshape(4, 4)

    def send_twist(self, twist: list[float]) -> None:
        # tuple() so brine encodes by value (lists go over as netrefs).
        self._send_via(self._rpc_send, tuple(twist), "send_twist")

    def send_joint_velocity(self, vel: list[float]) -> None:
        self._send_via(self._rpc_send_jv, tuple(vel), "send_joint_velocity")

    def _send_via(self, rpc, payload: tuple, label: str) -> None:
        # tuple() payload so brine encodes by value (lists go over as netrefs).
        try:
            with self._lock:
                rpc(self.robot, payload)
        except Exception as e:
            if any(t in str(e) for t in _RECOVERABLE_ERRORS):
                try:
                    with self._lock:
                        self._rpc_reset(self.robot)
                except Exception:
                    pass
            logger.warning("%s: %s", label, e)

    def stop(self) -> None:
        self._stop_via(self._rpc_stop, "stop")

    def stop_jv(self) -> None:
        self._stop_via(self._rpc_stop_jv, "stop_jv")

    def _stop_via(self, rpc, label: str) -> None:
        # Cleanup must never raise. The blocking stop move can hang past the rpyc
        # timeout (faulted/reflex state, tangled async motions) -> on any error
        # fall back to reset() (bounded: join_motion + recover) and warn. A
        # velocity stream decelerates on its own after its command window anyway.
        try:
            with self._lock:
                rpc(self.robot)
        except Exception as e:
            try:
                with self._lock:
                    self._rpc_reset(self.robot)
            except Exception:
                pass
            logger.warning("%s: %s", label, e)

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        # Release the server-side robot so the FCI master lock frees immediately;
        # otherwise the next connect must wait out the FR3's stale-session timeout.
        try:
            with self._lock:
                self._rpc_release()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


class MultiRobotWrapper:
    """Manager dispatching to per-arm RobotDriver instances in parallel."""

    def __init__(self):
        self.drivers: dict[str, RobotDriver] = {}
        self._pool = ThreadPoolExecutor(max_workers=4)

    def add_robot(self, name: str, server_ip: str, robot_ip: str, port: int,
                  stiffness: tuple[float, ...] = _DEFAULT_JOINT_STIFFNESS,
                  dynamics: tuple[float, ...] = _DEFAULT_JP_RELATIVE_DYNAMICS) -> None:
        if name in self.drivers:
            raise ValueError(f"Robot '{name}' already connected")
        self.drivers[name] = RobotDriver(server_ip, robot_ip, port, stiffness, dynamics)

    @property
    def num_alive(self) -> int:
        return sum(1 for d in self.drivers.values() if d.is_alive)

    def _gather(self, fn, names, timeout_s: float | None = None) -> dict[str, Any]:
        futs = [(n, self._pool.submit(fn, n)) for n in names]
        return {n: f.result(timeout=timeout_s) for n, f in futs}

    def current_kinematic_state(self, name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> KinematicSnapshot:
        return self.drivers[name].get_kinematic_state()

    def current_kinematic_state_batch(
        self, names: list[str], timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    ) -> dict[str, KinematicSnapshot]:
        return self._gather(lambda n: self.drivers[n].get_kinematic_state(), names, timeout_s)

    def current_ik_state_batch(
        self, names: list[str], timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    ) -> dict[str, IKSnapshot]:
        """{arm: (q[7], O_T_EE[4x4])} -- one read per arm for an IK tick."""
        return self._gather(lambda n: self.drivers[n].get_ik_state(), names, timeout_s)

    def move_twist_batch(self, twists: dict[str, list]) -> None:
        """twists: {arm: 6-vector [vx,vy,vz,wx,wy,wz]} in each arm's base frame."""
        self._gather(lambda n: self.drivers[n].send_twist(twists[n]), list(twists))

    def move_joint_velocity_batch(self, vels: dict[str, list]) -> None:
        """vels: {arm: 7-vector joint velocities [rad/s]}. Used by home() and the joint_ik loop."""
        self._gather(lambda n: self.drivers[n].send_joint_velocity(vels[n]), list(vels))

    def stop_all_motion(self) -> None:
        self._gather(lambda n: self.drivers[n].stop(), [n for n, d in self.drivers.items() if d.is_alive])

    def stop_all_joint_motion(self) -> None:
        self._gather(lambda n: self.drivers[n].stop_jv(), [n for n, d in self.drivers.items() if d.is_alive])

    def shutdown(self) -> None:
        try:
            self.stop_all_motion()
        except Exception:
            pass
        for d in self.drivers.values():
            d.shutdown()
        self.drivers.clear()
        self._pool.shutdown(wait=False)
