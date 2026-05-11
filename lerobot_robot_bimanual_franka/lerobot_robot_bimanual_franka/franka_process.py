"""Thread-based Franka robot driver for the bimanual plugin.

Each arm is managed by a RobotDriver that holds a direct RPyC connection to
the remote franky server. MultiRobotWrapper is the manager facade.

Why no multiprocessing: net_franky.franky stores a single (IP, PORT) in a
module-level singleton and opens one RPyC connection at import time, making
it impossible to connect two arms in the same process via that API. We bypass
it by calling rpyc.classic.connect() directly — exactly what net_franky does
internally — once per arm in the same process.

Why no locking: each arm has its own connection; no two threads ever call into
the same connection concurrently. The ThreadPoolExecutor assigns one dedicated
worker per arm, and the control loop is strictly read→compute→write sequential.

Why server-side helpers: a plain RPyC attribute access on a remote proxy is
one network round-trip. Reading a full RobotState naively costs ~50 round-trips.
We instead execute a small packing function on the server that bundles all
fields into plain Python lists — one round-trip regardless of state size.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, cast

import numpy as np
import rpyc
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Duration (ms) attached to each streamed velocity command. Caller must re-issue faster than this.
VELOCITY_COMMAND_DURATION_MS = 250

# Recompute Jacobian only when joints move more than this (L-inf, rad) from the cached config.
_JACOBIAN_CACHE_Q_THRESHOLD = 0.50

JOINT_RELATIVE_DYNAMICS = (1.0, 0.25, 1.0)
TORQUE_THRESHOLD = 100.0  # Nm
FORCE_THRESHOLD = 200.0   # N
JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

EE_DELTA_RELATIVE_DYNAMICS = (0.4, 0.25, 0.15)

NUM_JOINTS = 7
EE_DELTA_DIMS = 6  # linear(3) + angular(3)

# (q, dq, jacobian, ee_pos, ee_rot_xyzw, ee_twist) snapshot from one robot.state read.
KinematicSnapshot = tuple[
    NDArray[np.float64],  # joint pos
    NDArray[np.float64],  # joint velocities
    NDArray[np.float64],  # jacobian
    NDArray[np.float64],  # ee pos
    NDArray[np.float64],  # ee rot
    NDArray[np.float64],  # ee twist (velocity)
]

DEFAULT_REQUEST_TIMEOUT_S = 5.0
SHUTDOWN_STOP_TIMEOUT_S = 2.0

_RECOVERABLE_ERRORS = (
    "UDP receive: Timeout",
    "communication_constrains_violation",
    'current mode ("Reflex")',
    "type of motion cannot change",
)

# Installed on the remote RPyC server at connect-time (conn.execute).
# All helper functions operate on server-local objects — no nested RPyC calls.
# Basic Python containers (list, tuple, float) are sent by value by RPyC,
# so each helper call is exactly ONE network round-trip.
#
# CBRobot.get_last_callback_data() has a latent bug: if state is None it
# raises AttributeError *after* acquiring state_mutex but *before* releasing
# it, permanently deadlocking the server-side mutex across reconnections.
# We recover from any such stuck mutex at connect-time and never call
# get_last_callback_data() directly — we access cb_robot.state ourselves
# using a `with` block that guarantees release even on exception.
_SERVER_HELPERS = """
import threading
import net_franky.cb_robot as _cbm

# Recover from a mutex left locked by a previous crashed session.
# If acquire(blocking=False) fails the mutex is stuck; replace it with a
# fresh one so this and future connections aren't permanently blocked.
if not _cbm.state_mutex.acquire(blocking=False):
    _cbm.state_mutex = threading.Lock()
    _cbm.state = None
else:
    _cbm.state_mutex.release()

def _pack_state(s):
    # float() converts numpy scalars to native Python floats so that RPyC's
    # brine encoder sends the whole tuple by value (one round-trip).
    # Without this, numpy.float64 elements are not brine-encodable and the
    # entire return becomes a netref, triggering dozens of extra RPyC calls.
    return (
        [float(x) for x in s.q],
        [float(x) for x in s.dq],
        [float(x) for x in s.O_T_EE.translation],
        [float(x) for x in s.O_T_EE.quaternion],
        [float(x) for x in s.O_dP_EE_c.linear] + [float(x) for x in s.O_dP_EE_c.angular],
    )

def _get_robot_state_fast(robot):
    with _cbm.state_mutex:
        s = _cbm.state
    if s is not None:
        return _pack_state(s.robot_state)
    return _pack_state(robot.state)

def _get_jacobian_fast(robot, frame):
    import numpy as _np
    # Flatten to 1-D list of native floats so brine sends it by value.
    j = _np.asarray(robot.model.zero_jacobian(frame, robot.state))
    return [float(x) for x in j.flat]
"""


def _validate_vector(name: str, values, expected_len: int) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple of length {expected_len}, got {type(values).__name__}")
    if len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(values)}")
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must contain only numeric values") from e


class RobotDriver:
    """Owns one RPyC connection and Franka robot handle for a single arm.

    Designed for single-threaded use per instance (the MultiRobotWrapper
    executor ensures this). No locking required.
    """

    def __init__(
        self,
        server_ip: str,
        robot_ip: str,
        port: int,
        use_ee_delta: bool = False,
    ):
        self.use_ee_delta = use_ee_delta
        self._motion_started = False
        self._cached_jacobian: NDArray[np.float64] | None = None
        self._cached_jacobian_q: NDArray[np.float64] | None = None

        # Connect directly — bypasses net_franky's module-level (IP, PORT) singleton
        # so multiple arms can coexist in the same Python process.
        self._conn = rpyc.classic.connect(server_ip, port)
        self._conn._config["sync_request_timeout"] = 10
        _franky = self._conn.modules["franky"]
        _cb = self._conn.modules["net_franky.cb_robot"]

        self._JointVelocityMotion = _franky.JointVelocityMotion
        self._CartesianVelocityMotion = _franky.CartesianVelocityMotion
        self._Duration = _franky.Duration
        self._Frame = _franky.Frame
        self._Twist = _franky.Twist
        self._RelativeDynamicsFactor = _franky.RelativeDynamicsFactor

        self.robot = _cb.CBRobot(robot_ip)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = self._RelativeDynamicsFactor(*JOINT_RELATIVE_DYNAMICS)
        self.robot.set_collision_behavior(TORQUE_THRESHOLD, FORCE_THRESHOLD)
        self.robot.set_joint_impedance(JOINT_STIFFNESS)

        self._ee_dynamics = self._RelativeDynamicsFactor(*EE_DELTA_RELATIVE_DYNAMICS)
        self._zero_lin = np.zeros(3, dtype=np.float64)
        self._zero_joint = np.zeros(NUM_JOINTS, dtype=np.float64)

        # Install server-side helpers in this connection's remote __main__.
        # Replaces ~50 per-field RPyC round-trips with a single batched call.
        self._conn.execute(_SERVER_HELPERS)
        self._rpc_get_state = self._conn.namespace["_get_robot_state_fast"]
        self._rpc_get_jacobian = self._conn.namespace["_get_jacobian_fast"]

    @property
    def is_alive(self) -> bool:
        return not self._conn.closed

    def _make_prime_motion(self):
        if self.use_ee_delta:
            return self._CartesianVelocityMotion(
                self._Twist(cast(Any, self._zero_lin), cast(Any, self._zero_lin)),
                self._Duration(VELOCITY_COMMAND_DURATION_MS),
                self._ee_dynamics,
            )
        return self._JointVelocityMotion(cast(Any, self._zero_joint), self._Duration(VELOCITY_COMMAND_DURATION_MS))

    def get_kinematic_state(self) -> KinematicSnapshot:
        # One RPyC call — server packs the entire state into plain Python lists.
        q_l, dq_l, ee_pos_l, ee_rot_l, ee_vel_l = self._rpc_get_state(self.robot)
        q = np.array(q_l, dtype=np.float64)
        dq = np.array(dq_l, dtype=np.float64)
        ee_pos = np.array(ee_pos_l, dtype=np.float64)
        ee_rot = np.array(ee_rot_l, dtype=np.float64)
        ee_vel = np.array(ee_vel_l, dtype=np.float64)

        if (
            self._cached_jacobian is None
            or self._cached_jacobian_q is None
            or float(np.max(np.abs(q - self._cached_jacobian_q))) > _JACOBIAN_CACHE_Q_THRESHOLD
        ):
            j_list = self._rpc_get_jacobian(self.robot, self._Frame.EndEffector)
            self._cached_jacobian = np.array(j_list, dtype=np.float64).reshape(6, 7)
            self._cached_jacobian_q = q.copy()

        return (q, dq, self._cached_jacobian, ee_pos, ee_rot, ee_vel)

    def _move(self, motion, asynchronous: bool) -> None:
        try:
            self.robot.move(motion, asynchronous=asynchronous)
            self._motion_started = True
        except Exception as e:
            if any(tok in str(e) for tok in _RECOVERABLE_ERRORS):
                try:
                    self.robot.recover_from_errors()
                except Exception:
                    pass
            logger.warning("move error: %s", e)

    def move_joint_velocity(self, velocity, asynchronous: bool = False) -> None:
        vel = np.asarray(_validate_vector("move_joint_velocity", velocity, NUM_JOINTS), dtype=np.float64)
        self._move(
            self._JointVelocityMotion(cast(Any, vel), self._Duration(VELOCITY_COMMAND_DURATION_MS)),
            asynchronous,
        )

    def move_ee_delta(self, delta, asynchronous: bool = False) -> None:
        d = _validate_vector("move_ee_delta position", delta, EE_DELTA_DIMS)
        self._move(
            self._CartesianVelocityMotion(
                self._Twist(
                    cast(Any, np.asarray(d[:3], dtype=np.float64)),
                    cast(Any, np.asarray(d[3:], dtype=np.float64)),
                ),
                self._Duration(VELOCITY_COMMAND_DURATION_MS),
                self._ee_dynamics,
            ),
            asynchronous,
        )

    def stop_motion(self) -> None:
        self.robot.move(self._make_prime_motion(), asynchronous=False)

    def shutdown(self) -> None:
        try:
            self.stop_motion()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


class MultiRobotWrapper:
    """Manager that dispatches calls to per-arm RobotDriver instances in parallel."""

    def __init__(self):
        self._drivers: dict[str, RobotDriver] = {}
        self._executor: ThreadPoolExecutor | None = None

    def _parallel(
        self, fn_by_name: dict[str, Any], timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Submit callables in parallel, wait for all results."""
        assert self._executor is not None, "No robots added yet"
        futures = {name: self._executor.submit(fn) for name, fn in fn_by_name.items()}
        return {name: fut.result(timeout=timeout_s) for name, fut in futures.items()}

    def add_robot(
        self,
        name: str,
        server_ip: str,
        robot_ip: str,
        port: int,
        use_ee_delta: bool = False,
    ) -> None:
        if name in self._drivers:
            raise ValueError(f"Robot '{name}' is already connected")
        self._drivers[name] = RobotDriver(server_ip, robot_ip, port, use_ee_delta)
        self._executor = ThreadPoolExecutor(max_workers=len(self._drivers))

    @property
    def num_processes(self) -> int:
        """Alias kept for BimanualFranka.is_connected compatibility."""
        return sum(1 for d in self._drivers.values() if d.is_alive)

    def current_kinematic_state(
        self, robot_name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    ) -> KinematicSnapshot:
        return self._drivers[robot_name].get_kinematic_state()

    def current_kinematic_state_batch(
        self,
        robot_names: list[str],
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> dict[str, KinematicSnapshot]:
        return self._parallel(
            {name: self._drivers[name].get_kinematic_state for name in robot_names},
            timeout_s=timeout_s,
        )

    def move_joint_velocity_batch(
        self, velocities_by_robot: dict[str, list], asynchronous: bool = False
    ) -> dict[str, Any]:
        return self._parallel({
            name: partial(driver.move_joint_velocity, vel, asynchronous)
            for name, (driver, vel) in {
                n: (self._drivers[n], velocities_by_robot[n]) for n in velocities_by_robot
            }.items()
        })

    def move_ee_delta_batch(
        self, positions_by_robot: dict[str, list], asynchronous: bool = False
    ) -> dict[str, Any]:
        return self._parallel({
            name: partial(driver.move_ee_delta, pos, asynchronous)
            for name, (driver, pos) in {
                n: (self._drivers[n], positions_by_robot[n]) for n in positions_by_robot
            }.items()
        })

    def stop_all_motion(self, timeout_s: float = SHUTDOWN_STOP_TIMEOUT_S) -> dict[str, Any]:
        return self._parallel(
            {name: driver.stop_motion for name, driver in self._drivers.items() if driver.is_alive},
            timeout_s=timeout_s,
        )

    def shutdown(self) -> None:
        try:
            self.stop_all_motion()
        except Exception:
            pass
        for driver in self._drivers.values():
            driver.shutdown()
        self._drivers.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
