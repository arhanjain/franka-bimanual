"""Franka gripper driver with latest-wins non-blocking width commands.

The gripper runs on its own RPyC connection so width commands never share a
transport with arm motion. Calls to `move()` keep only the latest requested
width and the worker thread sends it once the previous move finishes.
"""

from __future__ import annotations

import threading

import rpyc

RPYC_TIMEOUT_S = 10


class FrankaGripper:
    GRIPPER_TRUE_MAX_MM = 80.0
    _MOVE_SPEED_M_S = 10.0
    _ASYNC_MOVE_SPEED_M_S = 0.20
    # Keep every meaningful width update so the latest command reaches the gripper.
    _TARGET_CHANGE_THRESH_MM = 0.0

    def __init__(self, name: str = "", server_ip: str = "", robot_ip: str = "", port: int = 0, do_print: bool = False):
        self.name = name
        self.do_print = do_print
        self._position_mm = self.GRIPPER_TRUE_MAX_MM
        self._last_sent_position_mm: float | None = None

        self._conn = rpyc.classic.connect(server_ip, port)
        self._conn._config["sync_request_timeout"] = RPYC_TIMEOUT_S
        self._conn.execute(
            """
import threading
import franky as _fr

class _QueuedGripper:
    def __init__(self, ip):
        self._gripper = _fr.Gripper(ip)
        self._cond = threading.Condition()
        self._pending_width_m = None
        self._current_future = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def home(self):
        self._gripper.homing()
        return True

    def submit(self, width_m):
        with self._cond:
            self._pending_width_m = float(width_m)
            if self._current_future is not None and hasattr(self._current_future, "cancel"):
                if self._current_future.cancel():
                    self._current_future = None
            self._cond.notify()

    def move_blocking(self, width_m, speed_m_s):
        return self._gripper.move(float(width_m), float(speed_m_s))

    def close(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

    def _run(self):
        while True:
            with self._cond:
                if self._stopped:
                    return
                if self._current_future is not None and hasattr(self._current_future, "done") and not self._current_future.done():
                    self._cond.wait(timeout=0.02)
                    continue
                self._current_future = None
                if self._pending_width_m is None:
                    self._cond.wait()
                    continue
                width_m = self._pending_width_m
                self._pending_width_m = None

            try:
                self._current_future = self._gripper.move_async(width_m, 1.00)
            except Exception:
                self._current_future = None

def init_gripper(ip):
    return _QueuedGripper(ip)

def home_gripper(controller):
    return controller.home()

def move_gripper(controller, width_m):
    controller.submit(width_m)

def move_gripper_blocking(controller, width_m, speed_m_s):
    return controller.move_blocking(width_m, speed_m_s)

def close_gripper(controller):
    controller.close()
"""
        )
        ns = self._conn.namespace
        self._controller = ns["init_gripper"](robot_ip)
        self._rpc_home = ns["home_gripper"]
        self._rpc_move = ns["move_gripper"]
        self._rpc_move_blocking = ns["move_gripper_blocking"]
        self._rpc_close = ns["close_gripper"]

    @staticmethod
    def _clamp_mm(position_mm: float) -> float:
        return float(max(0.0, min(FrankaGripper.GRIPPER_TRUE_MAX_MM, position_mm)))

    @property
    def position(self) -> float | None:
        return self._position_mm

    @property
    def gripper_state(self) -> int | None:
        return None

    def move(self, position_mm: float, blocking: bool = False) -> bool:
        target_mm = self._clamp_mm(position_mm)
        self._position_mm = target_mm
        if self._last_sent_position_mm is not None and abs(target_mm - self._last_sent_position_mm) < self._TARGET_CHANGE_THRESH_MM:
            return True

        self._last_sent_position_mm = target_mm
        if blocking:
            return bool(self._rpc_move_blocking(self._controller, target_mm / 1000.0, self._MOVE_SPEED_M_S))
        self._rpc_move(self._controller, target_mm / 1000.0)
        return True

    def home(self) -> bool:
        result = bool(self._rpc_home(self._controller))
        self._position_mm = self.GRIPPER_TRUE_MAX_MM
        self._last_sent_position_mm = self.GRIPPER_TRUE_MAX_MM
        return result

    def home_async(self) -> threading.Thread:
        thread = threading.Thread(target=self.home, daemon=True)
        thread.start()
        return thread

    def grip(self, force_n: float, blocking: bool = True):
        return self.move(0.0, blocking=blocking)

    def release(self, blocking: bool = True):
        return self.move(self.GRIPPER_TRUE_MAX_MM, blocking=blocking)

    def ack_fast_stop(self) -> bool:
        return True

    def set_verbose(self, verbose: bool = True) -> bool:
        return True

    def bye(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._rpc_close(self._controller)
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
