"""Schunk WSG (GCL) gripper driver — responsive position streaming.

The design is built around the single goal of making the gripper react
the instant a new target is commanded.

Architecture (one socket, two threads):

* Reader thread  blocks on ``recv``, parses every line, updates the
  cached position / gripper state, and fires waiters for blocking
  commands (HOME, GRIP, …) when their expected response token shows
  up.

* Sender thread sleeps on a ``Condition`` and only wakes when there
  is something to send. It services one-shot commands FIFO, dispatches
  ``MOVE`` only when the requested target actually changes, and sends
  ``STOP`` before changing a target while the firmware is positioning.
  This preserves streamed reversals without submitting overlapping
  motion plans, which WSG firmware rejects with ``ERR MOVE 04``.

* ``move()`` hot path clamps the value, stores it as the latest target,
  and notifies the sender. Repeat calls with the same (or near-same)
  target are normally a no-op. A meaningful update during a move emits
  one ``STOP``. The latest target is sent after both the STOP response and
  the active MOVE's completion response. Command timeouts trigger bounded
  position and system-flag polling before motion can resume.

Public API:
    move(position, blocking=False)   stream a target position (mm)
    position                         last cached gap (mm), never blocks
    gripper_state                    last cached GRIPSTATE int
    grip / release / home            blocking one-shot commands
    ack_fast_stop / set_verbose / bye / close
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Waiter:
    """Pending blocking command waiting on a response token."""

    expected: bytes
    event: threading.Event = field(default_factory=threading.Event)
    succeeded: bool = True


class WSG:
    # Motion / range tuning.
    GRIPPER_TRUE_MAX_MM = 110.0
    MOVE_SPEED_MM_S = 420.0
    GRIPPER_MIN_MM = 10.0
    GRIPPER_MAX_MM = 100.0
    DEFAULT_TIMEOUT_S = 10.0

    # Rate caps.  ``_MIN_MOVE_INTERVAL_S`` keeps overlapping motion plans
    # from piling up at the gripper, ``_TARGET_CHANGE_THRESH_MM`` absorbs
    # noisy teleop input without an actual dead-zone.
    _MIN_MOVE_INTERVAL_S = 0.1       # ~10 Hz max MOVE rate
    _TARGET_CHANGE_THRESH_MM = 5.0
    _OPEN_POSITION_TOLERANCE_MM = 5.0
    _OPEN_RETRY_DELAY_S = 1.0
    _OPEN_POSITION_MAX_AGE_S = 0.5
    _POS_POLL_INTERVAL_S = 0.050       # ~20 Hz POS? poll
    _RECOVERY_FLAGS_POLL_INTERVAL_S = 0.100
    _RECOVERY_QUERY_RETRY_S = 0.500
    _STOP_RESPONSE_TIMEOUT_S = 1.0
    _RECOVERY_TIMEOUT_S = 4.0
    _MAX_RECOVERY_MOVE_RETRIES = 2
    _SOCK_RECV_TIMEOUT_S = 0.5
    _RECV_BUF_SIZE = 4096
    _RELEASE_PULL_BACK_MM = 10.0
    _CLOSE_JOIN_TIMEOUT_S = 1.0

    def __init__(
        self,
        name: str = "",
        TCP_IP: str = "192.168.1.20",
        TCP_PORT: int = 1000,
        do_print: bool = False,
        event_callback=None,
        poll_position: bool = True,
    ):
        self.name = name
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT
        self.do_print = do_print
        # Optional, fast hook used by calibration recorders. No recording or
        # motion behavior changes when this is omitted.
        self._event_callback = event_callback
        self._poll_position = poll_position
        self._motion_suspended = False
        self._motion_faulted = False

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.DEFAULT_TIMEOUT_S)
        try:
            self._sock.connect((TCP_IP, TCP_PORT))
        except BaseException:
            self._sock.close()
            raise
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(self._SOCK_RECV_TIMEOUT_S)
        self._send_lock = threading.Lock()  # serialises socket writes

        # Reader-owned cached state.
        self._state_lock = threading.Lock()
        self._position_mm: float | None = None
        self._gripper_state: int | None = None
        self._last_position_monotonic_s = 0.0
        self._last_rx_monotonic_s = 0.0
        self._system_flags: tuple[bool, ...] | None = None
        self._last_system_flags_monotonic_s = 0.0
        self._io_error: str | None = None

        # All sender state — the streaming target, queued one-shots,
        # and pending waiters — live behind ``_cond``'s lock.
        self._cond = threading.Condition()
        self._target_mm: float | None = None
        self._last_sent_target_mm: float | None = None
        self._force_move_pending = False
        self._last_move_send_t: float = 0.0
        # The firmware accepts only one positioning command at a time. These
        # flags track MOVE / STOP protocol state, not physical telemetry.
        self._move_active = False
        self._stop_pending = False
        self._stop_acknowledged = False
        self._stop_sent_s = 0.0
        self._recovery_started_s: float | None = None
        self._recovery_reason: str | None = None
        self._recovery_pos_query_sent_s = 0.0
        self._recovery_flags_query_sent_s = 0.0
        self._recovery_retry_target_mm: float | None = None
        self._recovery_retry_position_mm: float | None = None
        self._recovery_retry_count = 0
        self._open_retry_count = 0
        self._cmd_queue: deque[tuple[bytes, _Waiter | None]] = deque()
        self._waiters: deque[_Waiter] = deque()

        self._closed = threading.Event()

        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"wsg-rd-{name}"
        )
        self._sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True, name=f"wsg-tx-{name}"
        )
        self._reader_thread.start()
        self._sender_thread.start()

        # Clear any latched FAST STOP from a previous session.
        self.ack_fast_stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def position(self) -> float | None:
        """Last known finger gap in mm; never blocks."""
        with self._state_lock:
            return self._position_mm

    @property
    def gripper_state(self) -> int | None:
        """Last known GRIPSTATE integer; never blocks."""
        with self._state_lock:
            return self._gripper_state

    @property
    def position_age_s(self) -> float:
        """Seconds since the last parsed POS response; infinity before the first."""
        with self._state_lock:
            stamp = self._last_position_monotonic_s
        return float("inf") if stamp <= 0.0 else time.monotonic() - stamp

    @property
    def health_error(self) -> str | None:
        """Latched transport, thread, or unrecovered motion error."""
        with self._state_lock:
            io_error = self._io_error
        if io_error:
            return io_error
        if self._closed.is_set():
            return "driver is closed"
        if not self._reader_thread.is_alive():
            return "reader thread stopped"
        if not self._sender_thread.is_alive():
            return "sender thread stopped"
        return None

    def move(self, position_mm: float, speed: float = 1.0, blocking: bool = False,
             *, force: bool = False) -> bool:
        """Stream a new target position (mm).

        Non-blocking by default: the latest target is published to the
        sender thread which wakes immediately and emits ``MOVE`` on the
        wire. Repeat calls with the same target are coalesced unless a
        full-open move has not reached its target after the retry delay.
        ``force=True`` explicitly sends small or repeated targets through
        the same rate-limited dispatcher, for precise pre-positioning or
        a deliberate retry after an unsuccessful move.

        Targets are clamped to GRIPPER_MIN_MM..GRIPPER_MAX_MM. The legacy
        speed argument is retained for compatibility; MOVE_SPEED_MM_S is used.
        """
        target = self._clip_target(position_mm)
        with self._cond:
            if self._motion_faulted:
                return False
            self._motion_suspended = False
            self._target_mm = target
            self._force_move_pending = self._force_move_pending or force
            self._cond.notify()
        self._emit_event("request", target_mm=target, requested_mm=float(position_mm), force=force)

        if blocking:
            return self._await_command(self._move_cmd(target), b"FIN MOVE")
        return True

    def home(self) -> bool:
        """Reference the gripper fingers (blocking)."""
        return self._await_command(b"HOME()\n", b"FIN HOME")

    def home_async(self) -> threading.Thread:
        t = threading.Thread(target=self.home, daemon=True)
        t.start()
        return t

    def grip(self, force_n: float, blocking: bool = True):
        """Grasp with ``force_n`` Newtons.  Blocks by default."""
        cmd = f"GRIP({force_n})\n".encode()
        if blocking:
            return self._await_command(cmd, b"FIN GRIP")
        return self._fire_and_forget(cmd, b"FIN GRIP")

    def release(self, blocking: bool = True):
        """Release a previously gripped part.  Blocks by default."""
        cmd = f"RELEASE({self._RELEASE_PULL_BACK_MM})\n".encode()
        if blocking:
            return self._await_command(cmd, b"FIN RELEASE")
        return self._fire_and_forget(cmd, b"FIN RELEASE")

    def ack_fast_stop(self) -> bool:
        return self._await_command(b"FSACK()\n", b"ACK FSACK")

    def set_verbose(self, verbose: bool = True) -> bool:
        msg = f"VERBOSE={1 if verbose else 0}\n".encode()
        return self._await_command(msg, msg.rstrip(b"\n"))

    def query(self, parameter: str) -> bool:
        """Query a value; an optional event callback receives its raw reply."""
        allowed = {"DEVTYPE", "VERSION", "SN", "TEMP", "SYSFLAGS", "GRIPSTATE", "POS", "SPEED", "FORCE"}
        if parameter not in allowed:
            raise ValueError(f"Unsupported query {parameter!r}")
        return self._await_command(f"{parameter}?\n".encode(), f"{parameter}=".encode())

    def configure_telemetry(self, position_interval_ms: int = 10) -> bool:
        """Request fresh GCL telemetry; keep the normal MOVE dispatcher."""
        if position_interval_ms < 10:
            raise ValueError("WSG auto-send intervals must be at least 10 ms")
        for name, interval in (("POS", position_interval_ms), ("SPEED", position_interval_ms),
                               ("FORCE", 50), ("GRIPSTATE", 50), ("SYSFLAGS", 50)):
            if not self._await_command(f'AUTOSEND("{name}",{interval})\n'.encode(), b"ACK AUTOSEND"):
                return False
        return True

    def suspend_motion(self) -> None:
        """Discard pending moves immediately, without waiting for device I/O."""
        with self._cond:
            self._motion_suspended = True
            self._target_mm = None
            self._last_sent_target_mm = None
            self._force_move_pending = False
            self._open_retry_count = 0
            self._cmd_queue = deque((cmd, waiter) for cmd, waiter in self._cmd_queue
                                    if not cmd.startswith(b"MOVE("))

    def stop(self) -> bool:
        """Discard pending streaming motion and stop before disconnecting."""
        self.suspend_motion()
        return self._await_command(b"STOP()\n", b"ACK STOP", timeout_s=2.0)

    def fast_stop(self) -> bool:
        """Stop after a device error; leave the drive disabled until FSACK."""
        self.suspend_motion()
        return self._await_command(b"FASTSTOP()\n", b"ACK FASTSTOP", timeout_s=2.0)

    def _emit_event(self, kind, **fields):
        if getattr(self, "_event_callback", None) is not None:
            try:
                self._event_callback({"kind": kind, "monotonic_s": time.monotonic(), **fields})
            except Exception as exc:
                with self._state_lock:
                    self._io_error = f"recording callback failed: {exc}"

    def bye(self) -> None:
        """Announce disconnect so the gripper does not raise FAST STOP."""
        try:
            self._send_raw(b"BYE()\n")
        except Exception:
            pass

    def close(self) -> None:
        if self._closed.is_set():
            return
        # Send BYE before tearing the threads down so the gripper does
        # not latch a FAST STOP on abrupt disconnect.
        try:
            self.bye()
        except Exception:
            pass
        self._closed.set()
        with self._cond:
            self._cond.notify_all()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        self._reader_thread.join(timeout=self._CLOSE_JOIN_TIMEOUT_S)
        self._sender_thread.join(timeout=self._CLOSE_JOIN_TIMEOUT_S)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        buf = b""
        while not self._closed.is_set():
            try:
                chunk = self._sock.recv(self._RECV_BUF_SIZE)
            except socket.timeout:
                continue
            except OSError as exc:
                with self._state_lock:
                    self._io_error = f"receive failed: {exc}"
                break
            if not chunk:
                with self._state_lock:
                    self._io_error = "peer closed the connection"
                break
            with self._state_lock:
                self._last_rx_monotonic_s = time.monotonic()
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._handle_line(line.strip())

    def _handle_line(self, line: bytes) -> None:
        if not line:
            return
        text = line.decode("utf-8", errors="replace")
        self._emit_event("rx", text=text)
        # GCL automatic telemetry has an @ prefix; query replies do not.
        text = text.removeprefix("@")
        now = time.monotonic()
        position_updated = False
        flags_updated = False
        if text.startswith("POS="):
            try:
                position = float(text.split("=", 1)[1])
            except ValueError:
                pass
            else:
                with self._state_lock:
                    self._position_mm = position
                    self._last_position_monotonic_s = now
                position_updated = True
        elif text.startswith("SYSFLAGS=["):
            try:
                values = tuple(int(value) for value in text.split("=", 1)[1].strip("[]").split(","))
                if len(values) != 32 or any(value not in (0, 1) for value in values):
                    raise ValueError("invalid system flags")
            except ValueError:
                pass
            else:
                with self._state_lock:
                    self._system_flags = tuple(bool(value) for value in values)
                    self._last_system_flags_monotonic_s = now
                flags_updated = True
        elif text.startswith("GRIPSTATE="):
            try:
                with self._state_lock:
                    self._gripper_state = int(text.split("=", 1)[1])
            except ValueError:
                pass
        elif text.startswith("ERR") and self.do_print:
            print(f"{self.name}: [WSG] {text}")

        recovery_started = False
        with self._cond:
            if position_updated:
                self._recovery_pos_query_sent_s = 0.0
            if flags_updated:
                self._recovery_flags_query_sent_s = 0.0
            # A STOP can acknowledge before or after its interrupted MOVE
            # reports completion. Wait for both responses before the next MOVE.
            move_terminal = text == "FIN MOVE" or text.startswith("ERR MOVE ")
            if move_terminal:
                self._move_active = False
                if self._stop_acknowledged:
                    self._stop_pending = False
                    self._stop_acknowledged = False
                if text == "FIN MOVE" and self._recovery_started_s is None:
                    self._reset_recovery_retry_locked()
            if text == "ACK STOP":
                self._stop_acknowledged = True
                if not self._move_active:
                    self._stop_pending = False
                    self._stop_acknowledged = False
            elif text.startswith("ERR STOP "):
                self._stop_pending = False
                self._stop_acknowledged = False

            if text.startswith(("ERR STOP 07", "ERR MOVE 07")):
                recovery_started = self._start_recovery_locked(text, now)
            elif text.startswith("ERR STOP ") or (
                text.startswith("ERR MOVE ") and not text.startswith("ERR MOVE 19")
            ):
                self._fault_motion_locked(f"WSG motion command failed: {text}")
            self._cond.notify_all()

        if recovery_started:
            self._emit_event("recovery", state="start", reason=text)

        # Match the oldest waiter whose expected token appears in this
        # line.  One match per line; subsequent waiters keep waiting.
        with self._cond:
            for waiter in self._waiters:
                command = waiter.expected.decode().split()[-1].split("=")[0]
                if text.startswith(f"ERR {command} "):
                    waiter.succeeded = False
                    waiter.event.set()
                    self._waiters.remove(waiter)
                    return
                if waiter.expected in line:
                    waiter.event.set()
                    self._waiters.remove(waiter)
                    return

    def _start_recovery_locked(self, reason: str, now: float) -> bool:
        """Enter one reconciliation window for related STOP/MOVE timeouts."""
        if self._motion_faulted or self._recovery_started_s is not None:
            return False
        self._recovery_started_s = now
        self._recovery_reason = reason
        self._recovery_pos_query_sent_s = 0.0
        self._recovery_flags_query_sent_s = 0.0
        self._open_retry_count = 0
        self._discard_queued_motion_locked()
        return True

    def _discard_queued_motion_locked(self) -> None:
        kept = deque()
        for cmd, waiter in self._cmd_queue:
            if cmd.startswith((b"MOVE(", b"HOME(", b"GRIP(", b"RELEASE(")):
                if waiter is not None:
                    waiter.succeeded = False
                    waiter.event.set()
            else:
                kept.append((cmd, waiter))
        self._cmd_queue = kept

    def _reset_recovery_retry_locked(self) -> None:
        self._recovery_retry_target_mm = None
        self._recovery_retry_position_mm = None
        self._recovery_retry_count = 0

    def _fault_motion_locked(self, reason: str) -> None:
        """Latch an unrecoverable motion error; caller holds ``_cond``."""
        self._motion_faulted = True
        self._motion_suspended = True
        self._recovery_started_s = None
        self._target_mm = None
        self._last_sent_target_mm = None
        self._force_move_pending = False
        self._open_retry_count = 0
        self._reset_recovery_retry_locked()
        self._discard_queued_motion_locked()
        with self._state_lock:
            if self._io_error is None:
                self._io_error = reason

    # ------------------------------------------------------------------
    # Sender thread
    # ------------------------------------------------------------------

    def _sender_loop(self) -> None:
        last_pos_poll_t = 0.0
        last_flags_poll_t = 0.0
        while not self._closed.is_set():
            cmd_to_send: bytes | None = None
            recovery_event: tuple[str, str] | None = None
            with self._state_lock:
                position_mm = self._position_mm
                position_stamp = self._last_position_monotonic_s
                flags = self._system_flags
                flags_stamp = self._last_system_flags_monotonic_s
            with self._cond:
                if self._closed.is_set():
                    break
                now = time.monotonic()
                if (
                    self._stop_pending and self._recovery_started_s is None
                    and now - self._stop_sent_s >= self._STOP_RESPONSE_TIMEOUT_S
                ):
                    reason = "timeout waiting for STOP and MOVE responses"
                    if self._start_recovery_locked(reason, now):
                        recovery_event = ("start", reason)

                if self._recovery_started_s is not None:
                    started = self._recovery_started_s
                    fault_flags = (2, 3, 12, 14, 15, 16, 17)
                    if flags_stamp > started and flags is not None and any(
                        flags[index] for index in fault_flags
                    ):
                        active = [index for index in fault_flags if flags[index]]
                        reason = f"WSG reported blocked axis or hardware fault flags {active} during recovery"
                        self._fault_motion_locked(reason)
                        recovery_event = ("failed", reason)
                    elif now - started >= self._RECOVERY_TIMEOUT_S:
                        reason = f"WSG recovery timed out after {self._recovery_reason}"
                        self._fault_motion_locked(reason)
                        recovery_event = ("failed", reason)
                    elif (
                        not self._move_active and not self._stop_pending
                        and position_mm is not None and position_stamp > started
                        and flags is not None and flags_stamp > started
                        and not flags[1]
                    ):
                        target = self._target_mm
                        if target is not None and abs(position_mm - target) < self._TARGET_CHANGE_THRESH_MM:
                            # The current policy target is already satisfied.
                            self._last_sent_target_mm = target
                            self._force_move_pending = False
                            self._reset_recovery_retry_locked()
                        elif target is not None and self._last_sent_target_mm == target:
                            # Retry a short MOVE only while position feedback
                            # shows progress, with a fixed attempt limit.
                            if self._recovery_retry_target_mm == target:
                                previous = self._recovery_retry_position_mm
                                progress = abs(previous - target) - abs(position_mm - target)
                                if progress < self._TARGET_CHANGE_THRESH_MM:
                                    reason = f"WSG MOVE to {target:.1f} mm timed out twice without progress"
                                elif self._recovery_retry_count >= self._MAX_RECOVERY_MOVE_RETRIES:
                                    reason = f"WSG MOVE to {target:.1f} mm exceeded recovery retry limit"
                                else:
                                    reason = None
                            else:
                                self._recovery_retry_count = 0
                                reason = None
                            if reason is not None:
                                self._fault_motion_locked(reason)
                                recovery_event = ("failed", reason)
                            else:
                                self._force_move_pending = True
                                self._recovery_retry_target_mm = target
                                self._recovery_retry_position_mm = position_mm
                                self._recovery_retry_count += 1
                        else:
                            self._reset_recovery_retry_locked()
                        if not self._motion_faulted:
                            self._recovery_started_s = None
                            self._recovery_reason = None
                            recovery_event = ("ready", f"position={position_mm:.2f} mm")

                if self._recovery_started_s is not None or self._motion_faulted:
                    self._discard_queued_motion_locked()
                if self._cmd_queue:
                    cmd_bytes, waiter = self._cmd_queue.popleft()
                    if waiter is not None:
                        # Register before send so the response cannot arrive
                        # before the waiter is published.
                        self._waiters.append(waiter)
                    cmd_to_send = cmd_bytes
                elif self._recovery_started_s is not None:
                    # Keep policy targets updated, but never issue a second
                    # STOP or another MOVE until fresh status proves idle.
                    next_pos = max(
                        last_pos_poll_t + self._POS_POLL_INTERVAL_S,
                        self._recovery_pos_query_sent_s + self._RECOVERY_QUERY_RETRY_S
                        if self._recovery_pos_query_sent_s else 0.0,
                    )
                    next_flags = max(
                        last_flags_poll_t + self._RECOVERY_FLAGS_POLL_INTERVAL_S,
                        self._recovery_flags_query_sent_s + self._RECOVERY_QUERY_RETRY_S
                        if self._recovery_flags_query_sent_s else 0.0,
                    )
                    if now >= next_pos:
                        cmd_to_send = b"POS?\n"
                        last_pos_poll_t = now
                        self._recovery_pos_query_sent_s = now
                    elif now >= next_flags:
                        cmd_to_send = b"SYSFLAGS?\n"
                        last_flags_poll_t = now
                        self._recovery_flags_query_sent_s = now
                    else:
                        next_poll = min(
                            next_pos,
                            next_flags,
                            self._recovery_started_s + self._RECOVERY_TIMEOUT_S,
                        )
                        self._cond.wait(timeout=max(0.0, next_poll - now))
                        continue
                else:
                    target_dirty = self._target_dirty_locked()
                    open_retry_due = self._open_retry_due_locked(
                        now, position_mm, position_stamp
                    )
                    if open_retry_due and self._open_retry_count >= 1:
                        with self._state_lock:
                            if self._io_error is None:
                                self._io_error = (
                                    f"open target {self.GRIPPER_MAX_MM:.1f} mm not reached "
                                    f"after retry (measured {position_mm:.1f} mm)"
                                )
                        open_retry_due = False

                    if target_dirty and self._move_active and not self._stop_pending and not self._motion_faulted:
                        self._stop_pending = True
                        self._stop_acknowledged = False
                        self._stop_sent_s = now
                        cmd_to_send = b"STOP()\n"
                    elif (
                        (target_dirty or open_retry_due)
                        and not self._stop_pending
                        and not self._move_active
                        and not self._motion_faulted
                        and (now - self._last_move_send_t) >= self._MIN_MOVE_INTERVAL_S
                    ):
                        target = self._target_mm
                        self._open_retry_count = 0 if target_dirty else 1
                        self._last_sent_target_mm = target
                        self._force_move_pending = False
                        self._last_move_send_t = now
                        self._move_active = True
                        cmd_to_send = self._move_cmd(target)
                    elif getattr(self, "_poll_position", True) and (now - last_pos_poll_t) >= self._POS_POLL_INTERVAL_S:
                        cmd_to_send = b"POS?\n"
                        last_pos_poll_t = now
                    else:
                        next_move = (
                            self._last_move_send_t + self._MIN_MOVE_INTERVAL_S
                            if (target_dirty or open_retry_due) and not self._move_active and not self._stop_pending
                            else float("inf")
                        )
                        next_stop_timeout = (
                            self._stop_sent_s + self._STOP_RESPONSE_TIMEOUT_S
                            if self._stop_pending else float("inf")
                        )
                        # Calibration uses AUTOSEND after HOME. Firmware can
                        # reject POS? while HOME is in progress.
                        next_poll = (last_pos_poll_t if getattr(self, "_poll_position", True) else now) + self._POS_POLL_INTERVAL_S
                        timeout_s = max(0.0, min(next_move, next_stop_timeout, next_poll) - now)
                        self._cond.wait(timeout=timeout_s)
                        continue

            if recovery_event is not None:
                state, reason = recovery_event
                self._emit_event("recovery", state=state, reason=reason)
            if cmd_to_send is not None:
                self._send_raw(cmd_to_send)

    def _target_dirty_locked(self) -> bool:
        """True for a forced request or a meaningful streaming target change.
        Caller must hold ``_cond``."""
        if self._target_mm is None:
            return False
        if self._force_move_pending or self._last_sent_target_mm is None:
            return True
        return abs(self._target_mm - self._last_sent_target_mm) >= self._TARGET_CHANGE_THRESH_MM

    def _open_retry_due_locked(
        self, now: float, position_mm: float | None, position_stamp: float
    ) -> bool:
        """Retry only a stalled full-open move, never a grasp against an object."""
        if (
            self._target_mm != self.GRIPPER_MAX_MM
            or self._last_sent_target_mm != self.GRIPPER_MAX_MM
            or self._move_active
            or position_mm is None
            or now - position_stamp > self._OPEN_POSITION_MAX_AGE_S
        ):
            return False
        if position_mm >= self.GRIPPER_MAX_MM - self._OPEN_POSITION_TOLERANCE_MM:
            self._open_retry_count = 0
            return False
        return now - self._last_move_send_t >= self._OPEN_RETRY_DELAY_S

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _clip_target(cls, position_mm: float) -> float:
        return float(min(max(position_mm, cls.GRIPPER_MIN_MM), cls.GRIPPER_MAX_MM))

    @classmethod
    def _move_cmd(cls, target_mm: float) -> bytes:
        return f"MOVE({target_mm:.2f},{cls.MOVE_SPEED_MM_S:.1f})\n".encode()

    def _send_raw(self, data: bytes) -> bool:
        try:
            with self._send_lock:
                with self._cond:
                    if data.startswith(b"MOVE(") and getattr(self, "_motion_suspended", False):
                        return False
                self._emit_event("tx", text=data.decode().rstrip())
                self._sock.sendall(data)
            return True
        except OSError as e:
            with self._state_lock:
                self._io_error = f"send failed: {e}"
            if self.do_print:
                print(f"{self.name}: [WSG] send error: {e}")
            return False

    def _await_command(
        self,
        cmd: bytes,
        expected: bytes,
        timeout_s: float | None = None,
    ) -> bool:
        timeout_s = self.DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        waiter = _Waiter(expected=expected)
        with self._cond:
            self._cmd_queue.append((cmd, waiter))
            self._cond.notify()

        if not waiter.event.wait(timeout=timeout_s):
            with self._state_lock:
                self._io_error = f"timeout waiting for {expected!r}"
            with self._cond:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
            if self.do_print:
                print(f"{self.name}: [WSG] timeout waiting for {expected!r}")
            return False
        return waiter.succeeded

    def _fire_and_forget(self, cmd: bytes, expected: bytes) -> threading.Thread:
        t = threading.Thread(
            target=lambda: self._await_command(cmd, expected),
            daemon=True,
        )
        t.start()
        return t
