#!/usr/bin/env python3
"""Analyze GRBL serial logs to identify unacked commands.

Reads a rayforge session log and tracks the send/ack lifecycle of every
gcode command, reporting which ones were never acknowledged by GRBL.

Requires logs produced by rayforge >= 0.9 (where ack lines include
the command text via ``PendingCommand.command``).

Usage:
    python analyze_grbl_acks.py <logfile>
"""

import re
import sys
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PendingCmd:
    line: int
    timestamp: str
    raw_command: str
    byte_len: int
    acked: bool = False
    ack_line: int | None = None
    ack_timestamp: str | None = None
    interactive: bool = False
    cancelled: bool = False

    @property
    def display(self):
        return self.raw_command.replace("\\n", "").replace("\\r", "").strip()


@dataclass
class AckEvent:
    line: int
    timestamp: str
    freed: int
    queue_size: int
    matched_cmd: PendingCmd | None = None


@dataclass
class BufferWaitEvent:
    line: int
    timestamp: str
    buf_used: int
    buf_total: int
    needed: int
    resume_line: int | None = None
    resume_timestamp: str | None = None
    resume_buf_used: int | None = None


@dataclass
class Report:
    pending: deque[PendingCmd] = field(default_factory=deque)
    all_cmds: list[PendingCmd] = field(default_factory=list)
    acks: list[AckEvent] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    buf_tracking: list[tuple[int, str, int, int]] = field(default_factory=list)
    buf_waits: list[BufferWaitEvent] = field(default_factory=list)
    _open_wait: BufferWaitEvent | None = field(default=None, repr=False)
    job_start_line: int | None = None
    job_end_line: int | None = None


RE_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
RE_USER_CMD = re.compile(rf"^{RE_TS}.*USER_COMMAND.*- (.+)$")
RE_TX_RAW = re.compile(
    rf"^{RE_TS}.*\[RAW_IO\].*TX: b'(.+?)'"
    r" \(buf: (\d+)/(\d+)\)"
)
RE_TX_RAW_NOBUF = re.compile(rf"^{RE_TS}.*\[RAW_IO\].*TX: b'(.+?)'$")
RE_PROCESSED_OK = re.compile(
    rf"^{RE_TS}.*Processed 'ok', freed (\d+) bytes"
    r" for '(.+?)'"
    r" \(buf: (\d+)/(\d+)"
)
RE_BUFFER_ACK = re.compile(
    rf"^{RE_TS}.*Buffer ack: freeing (\d+) bytes for '(.+?)'"
)
RE_ERROR = re.compile(rf"^{RE_TS}.*Extracted 'error:(\d+)' from raw buffer")
RE_TIMEOUT = re.compile(rf"^{RE_TS}.*Timeout waiting for buffer space")
RE_BUFFER_STALL = re.compile(
    rf"^{RE_TS}.*Buffer stall: timed out"
    r".*waiting for (\d+) bytes"
    r".*\(buf: (\d+)/(\d+)"
)
RE_BUFFER_WAIT = re.compile(
    rf"^{RE_TS}.*Buffer full \((\d+)/(\d+)\), waiting for (\d+) bytes"
)
RE_BUFFER_RESUME = re.compile(
    rf"^{RE_TS}.*Buffer space available, resuming"
    r".*\(buf: (\d+)/(\d+)\)"
)
RE_QUEUE_CLEARED = re.compile(
    r"Command queue cleared after cancel"
    r"|Deadlock recovery: sending"
)
RE_JOB_START = re.compile(r"Starting GRBL streaming job")
RE_JOB_END = re.compile(
    r"G-code streaming finished"
    r"|G-code streaming cancelled"
    r"|G-code streaming aborted"
    r"|G-code streaming ended unexpectedly"
)


def _payload_bytes(payload: str) -> int:
    return len(payload.encode().decode("unicode_escape"))


def _is_realtime(payload: str) -> bool:
    stripped = payload.replace("\\n", "").replace("\\r", "").strip()
    return stripped in ("?", "~", "!", "\x18")


def parse(path: str) -> Report:
    r = Report()

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")

            m = RE_USER_CMD.match(line)
            if m:
                r.job_start_line = r.job_start_line or lineno
                continue

            m = RE_TX_RAW.match(line)
            if m:
                ts, payload, used, total = (
                    m.group(1),
                    m.group(2),
                    int(m.group(3)),
                    int(m.group(4)),
                )
                r.buf_tracking.append((lineno, ts, used, total))
                if _is_realtime(payload):
                    continue
                p = PendingCmd(
                    line=lineno,
                    timestamp=ts,
                    raw_command=payload,
                    byte_len=_payload_bytes(payload),
                )
                r.pending.append(p)
                r.all_cmds.append(p)
                continue

            m = RE_TX_RAW_NOBUF.match(line)
            if m:
                ts, payload = m.group(1), m.group(2)
                if _is_realtime(payload):
                    continue
                p = PendingCmd(
                    line=lineno,
                    timestamp=ts,
                    raw_command=payload,
                    byte_len=_payload_bytes(payload),
                    interactive=True,
                )
                r.pending.append(p)
                r.all_cmds.append(p)
                continue

            m = RE_PROCESSED_OK.match(line)
            if m:
                ts = m.group(1)
                freed = int(m.group(2))
                ack_cmd = m.group(3)
                used = int(m.group(4))
                total = int(m.group(5))
                matched = None
                if r.pending:
                    matched = r.pending.popleft()
                    if ack_cmd != matched.raw_command:
                        r.anomalies.append(
                            f"L{lineno} {ts}: ack desync - "
                            f"ack references {ack_cmd!r} "
                            f"but queue head is "
                            f"{matched.raw_command!r} "
                            f"(L{matched.line})"
                        )
                    matched.acked = True
                    matched.ack_line = lineno
                    matched.ack_timestamp = ts
                else:
                    r.anomalies.append(
                        f"L{lineno} {ts}: ok for "
                        f"{ack_cmd!r} "
                        f"but pending queue is empty"
                    )
                ev = AckEvent(
                    line=lineno,
                    timestamp=ts,
                    freed=freed,
                    queue_size=0,
                    matched_cmd=matched,
                )
                r.acks.append(ev)
                continue

            m = RE_BUFFER_ACK.match(line)
            if m:
                ts = m.group(1)
                ack_cmd = m.group(2)
                if not r.pending:
                    r.anomalies.append(
                        f"L{lineno} {ts}: buffer ack for "
                        f"{ack_cmd!r} "
                        f"but pending queue is empty"
                    )
                continue

            m = RE_ERROR.match(line)
            if m:
                ts, err_code = m.group(1), int(m.group(2))
                if not r.pending:
                    r.anomalies.append(
                        f"L{lineno} {ts}: error:{err_code} "
                        f"but pending queue is empty"
                    )
                else:
                    leaked = r.pending.popleft()
                    r.anomalies.append(
                        f"L{lineno} {ts}: error:{err_code} "
                        f"popped unacked "
                        f"{leaked.raw_command!r} (L{leaked.line})"
                    )
                continue

            m = RE_TIMEOUT.match(line)
            if m:
                ts = m.group(1)
                r.anomalies.append(f"L{lineno} {ts}: buffer space timeout")

            m = RE_BUFFER_STALL.match(line)
            if m:
                ts = m.group(1)
                needed = int(m.group(2))
                used = int(m.group(3))
                total = int(m.group(4))
                r.anomalies.append(
                    f"L{lineno} {ts}: buffer stall - "
                    f"timed out waiting for {needed} bytes "
                    f"(buf: {used}/{total})"
                )

            m = RE_BUFFER_WAIT.match(line)
            if m:
                ts = m.group(1)
                buf_used = int(m.group(2))
                buf_total = int(m.group(3))
                needed = int(m.group(4))
                ev = BufferWaitEvent(
                    line=lineno,
                    timestamp=ts,
                    buf_used=buf_used,
                    buf_total=buf_total,
                    needed=needed,
                )
                r.buf_waits.append(ev)
                r._open_wait = ev

            m = RE_BUFFER_RESUME.match(line)
            if m:
                ts = m.group(1)
                resume_used = int(m.group(2))
                if r._open_wait:
                    r._open_wait.resume_line = lineno
                    r._open_wait.resume_timestamp = ts
                    r._open_wait.resume_buf_used = resume_used
                    r._open_wait = None

            if RE_QUEUE_CLEARED.search(line):
                while r.pending:
                    r.pending.popleft().cancelled = True

            if RE_JOB_END.search(line):
                r.job_end_line = r.job_end_line or lineno

    return r


def fmt_report(r: Report):
    streaming = [
        c for c in r.all_cmds if not c.interactive and not c.cancelled
    ]
    total = len(streaming)
    acked = sum(1 for c in streaming if c.acked)
    cancelled = sum(1 for c in r.all_cmds if c.cancelled and not c.interactive)
    unacked = [c for c in streaming if not c.acked]

    print(f"Commands sent:    {total}")
    print(f"Acked (ok):       {acked}")
    print(f"Unacked:          {len(unacked)}")
    if cancelled:
        print(f"Cancelled:        {cancelled}")
    print()

    if r.anomalies:
        print("=== Anomalies ===")
        for a in r.anomalies:
            print(f"  {a}")
        print()

    if unacked:
        print("=== Unacked Commands ===")
        for c in unacked:
            print(
                f"  L{c.line} {c.timestamp}  {c.display}  ({c.byte_len} bytes)"
            )
        print()

    pending_bytes = sum(c.byte_len for c in unacked)
    if pending_bytes:
        print(f"  Total orphaned buffer: {pending_bytes} bytes")

    if r.buf_waits:
        print()
        unresolved = [w for w in r.buf_waits if w.resume_line is None]
        print(
            "=== Buffer Waits "
            f"({len(r.buf_waits)} total, "
            f"{len(unresolved)} unresolved) ==="
        )
        for w in r.buf_waits[-50:]:
            status = (
                f"resumed L{w.resume_line} "
                f"(buf: {w.resume_buf_used}/{w.buf_total})"
                if w.resume_line
                else "UNRESOLVED"
            )
            print(
                f"  L{w.line:<6} {w.timestamp}  "
                f"wait {w.needed}B "
                f"(buf: {w.buf_used}/{w.buf_total})  "
                f"{status}"
            )

    print()
    print("=== Buffer Timeline (last 200 entries) ===")
    for lineno, ts, used, total in r.buf_tracking[-200:]:
        bar_len = total
        filled = int(bar_len * used / total) if total else 0
        bar = "|" + "#" * filled + "-" * (bar_len - filled) + "|"
        print(f"  L{lineno:<6} {ts} {bar} {used}/{total}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <logfile>", file=sys.stderr)
        sys.exit(1)
    r = parse(sys.argv[1])
    fmt_report(r)

    unacked = [
        c
        for c in r.all_cmds
        if not c.acked and not c.interactive and not c.cancelled
    ]
    if unacked:
        print()
        print("=" * 60)
        print("COMMANDS NOT ACKNOWLEDGED BY GRBL:")
        print("=" * 60)
        for c in unacked:
            print(f"  [{c.timestamp}] line {c.line}: {c.display}")
        print(f"\n  {len(unacked)} command(s) never received an ack.")
        sys.exit(1)


if __name__ == "__main__":
    main()
