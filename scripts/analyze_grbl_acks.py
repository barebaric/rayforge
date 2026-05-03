#!/usr/bin/env python3
"""Analyze GRBL serial logs to identify unacked commands.

Reads a rayforge session log and tracks the send/ack lifecycle of every
gcode command, reporting which ones were never acknowledged by GRBL.

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
    command: str
    byte_len: int
    acked: bool = False
    ack_line: int | None = None
    ack_timestamp: str | None = None


@dataclass
class AckEvent:
    line: int
    timestamp: str
    freed: int
    queue_size: int
    matched_cmd: PendingCmd | None = None


@dataclass
class Report:
    pending: deque[PendingCmd] = field(default_factory=deque)
    all_cmds: list[PendingCmd] = field(default_factory=list)
    acks: list[AckEvent] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    buf_tracking: list[tuple[int, str, int, int]] = field(default_factory=list)
    job_start_line: int | None = None
    job_end_line: int | None = None


RE_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
RE_USER_CMD = re.compile(rf"^{RE_TS}.*USER_COMMAND.*- (.+)$")
RE_TX_RAW = re.compile(
    rf"^{RE_TS}.*\[RAW_IO\].*TX: b'(.+?)'"
    r" \(buf: (\d+)/(\d+)\)"
)
RE_PROCESSED_OK = re.compile(
    rf"^{RE_TS}.*Processed 'ok', freed (\d+) bytes"
    r" \(buf: (\d+)/(\d+)"
)
RE_ERROR = re.compile(
    rf"^{RE_TS}.*Extracted 'error:(\d+)' from raw buffer"
)
RE_STATUS_POLL = re.compile(rf"^{RE_TS}.*STATUS_POLL.*- (<.+>)")
RE_TIMEOUT = re.compile(rf"^{RE_TS}.*Timeout waiting for buffer space")
RE_JOB_START = re.compile(r"Starting job")
RE_JOB_END = re.compile(r"_job_running = False|G-code streaming finished")


def parse(path: str) -> Report:
    r = Report()

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")

            m = RE_USER_CMD.match(line)
            if m:
                ts = m.group(1)
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
                display = payload.replace("\\n", "").replace("\\r", "")
                stripped = display.strip()
                r.buf_tracking.append((lineno, ts, used, total))
                if stripped in ("?", "~", "!", "\x18"):
                    continue
                cmd_bytes = len(payload.encode().decode("unicode_escape"))
                p = PendingCmd(
                    line=lineno,
                    timestamp=ts,
                    command=display,
                    byte_len=cmd_bytes,
                )
                r.pending.append(p)
                r.all_cmds.append(p)
                continue

            m = RE_PROCESSED_OK.match(line)
            if m:
                ts, freed, used, total = (
                    m.group(1),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                )
                matched = None
                if r.pending:
                    matched = r.pending.popleft()
                    matched.acked = True
                    matched.ack_line = lineno
                    matched.ack_timestamp = ts
                else:
                    r.anomalies.append(
                        f"L{lineno} {ts}: ok (freed {freed} bytes) "
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
                        f"'{leaked.command}' (L{leaked.line})"
                    )
                continue

            m = RE_TIMEOUT.match(line)
            if m:
                ts = m.group(1)
                r.anomalies.append(f"L{lineno} {ts}: buffer space timeout")

            if RE_JOB_END.search(line):
                r.job_end_line = r.job_end_line or lineno

    return r


def fmt_report(r: Report):
    total = len(r.all_cmds)
    acked = sum(1 for c in r.all_cmds if c.acked)
    unacked = [c for c in r.all_cmds if not c.acked]

    print(f"Commands sent:    {total}")
    print(f"Acked (ok):       {acked}")
    print(f"Unacked:          {len(unacked)}")
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
                f"  L{c.line} {c.timestamp}  {c.command}  ({c.byte_len} bytes)"
            )
        print()

    pending_bytes = sum(c.byte_len for c in unacked)
    if pending_bytes:
        print(f"  Total orphaned buffer: {pending_bytes} bytes")

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

    unacked = [c for c in r.all_cmds if not c.acked]
    if unacked:
        print()
        print("=" * 60)
        print("COMMANDS NOT ACKNOWLEDGED BY GRBL:")
        print("=" * 60)
        for c in unacked:
            print(f"  [{c.timestamp}] line {c.line}: {c.command}")
        print(f"\n  {len(unacked)} command(s) never received an ack.")
        sys.exit(1)


if __name__ == "__main__":
    main()
