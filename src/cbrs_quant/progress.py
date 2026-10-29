"""Dependency-free, throttled console progress for the experiment supervisor."""

from __future__ import annotations

import time


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


class ConsoleProgress:
    def __init__(self, interval=10, enabled=True):
        self.interval, self.enabled = interval, enabled
        self.last_key, self.last_print = None, 0

    def __call__(self, report):
        if not self.enabled:
            return
        active = report.get("active")
        last = report.get("last_result")
        counts = report.get("counts", {})
        completed = sum(counts.values())
        total = report["planned_routes"]
        prefix = f"[{completed}/{total} routes | {report.get('phase', 'SETUP')}]"
        if active:
            state = active.get("state", {})
            phase = state.get("phase", "starting")
            detail = state.get("detail", {})
            key = (active["job"], active["attempt"], phase, detail.get("round"), detail.get("measurement"))
            if key == self.last_key and time.monotonic()-self.last_print < self.interval:
                return
            text = f"{prefix} {active['model']} b={active['batch']} {active['mode']} | {phase}"
            if "total" in detail:
                text += f" {detail.get('completed', 0)}/{detail['total']} images"
            if "round" in detail:
                text += f" {detail['round']}/{detail['rounds']} {detail['measurement']}"
            elif "measurement" in detail:
                text += " " + detail["measurement"]
            text += f" | elapsed {duration(active['elapsed_seconds'])}"
        elif last:
            key = ("result", last["job"], last["status"])
            if key == self.last_key:
                return
            text = f"{prefix} {last['model']} b={last['batch']} {last['mode']} | {last['status'].upper()}"
            if last.get("error"):
                text += " | " + " ".join(last["error"].split())[:160]
                text += f" | details: {last['directory']}"
        else:
            key = ("phase", report.get("phase"), report["status"])
            if key == self.last_key:
                return
            text = f"{prefix} {report['status']}"
        print(text, flush=True)
        self.last_key, self.last_print = key, time.monotonic()
