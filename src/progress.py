"""Small terminal progress bar used by long-running lab scripts."""

from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, label: str, total: int, width: int = 28):
        self.label = label
        self.total = max(total, 1)
        self.width = width
        self.start = time.time()
        self.last_len = 0

    def update(self, current: int, detail: str = "") -> None:
        current = max(0, min(current, self.total))
        ratio = current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        suffix = f" | {detail}" if detail else ""
        line = f"\r{self.label} [{bar}] {current}/{self.total} {ratio * 100:5.1f}% {elapsed:6.1f}s{suffix}"
        padding = " " * max(0, self.last_len - len(line))
        sys.stdout.write(line + padding)
        sys.stdout.flush()
        self.last_len = len(line)
        if current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
